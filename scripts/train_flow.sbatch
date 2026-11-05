#!/bin/bash
#SBATCH --account=y307bd
#SBATCH --partition=h200
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --output=/hnvme/workspace/y307bd13-vgv/runs/oftvla/%x/slurm_%j.out
# Multi-node GR00T / PI training (oftvla.train_flow) with accelerate + DeepSpeed ZeRO-2 bf16, as the fork launches it
# (released PI slurm_script: srun ... accelerate launch --main_process_ip --machine_rank --num_machines).
# Usage: sbatch -N NODES --job-name=RUN_DIR_NAME scripts/train_flow.sbatch HEAD SETUP [TEXTDIR] [-- extra train_flow args]
#   GR00T: -N 2 (8 GPUs x 16), PI: -N 4 (16 GPUs x 8). Run dir = runs/oftvla/<job name>.
#   Env: OFTVLA_HOST_IB=1 binds the host's InfiniBand verbs libraries into the container (the image has none);
#        OFTVLA_NCCL_TEST=1 runs the NCCL bandwidth preflight first (sockets, and IB if OFTVLA_HOST_IB=1).
#        OFTVLA_SEGMENTS="150 300": run the training command once per stop step in this job (resume test).
source $HOME/VLA_FT/starvla_qwen_oftvla/scripts/common.sh
export REPO SIF BASE_VLM DATA SPLITS TEXTS RUNS PYLIBS  # used by the shell inside the container
cd $REPO
export HEAD=${1:?HEAD gr00t|pi}; export SETUP=${2:?SETUP A0|AS|MAS}; shift 2
export TEXTDIR=$TEXTS
if [ $# -gt 0 ] && [ "$1" != "--" ]; then TEXTDIR=$1; shift; fi  # optional TEXTDIR
[ "${1:-}" = "--" ] && shift
export EXTRA="$*"
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export NCCL_DEBUG=${NCCL_DEBUG:-INFO} NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ibs9}  # IPoIB interface for bootstrap / socket transport
export RUN_DIR=$RUNS/$SLURM_JOB_NAME
echo "HEAD=$HEAD SETUP=$SETUP TEXTDIR=$TEXTDIR NODES=$SLURM_NNODES ($SLURM_JOB_NODELIST) EXTRA=$EXTRA"
git -C $REPO rev-parse --short HEAD; git -C $REPO status --short
md5sum $TEXTDIR/train.csv
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

IB_BIND=""
if [ "${OFTVLA_HOST_IB:-0}" = "1" ]; then
  for l in libibverbs.so.1 libmlx5.so.1 librdmacm.so.1 libnl-3.so.200 libnl-route-3.so.200; do
    IB_BIND="$IB_BIND --bind $(readlink -f /usr/lib64/$l):/opt/hostib/$l"; done
  IB_BIND="$IB_BIND --bind /usr/lib64/libibverbs:/usr/lib64/libibverbs --bind /etc/libibverbs.d:/etc/libibverbs.d"
fi
export IB_BIND
in_container() {  # $1 = extra binds, rest = command run in bash inside the container on every node
  local binds="$1"; shift
  srun --kill-on-bad-exit=1 apptainer exec --nv $binds --env PYTHONPATH=$REPO:$PYLIBS $SIF bash -c "$*"
}
SETLIB='[ -d /opt/hostib ] && export LD_LIBRARY_PATH=/opt/hostib:$LD_LIBRARY_PATH;'
TORCHRUN='torchrun --nnodes $SLURM_NNODES --nproc_per_node 4 --node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port $MASTER_PORT'

if [ "${OFTVLA_NCCL_TEST:-0}" = "1" ]; then
  echo "### NCCL preflight: sockets (NCCL_IB_DISABLE=1)"
  in_container "" "NCCL_IB_DISABLE=1 $TORCHRUN scripts/checks/nccl_bw.py" 2>&1 | grep -E "^\[nccl\]|NET/|Using network|rror|Traceback|failed" | head -40
  if [ -n "$IB_BIND" ]; then
    echo "### NCCL preflight: InfiniBand (host verbs libraries bound)"
    in_container "$IB_BIND" "$SETLIB $TORCHRUN scripts/checks/nccl_bw.py" 2>&1 | grep -E "^\[nccl\]|NET/|Using network|IB|rror|Traceback|failed" | head -40
  fi
fi

LAUNCH='accelerate launch --config_file configs/deepspeed/deepspeed_zero2.yaml --main_process_ip $MASTER_ADDR
  --main_process_port $MASTER_PORT --machine_rank $SLURM_PROCID --num_machines $SLURM_NNODES
  --num_processes $((4 * SLURM_NNODES)) -m oftvla.train_flow --head $HEAD --setup $SETUP --run_dir $RUN_DIR
  --data_root $DATA --train_ids $SPLITS/e3_train_ids.csv --forbidden_ids $SPLITS/e3_testdemo_ids.csv
  --text_csv $TEXTDIR/train.csv --base_vlm $BASE_VLM --num_workers 12'
LAUNCH=$(echo "$LAUNCH" | tr '\n' ' ')  # one line: bash -c would treat newlines as command separators
for STOP in ${OFTVLA_SEGMENTS:-full}; do
  echo "### train segment: $STOP"
  if [ "$STOP" = full ]; then SEG=""; else SEG="--exit_after_step $STOP"; fi
  in_container "$IB_BIND" "$SETLIB $LAUNCH $EXTRA $SEG"
  echo "exit=$?"
  if [ -d $RUN_DIR/resume ]; then
    echo "[disk] resume state: $(du -sb $RUN_DIR/resume | cut -f1) bytes, $(find $RUN_DIR/resume -type f | wc -l) files"
  fi
done
for f in $RUN_DIR/steps_*_model.pt; do [ -f $f ] && echo "[disk] $(basename $f): $(stat -c %s $f) bytes"; done
