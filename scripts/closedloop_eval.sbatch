#!/bin/bash
#SBATCH --account=y307bd
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=/hnvme/workspace/y307bd13-vgv/runs/oftvla/closedloop_logs/%x_%j.out
# M15 closed-loop eval with the oracle switcher + scorer (oftvla/closedloop_client.py). Same server / client
# split and containers as eval_libero.sbatch: policy server (starvla.sif) + LIBERO client (sym-openvla-oft.sif).
# Usage: sbatch --job-name=NAME scripts/closedloop_eval.sbatch RUN STEP IDS_CSV OUT_DIR SETUP TEXTS [--log_prompts]
#   RUN = run dir name under runs/oftvla, SETUP = A0 | AS | MAS (prompt text), TEXTS = spec | labels
#   OUT_DIR is resumable (episodes.csv: done (task, init) pairs are skipped); a new result needs a new OUT_DIR.
source $HOME/VLA_FT/starvla_qwen_oftvla/scripts/common.sh
RUN=$1; STEP=$2; IDS=$3; OUT=$4; SETUP=$5; TEXTS=$6; EXTRA=$7
mkdir -p $OUT
LIBERO_SIF=/hnvme/workspace/y307bd13-vgv/containers/sym-openvla-oft.sif
SPEC=$HOME/VLA_FT/DataSamples/Libero/ClosedLoop/switch_spec.json
PORT=$((10000 + SLURM_JOB_ID % 20000)); READY=$OUT/.ready_$SLURM_JOB_ID
cd $REPO; git -C $REPO rev-parse --short HEAD; git -C $REPO status --short
echo "RUN=$RUN STEP=$STEP IDS=$IDS OUT=$OUT SETUP=$SETUP TEXTS=$TEXTS"; md5sum $SPEC $IDS

run_py -m oftvla.policy --run_dir $RUNS/$RUN --step $STEP --base_vlm $BASE_VLM --port $PORT --ready_file $READY &
SERVER=$!
trap "kill $SERVER 2>/dev/null; rm -f $READY" EXIT
for i in $(seq 300); do [ -f $READY ] && break; kill -0 $SERVER 2>/dev/null || { echo "server died"; exit 1; }; sleep 2; done
[ -f $READY ] || { echo "server not ready after 10 min"; exit 1; }

MUJOCO_GL=egl apptainer exec --nv --env PYTHONPATH=$REPO $LIBERO_SIF python3 -m oftvla.closedloop_client \
  --ids $IDS --spec $SPEC --setup $SETUP --texts $TEXTS --labels_csv $LABELS --out_dir $OUT --port $PORT $EXTRA
