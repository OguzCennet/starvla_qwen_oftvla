#!/bin/bash
#SBATCH --account=y307bd
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --dependency=singleton
#SBATCH --output=/hnvme/workspace/y307bd13-vgv/runs/oftvla/eval_logs/%x_%j.out
# Held-out eval of one flow-head run (GR00T / PI) at one checkpoint, K=8 samples per frame (oftvla/flow_heldout_eval.py):
# every frame of the 1,158 E3 test demos, text column and test.csv from the run's args.json. Writes RUN/heldout_eval/.
# Normally submitted by scripts/submit_flow_eval.sh (shards + merge). By hand:
#   sbatch --job-name=eval_<RUN>_<STEP>      scripts/flow_heldout_eval.sbatch RUN STEP [-- extra args]          (unsharded)
#   sbatch --job-name=eval_<RUN>_<STEP>_s<i> scripts/flow_heldout_eval.sbatch RUN STEP i N [-- extra args]  (shard i of N)
#   Measured on one H200 (batch 64, K=8, 186,874 frames): GR00T ~35 s / 1,000 frames -> ~1.8 h per checkpoint;
#   PI ~300 s / 1,000 frames (96% in the K head samplings) -> ~15.7 h per checkpoint unsharded.
#   The job name must NOT be the run name (train_flow.sbatch holds singleton on the run name); singleton here
#   allows one job per run x checkpoint x shard at a time.
source $HOME/VLA_FT/starvla_qwen_oftvla/scripts/common.sh
cd $REPO
RUN=${1:?RUN (run dir name)}; STEP=${2:?STEP}; shift 2
SHARD=0; NSHARDS=1; NAME=eval_${RUN}_${STEP}
if [ $# -ge 2 ] && [ "$1" != "--" ]; then SHARD=$1; NSHARDS=$2; NAME=${NAME}_s$SHARD; shift 2; fi
[ "${1:-}" = "--" ] && shift
if [ "$SLURM_JOB_NAME" != "$NAME" ]; then echo "[eval] job name must be $NAME, got $SLURM_JOB_NAME"; exit 1; fi
CKPT=$RUNS/$RUN/steps_${STEP}_model.pt
if [ ! -f $CKPT ]; then echo "[eval] missing $CKPT"; exit 1; fi
git -C $REPO rev-parse --short HEAD
git -C $REPO status --short
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
run_py -m oftvla.flow_heldout_eval --run_dir $RUNS/$RUN --ckpt $STEP --data_root $DATA \
  --test_ids $SPLITS/e3_testdemo_ids.csv --base_vlm $BASE_VLM --K 8 --batch_size 64 --num_workers 14 \
  --shard $SHARD --num_shards $NSHARDS "$@"
