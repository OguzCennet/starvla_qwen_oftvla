#!/bin/bash
#SBATCH --account=y307bd
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=01:00:00
#SBATCH --output=/hnvme/workspace/y307bd13-vgv/runs/oftvla/eval_logs/%x_%j.out
# Merge the shard outputs of one flow-head held-out eval (no GPU, no model): writes RUN/heldout_eval/step<STEP>_*,
# checks that every frame is covered exactly once, removes the shard files. Submitted by scripts/submit_flow_eval.sh.
# Usage: sbatch --job-name=eval_<RUN>_<STEP>_merge scripts/flow_heldout_merge.sbatch RUN STEP N [-- extra args]
source $HOME/VLA_FT/starvla_qwen_oftvla/scripts/common.sh
cd $REPO
RUN=${1:?RUN}; STEP=${2:?STEP}; NSHARDS=${3:?N}; shift 3
[ "${1:-}" = "--" ] && shift
if [ "$SLURM_JOB_NAME" != "eval_${RUN}_${STEP}_merge" ]; then
  echo "[merge] job name must be eval_${RUN}_${STEP}_merge, got $SLURM_JOB_NAME"; exit 1; fi
git -C $REPO rev-parse --short HEAD
apptainer exec --env PYTHONPATH=$REPO:$PYLIBS $SIF python -m oftvla.flow_heldout_eval --run_dir $RUNS/$RUN \
  --ckpt $STEP --data_root $DATA --test_ids $SPLITS/e3_testdemo_ids.csv --base_vlm $BASE_VLM --K 8 \
  --merge --num_shards $NSHARDS "$@"
