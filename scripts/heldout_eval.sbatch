#!/bin/bash
#SBATCH --account=y307bd
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/hnvme/workspace/y307bd13-vgv/runs/oftvla/%x/heldout_eval_%j.out
# Held-out test demos (E3 test, A0ASMAS test.csv, 1,158 demos, every frame) for one run, all checkpoints.
# The text column is the run's training setup (from its args.json). Writes RUN/heldout_eval/.
# Usage: sbatch --job-name=e3v5_MAS scripts/heldout_eval.sbatch      (job name = run dir, which exists)
source $HOME/VLA_FT/starvla_qwen_oftvla/scripts/common.sh
cd $REPO
git -C $REPO rev-parse --short HEAD
git -C $REPO status --short
md5sum $TEXTS/test.csv
run_py -m oftvla.heldout_eval --run_dir $RUNS/$SLURM_JOB_NAME --ckpts 30000 20000 10000 \
  --data_root $DATA --test_ids $SPLITS/e3_testdemo_ids.csv --text_csv $TEXTS/test.csv \
  --base_vlm $BASE_VLM --stride 1 --batch_size 64 --num_workers 14
