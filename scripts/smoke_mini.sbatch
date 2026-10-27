#!/bin/bash
#SBATCH --job-name=oftvla_smoke
#SBATCH --account=y307bd
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/hnvme/workspace/y307bd13-vgv/runs/oftvla/%x/slurm_%j.out
# Smoke test (M4): mini set (5 tasks, 10 demos), no accumulation.
# Part 1 stops after MAX_H hours of training and saves resume state; part 2 resumes and finishes.
# Usage: sbatch --job-name=NAME scripts/smoke_mini.sbatch [MICRO_BATCH] [MAX_H]   (mkdir runs/oftvla/NAME first)
source $HOME/VLA_FT/starvla_qwen_oftvla/scripts/common.sh
cd $REPO
MICRO=${1:-16}; MAX_H=${2:-0.01}
OUT=$RUNS/$SLURM_JOB_NAME
ARGS="--data_root $DATA --train_ids $SPLITS/mini_ids.csv --forbidden_ids $SPLITS/e3_testdemo_ids.csv
      --labels_csv $LABELS --base_vlm $BASE_VLM --run_dir $OUT --num_workers 12
      --micro_batch $MICRO --grad_accum 1 --max_steps 300 --warmup_steps 20
      --log_every 10 --eval_every 50 --save_every 300 --resume_every 100000"
nvidia-smi --query-gpu=name,memory.total --format=csv
run_py -m oftvla.train $ARGS --max_hours $MAX_H   # part 1
run_py -m oftvla.train $ARGS                    # part 2 (resumes)
