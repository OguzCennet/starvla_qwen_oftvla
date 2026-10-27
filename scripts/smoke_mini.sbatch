#!/bin/bash
#SBATCH --job-name=oftvla_smoke
#SBATCH --account=y307bd
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/hnvme/workspace/y307bd13-vgv/runs/oftvla/smoke_mini/slurm_%j.out
# Smoke test (M4): mini set (5 tasks, 10 demos), no accumulation.
# Part 1 stops after ~4 min and saves resume state; part 2 resumes and finishes 300 steps.
source $HOME/VLA_FT/starvla_qwen_oftvla/scripts/common.sh
cd $REPO
OUT=$RUNS/smoke_mini
ARGS="--data_root $DATA --train_ids $SPLITS/mini_ids.csv --forbidden_ids $SPLITS/e3_testdemo_ids.csv
      --labels_csv $LABELS --base_vlm $BASE_VLM --run_dir $OUT --num_workers 12
      --micro_batch 16 --grad_accum 1 --max_steps 300 --warmup_steps 20
      --log_every 10 --eval_every 50 --save_every 300 --resume_every 100000"
nvidia-smi --query-gpu=name,memory.total --format=csv
run_py -m oftvla.train $ARGS --max_hours 0.07   # part 1
run_py -m oftvla.train $ARGS                    # part 2 (resumes)
