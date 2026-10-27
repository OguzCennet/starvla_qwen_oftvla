#!/bin/bash
#SBATCH --job-name=oftvla_e3_A0
#SBATCH --account=y307bd
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/hnvme/workspace/y307bd13-vgv/runs/oftvla/e3_A0/slurm_%j.out
# Full run: 76 E3 training tasks, A0, StarVLA LIBERO recipe (30K steps x 128 = 32 x 4 accumulation).
# Resumable: stops after 23 h, saves resume.pt; the next job in the chain continues.
source $HOME/VLA_FT/starvla_qwen_oftvla/scripts/common.sh
cd $REPO
git -C $REPO rev-parse --short HEAD   # code version in the log
nvidia-smi --query-gpu=name,memory.total --format=csv
run_py -m oftvla.train \
  --data_root $DATA --train_ids $SPLITS/e3_train_ids.csv --forbidden_ids $SPLITS/e3_testdemo_ids.csv \
  --labels_csv $LABELS --setup A0 --base_vlm $BASE_VLM --run_dir $RUNS/e3_A0 --num_workers 14 \
  --micro_batch 32 --grad_accum 4 --max_steps 30000 --warmup_steps 5000 \
  --log_every 10 --eval_every 100 --save_every 10000 --resume_every 1000 --max_hours 23
