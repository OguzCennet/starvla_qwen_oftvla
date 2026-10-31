#!/bin/bash
#SBATCH --account=y307bd
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/hnvme/workspace/y307bd13-vgv/runs/oftvla/%x/slurm_%j.out
# Full run on the v5 demo set with per-frame texts from the A0ASMAS CSV (setup A0 | AS | MAS).
# Identical to train_e3_A0.sbatch except the text source (--text_csv instead of --labels_csv).
# Run dir = runs/oftvla/<job name>. Resumable: stops after 23 h, the next job in the chain continues.
# Usage: bash scripts/submit_chain.sh scripts/train_e3_text.sbatch 3 e3v5_MAS MAS [TEXTDIR] [SEED]
#   TEXTDIR = folder with train.csv (default $TEXTS), SEED default 42 (= train.py default).
source $HOME/VLA_FT/starvla_qwen_oftvla/scripts/common.sh
cd $REPO
SETUP=${1:?usage: sbatch --job-name=NAME train_e3_text.sbatch A0|AS|MAS [TEXTDIR] [SEED]}
TEXTDIR=${2:-$TEXTS}
SEED=${3:-42}
echo "SETUP=$SETUP TEXTDIR=$TEXTDIR SEED=$SEED"
git -C $REPO rev-parse --short HEAD   # code version in the log
git -C $REPO status --short           # uncommitted changes would show here
md5sum $TEXTDIR/train.csv
nvidia-smi --query-gpu=name,memory.total --format=csv
run_py -m oftvla.train \
  --data_root $DATA --train_ids $SPLITS/e3_train_ids.csv --forbidden_ids $SPLITS/e3_testdemo_ids.csv \
  --text_csv $TEXTDIR/train.csv --setup $SETUP --seed $SEED --base_vlm $BASE_VLM --run_dir $RUNS/$SLURM_JOB_NAME --num_workers 14 \
  --micro_batch 16 --grad_accum 8 --max_steps 30000 --warmup_steps 5000 \
  --log_every 10 --eval_every 100 --save_every 10000 --resume_every 1000 --max_hours 23
