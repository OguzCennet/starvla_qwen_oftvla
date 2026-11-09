#!/bin/bash
#SBATCH --account=y307bd
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=/hnvme/workspace/y307bd13-vgv/runs/oftvla/%x/slurm_%j.out
# Smoke test for train_e3_text.sbatch with the EXACT run configuration (full v5 train set, 16 x 8
# accumulation, same workers), only shorter: 40 steps, interrupted after MAX_H hours and resumed.
# Checks: data loads (demo/sample counts), loss falls, peak GPU memory with the long MAS prompts, resume.
# Usage: mkdir -p /hnvme/workspace/y307bd13-vgv/runs/oftvla/smoke_MAS   (FIRST: Slurm needs the log folder at job start)
#        sbatch --job-name=smoke_MAS scripts/smoke_text.sbatch MAS [MAX_H] [TEXTDIR] [SEED]
#   TEXTDIR = folder with train.csv (default $TEXTS), SEED default 42 (= train.py default), as in train_e3_text.sbatch.
source $HOME/VLA_FT/starvla_qwen_oftvla/scripts/common.sh
cd $REPO
SETUP=${1:?usage: sbatch --job-name=NAME smoke_text.sbatch A0|AS|MAS [MAX_H] [TEXTDIR] [SEED]}; MAX_H=${2:-0.03}
TEXTDIR=${3:-$TEXTS}; SEED=${4:-42}
echo "SETUP=$SETUP MAX_H=$MAX_H TEXTDIR=$TEXTDIR SEED=$SEED"
md5sum $TEXTDIR/train.csv
mkdir -p $RUNS/$SLURM_JOB_NAME
git -C $REPO rev-parse --short HEAD
nvidia-smi --query-gpu=name,memory.total --format=csv
ARGS="--data_root $DATA --train_ids $SPLITS/e3_train_ids.csv --forbidden_ids $SPLITS/e3_testdemo_ids.csv
      --text_csv $TEXTDIR/train.csv --setup $SETUP --seed $SEED --base_vlm $BASE_VLM --run_dir $RUNS/$SLURM_JOB_NAME --num_workers 14
      --micro_batch 16 --grad_accum 8 --max_steps 40 --warmup_steps 5000
      --log_every 5 --eval_every 20 --save_every 100000 --resume_every 100000"
run_py -m oftvla.train $ARGS --max_hours $MAX_H   # part 1: stops after MAX_H h of training, saves resume state
run_py -m oftvla.train $ARGS                    # part 2: resumes and finishes step 40
