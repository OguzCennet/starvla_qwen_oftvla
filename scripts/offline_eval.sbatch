#!/bin/bash
#SBATCH --job-name=oftvla_offline_eval
#SBATCH --account=y307bd
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=/hnvme/workspace/y307bd13-vgv/runs/oftvla/e3_A0/offline_eval_%j.out
# Held-out demo error (val) vs training demo error at fixed checkpoints. Writes e3_A0/offline_eval.json.
source $HOME/VLA_FT/starvla_qwen_oftvla/scripts/common.sh
cd $REPO
git -C $REPO rev-parse --short HEAD
run_py -m oftvla.offline_eval --run_dir $RUNS/e3_A0 --ckpts 10000 20000 30000 \
  --data_root $DATA --train_ids $SPLITS/e3_train_ids.csv --val_ids $SPLITS/e3_val_ids.csv \
  --forbidden_ids $SPLITS/e3_testdemo_ids.csv --labels_csv $LABELS --base_vlm $BASE_VLM
