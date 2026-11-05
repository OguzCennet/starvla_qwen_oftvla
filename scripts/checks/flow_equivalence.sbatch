#!/bin/bash
#SBATCH --job-name=flow_eq
#SBATCH --account=y307bd
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:30:00
#SBATCH --output=/home/hpc/y307bd/y307bd13/VLA_FT/oftvla_checks/flow_eq_%j.out
# Full-size GR00T / PI equivalence vs the fork (bf16, 1 H200). Usage: sbatch scripts/checks/flow_equivalence.sbatch gr00t|pi [--deterministic]
source $HOME/VLA_FT/starvla_qwen_oftvla/scripts/common.sh
cd $REPO
HEAD=${1:?gr00t|pi}
shift  # further args go to the script, e.g. --deterministic
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # only used with --deterministic
mkdir -p $HOME/VLA_FT/oftvla_checks
git -C $REPO rev-parse --short HEAD; git -C $REPO status --short
nvidia-smi --query-gpu=name,memory.total --format=csv
run_py scripts/checks/flow_equivalence.py --head $HEAD --out $HOME/VLA_FT/oftvla_checks/flow_eq_${HEAD}${1:+_det}_$SLURM_JOB_ID.json \
  --starvla_root $HOME/VLA_FT/starVLA --base_vlm $BASE_VLM --data_root $DATA --val_ids $SPLITS/e3_val_ids.csv \
  --text_csv $TEXTS/val.csv --stats $RUNS/e3v5_A0/dataset_statistics.json "$@"
echo "exit=$?"
