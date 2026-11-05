#!/bin/bash
#SBATCH --job-name=oft_gate
#SBATCH --account=y307bd
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/home/hpc/y307bd/y307bd13/VLA_FT/oftvla_checks/oft_gate_%j.out
# Usage: sbatch scripts/checks/oft_gate.sbatch TAG   (TAG = before | after); log + captures in ~/VLA_FT/oftvla_checks/
source $HOME/VLA_FT/starvla_qwen_oftvla/scripts/common.sh
cd $REPO
TAG=${1:?tag}
mkdir -p $HOME/VLA_FT/oftvla_checks
git -C $REPO rev-parse --short HEAD; git -C $REPO status --short
echo "### check_equivalence (atol 0)"
run_py scripts/check_equivalence.py --starvla_root $HOME/VLA_FT/starVLA --base_vlm $BASE_VLM --atol 0
echo "exit=$?"
echo "### capture"
run_py $REPO/scripts/checks/capture_oft_io.py $HOME/VLA_FT/oftvla_checks/io_$TAG.pt $BASE_VLM $TEXTS/train.csv $HOME/VLA_FT/oftvla_checks/head_state.pt
echo "exit=$?"
