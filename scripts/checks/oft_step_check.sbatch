#!/bin/bash
#SBATCH --job-name=oft_step_check
#SBATCH --account=y307bd
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/home/hpc/y307bd/y307bd13/VLA_FT/oftvla_checks/oft_step_check_%j.out
# OFT training path unchanged: 3 optimizer steps of the exact e3v5_MAS config (16 x 8) with the code of an old
# commit (git worktree OLD_WT) and twice with the current code; compare loss / grad_norm per step.
# Usage: sbatch scripts/checks/oft_step_check.sbatch OLD_WT
source $HOME/VLA_FT/starvla_qwen_oftvla/scripts/common.sh
OLD_WT=${1:?worktree of the old commit}
cd $REPO
echo "new: $(git rev-parse --short HEAD) $(git status --short | wc -l) changed; old: $(git -C $OLD_WT rev-parse --short HEAD)"
for tag in old new new2; do
  code=$REPO; [ $tag = old ] && code=$OLD_WT
  dir=$RUNS/check_oft_step_$tag; rm -rf "${dir:?}"
  echo "### $tag ($code)"
  (cd $code && apptainer exec --nv --env PYTHONPATH=$code:$PYLIBS $SIF python -m oftvla.train \
    --data_root $DATA --train_ids $SPLITS/e3_train_ids.csv --forbidden_ids $SPLITS/e3_testdemo_ids.csv \
    --text_csv $TEXTS/train.csv --setup MAS --base_vlm $BASE_VLM --run_dir $dir --num_workers 14 \
    --micro_batch 16 --grad_accum 8 --max_steps 3 --warmup_steps 5000 \
    --log_every 1 --eval_every 1000 --save_every 1000 --resume_every 1000) 2>&1 | grep -E '^\{"step"|^\[data\] setup|Error|Traceback'
  rm -f $dir/steps_3_model.pt
done
