#!/bin/bash
#SBATCH --account=y307bd
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --output=/hnvme/workspace/y307bd13-vgv/runs/oftvla/closedloop_validation/replay_%x_%j.out
# M15 Step 2: replay the 1,158 E3 test demos with the oracle switcher attached (oftvla/replay_switcher.py).
# No rendering, CPU only (the GPU is the partition's unit). Usage: sbatch --job-name=switch_MODE scripts/replay_switcher.sbatch MODE
#   MODE = actions | states. Output: runs/oftvla/closedloop_validation/switch_replay_MODE.csv (refuses to overwrite)
source $HOME/VLA_FT/starvla_qwen_oftvla/scripts/common.sh
cd $REPO; git -C $REPO rev-parse --short HEAD; git -C $REPO status --short
MODE=${1:?actions|states}
LIBERO_SIF=/hnvme/workspace/y307bd13-vgv/containers/sym-openvla-oft.sif
OUT=$RUNS/closedloop_validation
MUJOCO_GL=egl apptainer exec --env PYTHONPATH=$REPO $LIBERO_SIF python3 -m oftvla.replay_switcher \
  --spec $HOME/VLA_FT/DataSamples/Libero/ClosedLoop/switch_spec.json --texts $HOME/VLA_FT/DataSamples/Libero/A0ASMAS/test.csv \
  --mode $MODE --out $OUT/switch_replay_$MODE.csv --workers 24
