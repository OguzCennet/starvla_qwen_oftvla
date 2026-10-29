#!/bin/bash
#SBATCH --job-name=oftvla_eval
#SBATCH --account=y307bd
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/hnvme/workspace/y307bd13-vgv/runs/oftvla/e3_A0/eval/slurm_%j.out
# Closed-loop LIBERO eval of one checkpoint. Policy server (starvla.sif) + LIBERO client (sym-openvla-oft.sif).
# Usage: sbatch scripts/eval_libero.sbatch STEP IDS_CSV NAME [video]
#   e.g. sbatch scripts/eval_libero.sbatch 30000 $HOME/VLA_FT/DataSamples/Libero/Splits/sanity_train_ids.csv sanity video
# Output: runs/oftvla/e3_A0/eval/NAME_stepSTEP.csv (resumable), videos in .../eval/videos_NAME_stepSTEP/
source $HOME/VLA_FT/starvla_qwen_oftvla/scripts/common.sh
STEP=$1; IDS=$2; NAME=$3; VIDEO=$4
RUN=$RUNS/e3_A0; OUT=$RUN/eval; mkdir -p $OUT
LIBERO_SIF=/hnvme/workspace/y307bd13-vgv/containers/sym-openvla-oft.sif
PORT=$((10000 + SLURM_JOB_ID % 20000)); READY=$OUT/.ready_$SLURM_JOB_ID
cd $REPO; git -C $REPO rev-parse --short HEAD

run_py -m oftvla.policy --run_dir $RUN --step $STEP --base_vlm $BASE_VLM --port $PORT --ready_file $READY &
SERVER=$!
trap "kill $SERVER 2>/dev/null; rm -f $READY" EXIT
for i in $(seq 300); do [ -f $READY ] && break; kill -0 $SERVER 2>/dev/null || { echo "server died"; exit 1; }; sleep 2; done
[ -f $READY ] || { echo "server not ready after 10 min"; exit 1; }

VIDEO_ARG=""; [ "$VIDEO" = "video" ] && VIDEO_ARG="--video_dir $OUT/videos_${NAME}_step${STEP}"
MUJOCO_GL=egl apptainer exec --nv --env PYTHONPATH=$REPO $LIBERO_SIF python3 -m oftvla.libero_client \
  --ids $IDS --labels_csv $LABELS --out $OUT/${NAME}_step${STEP}.csv --port $PORT $VIDEO_ARG
