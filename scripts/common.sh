# Shared paths for job scripts (sourced). Written out in full so jobs don't depend on ~/.bashrc.
REPO=$HOME/VLA_FT/starvla_qwen_oftvla
SIF=/hnvme/workspace/y307bd13-vgv/containers/starvla.sif
BASE_VLM=/hnvme/workspace/y307bd13-vgv/models/Qwen3-VL-4B-Instruct
DATA=/hnvme/workspace/y307bd13-vgv/data/libero_lang/v1
SPLITS=$HOME/VLA_FT/DataSamples/Libero/Splits
LABELS=$HOME/VLA_FT/DataSamples/Libero/Labels/switch_labels_v1.csv
TEXTS=$HOME/VLA_FT/DataSamples/Libero/A0ASMAS   # per-frame A0 / AS / MAS texts (label dictionary v2.3)
RUNS=/hnvme/workspace/y307bd13-vgv/runs/oftvla
PYLIBS=$HOME/pylibs/h5py
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # less memory fragmentation
run_py() { apptainer exec --nv --env PYTHONPATH=$REPO:$PYLIBS $SIF python "$@"; }
