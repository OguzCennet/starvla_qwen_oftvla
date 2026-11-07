#!/bin/bash
# Submit the held-out eval of one flow-head run x checkpoint as N one-GPU shard jobs plus a CPU merge job
# (afterok on all shards). Default shards: PI 16, GR00T 4 (head from the run's args.json).
# Usage: scripts/submit_flow_eval.sh RUN STEP [N] [-- extra flow_heldout_eval args, also passed to the merge]
set -e
source $HOME/VLA_FT/starvla_qwen_oftvla/scripts/common.sh
cd $REPO
RUN=${1:?RUN}; STEP=${2:?STEP}; shift 2
HEAD=$(python3 -c "import json; print(json.load(open('$RUNS/$RUN/args.json'))['head'])")
case $HEAD in pi) N=16; T=02:30:00;; gr00t) N=4; T=01:30:00;; *) echo "unknown head $HEAD"; exit 1;; esac
if [ $# -gt 0 ] && [ "$1" != "--" ]; then N=$1; shift; fi
[ "${1:-}" = "--" ] && shift
[ -f $RUNS/$RUN/steps_${STEP}_model.pt ] || { echo "missing $RUNS/$RUN/steps_${STEP}_model.pt"; exit 1; }
JIDS=""
for ((i = 0; i < N; i++)); do
  J=$(sbatch --parsable --time=$T --job-name=eval_${RUN}_${STEP}_s$i scripts/flow_heldout_eval.sbatch $RUN $STEP $i $N -- "$@")
  JIDS="$JIDS:$J"
done
M=$(sbatch --parsable --dependency=afterok$JIDS --job-name=eval_${RUN}_${STEP}_merge \
  scripts/flow_heldout_merge.sbatch $RUN $STEP $N -- "$@")
echo "$RUN step $STEP ($HEAD): $N shards (--time $T) ${JIDS#:} -> merge $M"
