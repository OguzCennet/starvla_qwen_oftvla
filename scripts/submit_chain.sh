#!/bin/bash
# Submit N copies of a job script, each starting after the previous one ends (any state).
# Usage: bash scripts/submit_chain.sh scripts/train_e3_A0.sbatch 4
#        bash scripts/submit_chain.sh scripts/train_e3_text.sbatch 3 e3v5_MAS MAS   (job name = run dir, then script args)
set -e
SCRIPT=$1; N=${2:-3}; NAME=${3:-e3_A0}; shift $(( $# < 3 ? $# : 3 ))
mkdir -p /hnvme/workspace/y307bd13-vgv/runs/oftvla/$NAME
OPTS="--parsable --job-name=$NAME"
JID=$(sbatch $OPTS "$SCRIPT" "$@"); echo "job 1: $JID"
for i in $(seq 2 "$N"); do
  JID=$(sbatch $OPTS --dependency=afterany:$JID "$SCRIPT" "$@"); echo "job $i: $JID"
done
