#!/bin/bash
# Submit N copies of a job script, each starting after the previous one ends (any state).
# Usage: bash scripts/submit_chain.sh scripts/train_e3_A0.sbatch 4
set -e
SCRIPT=$1; N=${2:-3}
mkdir -p /hnvme/workspace/y307bd13-vgv/runs/oftvla/e3_A0
JID=$(sbatch --parsable "$SCRIPT"); echo "job 1: $JID"
for i in $(seq 2 "$N"); do
  JID=$(sbatch --parsable --dependency=afterany:$JID "$SCRIPT"); echo "job $i: $JID"
done
