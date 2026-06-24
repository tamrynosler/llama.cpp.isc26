#!/bin/bash
# Submit a tuning sweep to whichever 80 GB A100 node (xeon-cg2 / xeon-cg3) is free right now.
#
# A SLURM batch script can't choose its own node (the node is assigned before it runs), and
# --nodelist=xeon-cg2,xeon-cg3 means "require BOTH", not "either". So we decide at submit time:
# check node state with sinfo and pin the submission to a free one.
#
# Usage:
#   bash tools/orchestrator/slurm/tuning/submit-free.sh tools/orchestrator/slurm/tuning/tune-t1-decode.slurm
#   # extra sbatch args pass through, e.g.  ... tune-t4-heavy.slurm --time=02:00:00

set -euo pipefail

SCRIPT="${1:?usage: submit-free.sh <slurm-script> [extra sbatch args...]}"
shift

CANDIDATES=(xeon-cg2 xeon-cg3)
node=""

# prefer a fully idle node; fall back to a partially-used (mix) node that may still have a free GPU.
for want in idle mix; do
    for n in "${CANDIDATES[@]}"; do
        st=$(sinfo -h -n "$n" -o "%t" 2>/dev/null | head -1)
        if [[ "$st" == "$want" ]]; then node="$n"; break 2; fi
    done
done
node="${node:-xeon-cg2}"   # last-resort default; the job just queues if it's busy

echo "submit-free: node states -> $(sinfo -h -n xeon-cg2,xeon-cg3 -o '%n=%t' 2>/dev/null | tr '\n' ' ')"
echo "submit-free: submitting '$SCRIPT' to $node"
sbatch --nodelist="$node" "$@" "$SCRIPT"
