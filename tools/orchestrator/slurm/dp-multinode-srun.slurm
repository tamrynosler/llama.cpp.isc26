#!/usr/bin/env bash
#SBATCH --job-name=bb-dp-mn
#SBATCH --nodes=2
#SBATCH --nodelist=cn002,cn003
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=bb-dp-mn-%j.log
#
# Multi-node (srun fan-out) data-parallel batched-bench. Each node independently runs the
# existing SINGLE-NODE DP sweep (--data-parallel 4) across its own 4 GPUs, concurrently; there
# is NO inter-node traffic and NO RPC. The aggregator then sums per-node throughput into a
# full-cluster number and reports node scaling (cluster / fastest-node, ideal = #nodes).
#
# This srun+python harness is a PROTOTYPE / de-risk. The intended end state is a C++-native
# cross-node orchestrator inside the repo (see PROGRESS "Ordered next steps", S19); this script
# proves the workload + numbers first.
#
# Cluster notes: NO GRES - pick GPUs with CUDA_VISIBLE_DEVICES, no "#SBATCH --gres". Modules via
# lmod. Home is shared (NFS), so one build + one results dir are visible on every node.
#
# Usage: fill in the variables, build the fork (-DGGML_CUDA=ON), then:
#   sbatch tools/orchestrator/slurm/dp-multinode-srun.slurm

set -uo pipefail

# ---- fill these in ----------------------------------------------------------
REPO="$HOME/llama.cpp.isc26"
BIN="$REPO/build/bin/llama-batched-bench"
AGG="$REPO/tools/orchestrator/slurm/aggregate-multinode.py"
MODELS=(
  "$HOME/models/qwen2-0_5b-instruct-q4_k_m.gguf"        # same 5 models as the single-node script
  "$HOME/models/qwen2.5-1.5b-instruct-q5_k_m.gguf"
  "$HOME/models/qwen2.5-3b-instruct-q5_k_m.gguf"
  "$HOME/models/OLMo-2-1124-13B-Instruct-Q4_0.gguf"
  "$HOME/models/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf"
)
CTX=65536
NGL=99
DP=4                 # replicas per node (one per local GPU)
NBATCH=4096
UBATCH=2048
PP=512
TG=128
PL=1,2,4,8,16,32,64
# -----------------------------------------------------------------------------

ml gcc cuda          # pin versions if needed: ml gcc/15.2 cuda/13.1

RESULTS="$REPO/results/dp-mn-${SLURM_JOB_ID:-local}"
mkdir -p "$RESULTS"

echo "host=$(hostname)  nodelist=${SLURM_JOB_NODELIST:-?}  nnodes=${SLURM_NNODES:-?}"
( cd "$REPO" && echo "fork commit: $(git rev-parse --short HEAD 2>/dev/null)" ) || true
echo

# exported so the single-quoted srun body below evaluates them ON each node (srun propagates the
# environment by default). single quotes keep $(hostname) node-local and dodge all quoting traps.
export DPMN_BIN="$BIN" DPMN_RESULTS="$RESULTS"
export DPMN_CTX="$CTX" DPMN_NBATCH="$NBATCH" DPMN_UBATCH="$UBATCH" DPMN_NGL="$NGL"
export DPMN_DP="$DP" DPMN_PP="$PP" DPMN_TG="$TG" DPMN_PL="$PL"

# one srun per model; each fans out to ALL allocated nodes (one task per node). Every node runs
# the full single-node DP sweep on its own 4 GPUs and writes a node-tagged JSONL.
for M in "${MODELS[@]}"; do
  name="$(basename "$M" .gguf)"
  export DPMN_MODEL="$M" DPMN_NAME="$name"
  echo "=== $name : launching on ${SLURM_NNODES:-?} node(s) ==="
  srun --ntasks-per-node=1 bash -c '
    out="$DPMN_RESULTS/$DPMN_NAME.node-$(hostname).jsonl"
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    "$DPMN_BIN" -m "$DPMN_MODEL" -c "$DPMN_CTX" -b "$DPMN_NBATCH" -ub "$DPMN_UBATCH" \
        -ngl "$DPMN_NGL" -npp "$DPMN_PP" -ntg "$DPMN_TG" -npl "$DPMN_PL" \
        --data-parallel "$DPMN_DP" --dp-devices 0,1,2,3 --output-format jsonl \
        > "$out" 2> "${out%.jsonl}.err"
  '
done

echo
echo "=== aggregating across nodes ==="
python3 "$AGG" "$RESULTS"
echo "done -> $RESULTS"
echo "each per-node value is that node's single-node DP throughput; cluster = sum; ideal scaling = #nodes."
