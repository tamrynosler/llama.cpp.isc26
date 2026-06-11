#!/usr/bin/env bash
#SBATCH --job-name=ppl-dp-orch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=ppl-dp-orch-%j.log
#
# Fork (llama.cpp.isc26) perplexity across replicas. For each of 3 models, runs:
#   (1) 1 GPU, single replica, DEFAULT batching  -> throughput baseline + ref PPL
#   (2) 1 GPU, single replica, -b $CTX (n_seq=1)  -> matched-granularity PPL
#   (3) 4 GPUs, --data-parallel 4                 -> orchestrator throughput + PPL
#
# Legality check: run (3) PPL must equal run (2) PPL EXACTLY (both use n_seq=1).
# Run (1) uses default n_seq batching, so its PPL differs slightly by GEMM
# reduction order - that is expected (see README), not a defect. Throughput is
# the wall-clock / "chunks across N replicas in Y s" line; compare (3) vs (1).
#
# Cluster notes: NO GRES on this cluster - select GPUs with CUDA_VISIBLE_DEVICES,
# do NOT add "#SBATCH --gres". Modules via lmod.
#
# Usage: fill in the variables, build the fork (-DGGML_CUDA=ON), then:
#   sbatch tools/orchestrator/slurm/dp-orchestrator.slurm

set -uo pipefail

# ---- fill these in ----------------------------------------------------------
REPO="$HOME/llama.cpp.isc26"
BIN="$REPO/build/bin/llama-perplexity"
INPUT="$HOME/data/wikitext-2-raw/wiki.test.raw"
MODELS=(
  "/path/to/model1.gguf"   # TODO: SAME 3 models as baseline-original.slurm
  "/path/to/model2.gguf"   # TODO
  "/path/to/model3.gguf"   # TODO
)
CTX=512        # perplexity context window
NGL=99         # offload all layers to the GPU (required: orchestrator warns, does not force this)
DP=4           # replicas across the node's 4 GPUs
# -----------------------------------------------------------------------------

ml gcc cuda    # pin versions if needed: ml gcc/15.2 cuda/13.1

TIME=""; command -v /usr/bin/time >/dev/null 2>&1 && TIME="/usr/bin/time -v"

RESULTS="results/dp-orch-${SLURM_JOB_ID:-local}"
mkdir -p "$RESULTS"

echo "host=$(hostname)"
nvidia-smi -L 2>/dev/null || true
( cd "$REPO" && echo "fork commit: $(git rev-parse --short HEAD 2>/dev/null)" ) || true
echo

# run <label> <logfile> <command...>
run() {
  local label="$1"; shift
  local log="$1"; shift
  echo "=== $label ==="
  if $TIME "$@" > "$log" 2>&1; then
    grep -iE "Final estimate|chunks across .* replicas in|perf_context_print|Elapsed \(wall" "$log" || true
  else
    echo "FAILED ($label) - see $log"; tail -n 20 "$log"
  fi
  echo
}

for M in "${MODELS[@]}"; do
  name="$(basename "$M" .gguf)"

  # (1) 1 GPU, default batching - throughput baseline + reference PPL
  export CUDA_VISIBLE_DEVICES=0
  run "FORK | $name | 1 GPU | default batching" \
      "$RESULTS/${name}.1gpu-default.log" \
      "$BIN" -m "$M" -f "$INPUT" --ctx-size "$CTX" -ngl "$NGL"

  # (2) 1 GPU, -b $CTX => n_seq=1 - matched granularity for the exact legality check
  export CUDA_VISIBLE_DEVICES=0
  run "FORK | $name | 1 GPU | -b $CTX (n_seq=1, matched)" \
      "$RESULTS/${name}.1gpu-matched.log" \
      "$BIN" -m "$M" -f "$INPUT" --ctx-size "$CTX" -ngl "$NGL" -b "$CTX"

  # (3) 4 GPUs via the orchestrator - throughput + PPL (should equal run 2 exactly)
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  run "FORK | $name | 4 GPUs | --data-parallel $DP" \
      "$RESULTS/${name}.4gpu-dp.log" \
      "$BIN" -m "$M" -f "$INPUT" --ctx-size "$CTX" -ngl "$NGL" \
             --data-parallel "$DP" --dp-devices 0,1,2,3
done

echo "done -> $RESULTS"
echo "legality: diff the 'Final estimate' from *.1gpu-matched.log vs *.4gpu-dp.log (must match)"
echo "throughput: compare wall-clock of *.4gpu-dp.log vs *.1gpu-default.log"
