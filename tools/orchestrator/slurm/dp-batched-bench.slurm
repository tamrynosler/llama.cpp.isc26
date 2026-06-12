#!/usr/bin/env bash
#SBATCH --job-name=bb-dp-orch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=bb-dp-orch-%j.log
#
# Fork (llama.cpp.isc26) batched-bench throughput across replicas. For each model:
#   (A) 1 GPU,  no -dp            -> fork single-GPU baseline (per-config tokens/sec)
#   (B) 4 GPUs, --data-parallel 4 -> orchestrator weak scaling (aggregate "ALL" row)
#
# batched-bench is the GPU-bound throughput showcase: token generation is GPU-bound (unlike
# perplexity, which is host-bound on log_softmax), so the DP aggregate should scale toward Nx
# the 1-GPU number. Weak scaling: every replica runs the byte-identical single-context sweep;
# the tool's "ALL" row reports both the max-based aggregate (S t/s) and the end-to-end
# drain-span aggregate (S_e2e t/s). Per-config timing already excludes model load, so there is
# no compute-only caveat here (unlike the perplexity script).
#
# Cluster notes: NO GRES on this cluster - select GPUs with CUDA_VISIBLE_DEVICES, do NOT add
# "#SBATCH --gres". Modules via lmod. No input dataset needed (batched-bench uses random tokens).
#
# Usage: fill in the variables, build the fork (-DGGML_CUDA=ON), then:
#   sbatch tools/orchestrator/slurm/dp-batched-bench.slurm

set -uo pipefail

# ---- fill these in ----------------------------------------------------------
REPO="$HOME/llama.cpp.isc26"
BIN="$REPO/build/bin/llama-batched-bench"
MODELS=(
  "$HOME/models/qwen2-0_5b-instruct-q4_k_m.gguf"        # SAME 5 models as the perplexity script
  "$HOME/models/qwen2.5-1.5b-instruct-q5_k_m.gguf"
  "$HOME/models/qwen2.5-3b-instruct-q5_k_m.gguf"
  "$HOME/models/OLMo-2-1124-13B-Instruct-Q4_0.gguf"
  "$HOME/models/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf"
)
CTX=16384            # per-replica context; must hold the largest config: max(PL)*(PP+TG) <= CTX
NGL=99               # offload all layers to the GPU (orchestrator warns, does not force this)
DP=4                 # replicas across the node's 4 GPUs
PP=512               # prompt lengths to sweep (comma-separated ok, e.g. 128,256,512)
TG=128               # generation lengths
PL=1,2,4,8,16        # parallel sequences per replica
# -----------------------------------------------------------------------------

ml gcc cuda          # pin versions if needed: ml gcc/15.2 cuda/13.1

RESULTS="results/dp-bb-${SLURM_JOB_ID:-local}"
mkdir -p "$RESULTS"

echo "host=$(hostname)"
nvidia-smi -L 2>/dev/null || true
( cd "$REPO" && echo "fork commit: $(git rev-parse --short HEAD 2>/dev/null)" ) || true
echo

# run <label> <logfile> <command...> - capture full output, echo the table rows.
run() {
  local label="$1"; shift
  local log="$1";   shift
  echo "=== $label ==="
  if "$@" > "$log" 2>&1; then
    grep -E "^\|" "$log" || echo "  (no table rows - see $log)"
  else
    echo "FAILED ($label) - see $log"; tail -n 20 "$log"
  fi
  echo
}

for M in "${MODELS[@]}"; do
  name="$(basename "$M" .gguf)"

  # (A) 1 GPU baseline - byte-identical single-context batched-bench
  export CUDA_VISIBLE_DEVICES=0
  run "FORK | $name | 1 GPU | baseline" \
      "$RESULTS/${name}.1gpu.log" \
      "$BIN" -m "$M" -c "$CTX" -ngl "$NGL" -npp "$PP" -ntg "$TG" -npl "$PL"

  # (B) 4 GPUs via the orchestrator - weak-scaling aggregate (ALL row)
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  run "FORK | $name | 4 GPUs | --data-parallel $DP" \
      "$RESULTS/${name}.4gpu-dp.log" \
      "$BIN" -m "$M" -c "$CTX" -ngl "$NGL" -npp "$PP" -ntg "$TG" -npl "$PL" \
             --data-parallel "$DP" --dp-devices 0,1,2,3
done

echo "done -> $RESULTS"
echo "throughput: for each config, compare the DP 'ALL' row 'S t/s' (and 'S_e2e t/s') in"
echo "            *.4gpu-dp.log against the matching 1-GPU 'S t/s' in *.1gpu.log."
echo "            weak scaling should push the ALL aggregate toward ${DP}x the 1-GPU number."
echo "tip: re-run with --output-format jsonl for machine-readable per-replica + aggregate fields."
