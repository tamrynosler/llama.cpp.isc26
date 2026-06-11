#!/usr/bin/env bash
#SBATCH --job-name=ppl-baseline-orig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=ppl-baseline-orig-%j.log
#
# Baseline perplexity on the ORIGINAL (upstream) llama.cpp checkout, 3 models,
# ONE GPU. This is the reference PPL + timing that the fork's single-replica
# path is compared against (does our fork regress vs upstream?).
#
# Cluster notes: this cluster has NO GRES - every node has 4x H100 and we select
# GPUs with CUDA_VISIBLE_DEVICES. Do NOT add "#SBATCH --gres". Modules via lmod.
#
# Usage: fill in the variables below, build the original repo first, then:
#   sbatch tools/orchestrator/slurm/baseline-original.slurm
# (sbatch can be run from anywhere; paths below are absolute.)

set -uo pipefail

# ---- fill these in ----------------------------------------------------------
ORIG_REPO="$HOME/llama.cpp"                              # original upstream checkout
ORIG_BIN="$ORIG_REPO/build/bin/llama-perplexity"        # build with -DGGML_CUDA=ON first
INPUT="$HOME/data/wikitext-2-raw/wiki.test.raw"         # perplexity input text (see README)
MODELS=(
  "/path/to/model1.gguf"   # TODO: fill in - use the SAME 3 models in dp-orchestrator.slurm
  "/path/to/model2.gguf"   # TODO
  "/path/to/model3.gguf"   # TODO
)
CTX=512        # standard perplexity context window
NGL=99         # offload all layers to the GPU
# -----------------------------------------------------------------------------

ml gcc cuda    # pin versions if your site needs them, e.g.: ml gcc/15.2 cuda/13.1

export CUDA_VISIBLE_DEVICES=0   # single-GPU baseline

# optional wall-clock/peak-RSS wrapper (skipped if /usr/bin/time is absent)
TIME=""; command -v /usr/bin/time >/dev/null 2>&1 && TIME="/usr/bin/time -v"

RESULTS="results/baseline-orig-${SLURM_JOB_ID:-local}"
mkdir -p "$RESULTS"

echo "host=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi -L 2>/dev/null || true
( cd "$ORIG_REPO" && echo "orig repo commit: $(git rev-parse --short HEAD 2>/dev/null)" ) || true
echo

for M in "${MODELS[@]}"; do
  name="$(basename "$M" .gguf)"
  log="$RESULTS/${name}.1gpu.log"
  echo "=== ORIGINAL | $name | 1 GPU | ctx=$CTX ngl=$NGL ==="
  if $TIME "$ORIG_BIN" -m "$M" -f "$INPUT" --ctx-size "$CTX" -ngl "$NGL" > "$log" 2>&1; then
    grep -iE "Final estimate|perf_context_print|Elapsed \(wall" "$log" || true
  else
    echo "FAILED ($name) - see $log"; tail -n 20 "$log"
  fi
  echo
done

echo "done -> $RESULTS"
