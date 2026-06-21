#!/usr/bin/env bash
#SBATCH --job-name=decode-ppl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --nodelist=xeon-cg2
#SBATCH --output=decode-ppl-%j.log
#
# Track 2 — perplexity (quality / quantitative correctness oracle).
#
# Two uses:
#   1. Quantify the QUALITY cost of KV-cache quantization (f16 vs q8_0 vs q4_0) — pairs with
#      decode-kvquant.slurm, which showed q8_0/q4_0 are SLOWER for decode. PPL here shows the
#      quality side, so the KV-quant tradeoff is fully characterized (slower AND lossy).
#   2. Reusable correctness oracle for L3 kernel work: a decode-path change should leave PPL
#      unchanged within tolerance (run f16 before vs after; ~identical PPL = output preserved).
#      This complements the llama-simple greedy diff (exact tokens); PPL is the quantitative check.
#
# Quick by default (CHUNKS modest). Raise CHUNKS for a tighter number. Comparing configs at the
# SAME -c/--chunks is what matters; the absolute PPL won't match the textbook -c 512 wikitext value.
#
# Pinned: xeon-cg2 GPU 0. Build first with decode-temp/build-decode.sh 80.
# Needs the wikitext-2-raw test file (Track-1 baseline used the same path).
# Usage:  sbatch decode-temp/decode-perplexity.slurm

set -uo pipefail

# ---- config -----------------------------------------------------------------
REPO="$HOME/llama.cpp.isc26"
BIN="$REPO/build/bin/llama-perplexity"
DENSE="/shared/llms/models/Qwen2.5-14B-Instruct-Q4_0.gguf"
# shared filesystem (same root as the models). Override at submit time if the filename differs:
#   DATASET=/shared/llms/datasets/<exact>.raw sbatch decode-temp/decode-perplexity.slurm
DATASET="${DATASET:-/shared/llms/datasets/wikitext-2-raw/wiki.test.raw}"
GPU_IDX=0
CTX=4096            # cache reaches this depth within a chunk -> KV-quant quality effect shows
CHUNKS=20           # quick; -1 = full corpus (slow). raise for a tighter estimate
KVTYPES="f16 q8_0 q4_0"
# -----------------------------------------------------------------------------

ml gcc cuda
export CUDA_VISIBLE_DEVICES="$GPU_IDX"

if [ ! -f "$DATASET" ]; then
    echo "ERROR: dataset not found: $DATASET" >&2
    echo "  get wikitext-2-raw (wiki.test.raw) or point DATASET at an existing text corpus." >&2
    exit 1
fi

echo "node=$(hostname) commit=$(git -C "$REPO" rev-parse --short HEAD) ctx=$CTX chunks=$CHUNKS" >&2

for t in $KVTYPES; do
    out="decode-ppl-$t.txt"
    echo ">>> KV=$t  (-fa 1 -ctk $t -ctv $t -c $CTX --chunks $CHUNKS)" >&2
    "$BIN" -m "$DENSE" -f "$DATASET" -ngl 99 -fa 1 -ctk "$t" -ctv "$t" \
           -c "$CTX" --chunks "$CHUNKS" 2>&1 | tee "$out"
    echo "    RESULT KV=$t -> $(grep -i 'Final estimate' "$out" || echo '(no PPL line - check log)')" >&2
done

echo "done. Compare 'Final estimate: PPL' across f16/q8_0/q4_0 = KV-quant quality cost." >&2
