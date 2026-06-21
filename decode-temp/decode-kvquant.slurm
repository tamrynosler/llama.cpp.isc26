#!/usr/bin/env bash
#SBATCH --job-name=decode-kvquant
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --nodelist=xeon-cg2
#SBATCH --output=decode-kvquant-%j.log
#
# Track 2 — KV-cache quantization speedup at depth (dense Qwen-14B).
#
# KV-quant's payoff grows with context: a smaller KV cache means fewer bytes read by the
# attention decode kernel each step. At depth 0 it does ~nothing (no cache to read); the
# benefit shows up at long context. This sweeps f16 (baseline) vs q8_0 vs q4_0 KV across
# increasing KV depth so we can see the speedup curve.
#
# REQUIREMENTS (verified in-source):
#   - flash attention ON (-fa 1): quantized K/V is only handled by the CUDA flash-attn path.
#   - build with -DGGML_CUDA_FA_ALL_QUANTS=ON (build-decode.sh already does) for q8_0/q4_0 KV.
#
# CAVEAT: KV-quant is LOSSY — it does NOT preserve output (the greedy-diff oracle will differ;
# that's expected). This measures SPEED only. Quantify the quality cost separately with
# llama-perplexity at each KV type (a follow-up run).
#
# Pinned: xeon-cg2 GPU 0. Build first with decode-temp/build-decode.sh 80.
# Usage:  sbatch decode-temp/decode-kvquant.slurm

set -uo pipefail

# ---- config -----------------------------------------------------------------
REPO="$HOME/llama.cpp.isc26"
BIN="$REPO/build/bin/llama-bench"
DENSE="/shared/llms/models/Qwen2.5-14B-Instruct-Q4_0.gguf"
GPU_IDX=0
DEPTHS="0,2048,8192,16384,32768"   # KV-quant benefit grows with depth
NGEN=128
REPS=5
KVTYPES="f16 q8_0 q4_0"            # matched K=V; f16 is the baseline
# -----------------------------------------------------------------------------

ml gcc cuda
export CUDA_VISIBLE_DEVICES="$GPU_IDX"

echo "=== provenance ==="
echo "node=$(hostname) commit=$(git -C "$REPO" rev-parse --short HEAD) gpu=$GPU_IDX"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "=================="

for t in $KVTYPES; do
    echo ">>> KV cache type = $t (-ctk $t -ctv $t -fa 1)"
    "$BIN" -m "$DENSE" -ngl 99 -fa 1 -ctk "$t" -ctv "$t" -p 0 -n "$NGEN" -d "$DEPTHS" -r "$REPS" -o md
    echo ""
done

echo "done. Compare q8_0/q4_0 tg vs f16 at each depth = the KV-quant speedup curve." >&2
