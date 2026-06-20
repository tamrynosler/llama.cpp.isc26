#!/usr/bin/env bash
#SBATCH --job-name=decode-correct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --nodelist=xeon-cg2
#SBATCH --output=decode-correct-%j.log
#
# Track 2 correctness oracle — deterministic greedy generation, fixed prompt + seed.
#
# The generated token text is the reference output, per model. After a decode-path change,
# re-run on the SAME node + GPU and `diff` the new file against the baseline file:
# identical text  => the change preserved decode output (a valid speed-only change).
# any difference  => the optimization altered results — investigate before trusting it.
# (Greedy/argmax on a fixed model+build+GPU is deterministic; FP reduction order can rarely
#  flip a token, so treat a diff as "look", not automatically "broken".)
#
# Pinned to: node xeon-cg2, GPU 0 (80 GB A100).
#
# Usage:  sbatch decode-temp/decode-correctness.slurm <tag>
#   baseline:  sbatch decode-temp/decode-correctness.slurm baseline   -> decode-correct-{dense,moe}-baseline.txt
#   after opt: sbatch decode-temp/decode-correctness.slurm opt1       -> decode-correct-{dense,moe}-opt1.txt
#   then:      diff decode-correct-dense-baseline.txt decode-correct-dense-opt1.txt

set -uo pipefail

# ---- config -----------------------------------------------------------------
REPO="$HOME/llama.cpp.isc26"
BIN="$REPO/build/bin/llama-cli"
DENSE="/shared/llms/models/Qwen2.5-14B-Instruct-Q4_0.gguf"
MOE="/shared/llms/models/deepseek-moe-16b-base.Q4_K_M.gguf"   # NOTE: dir assumed = /shared/llms/models
GPU_IDX=0
NGEN=256
SEED=1234
PROMPT="The history of computing began"
# -----------------------------------------------------------------------------

TAG="${1:-$(date +%s)}"

ml gcc cuda
export CUDA_VISIBLE_DEVICES="$GPU_IDX"

echo "node=$(hostname) commit=$(git -C "$REPO" rev-parse --short HEAD) tag=$TAG" >&2

for entry in "dense:$DENSE" "moe:$MOE"; do
    name="${entry%%:*}"
    model="${entry#*:}"
    out="decode-correct-$name-$TAG.txt"     # written to CWD (launch dir), not the repo tree
    echo ">>> $name: $model -> $out" >&2
    # -no-cnv: one-shot completion (no chat loop); --temp 0: greedy/argmax (deterministic);
    # --no-display-prompt: emit only generated tokens; fixed --seed for full reproducibility.
    "$BIN" -m "$model" -ngl 99 -no-cnv --no-display-prompt \
           --temp 0 --seed "$SEED" -n "$NGEN" -p "$PROMPT" 2>/dev/null | tee "$out"
done

echo "done" >&2
