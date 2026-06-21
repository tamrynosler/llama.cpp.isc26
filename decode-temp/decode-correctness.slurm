#!/usr/bin/env bash
#SBATCH --job-name=decode-correct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --nodelist=xeon-cg2
#SBATCH --output=decode-correct-%j.log
#
# Track 2 correctness oracle — deterministic greedy generation, per model.
#
# Uses llama-simple (examples/simple): greedy argmax sampling, runs the real batch-1
# autoregressive decode loop (exactly the path Track 2 optimizes), prints tokens to stdout,
# then exits. No chat template, no interactive mode, no stdin, no sampler params to misset —
# greedy on a fixed model+build+GPU is deterministic, so the output is a stable reference.
# (We deliberately do NOT use llama-cli here: it defaults to interactive/conversation mode
#  and loops on a batch job's empty stdin — that was the "infinite >" hang.)
#
# After a decode-path change, re-run on the SAME node + GPU and `diff` the new file against
# the baseline file:
#   identical => the change preserved decode output (a valid speed-only change).
#   differs   => the optimization altered results — investigate before trusting it.
#   (FP reduction-order changes can rarely flip a late token; treat a diff as "look", not
#    automatically "broken" — and a perplexity-within-tolerance check is the quantitative backup.)
#
# Pinned to: node xeon-cg2, GPU 0 (80 GB A100).
#
# Usage:  sbatch decode-temp/decode-correctness.slurm <tag>
#   baseline:  sbatch decode-temp/decode-correctness.slurm baseline -> decode-correct-{dense,moe}-baseline.txt
#   after opt: sbatch decode-temp/decode-correctness.slurm opt1     -> decode-correct-{dense,moe}-opt1.txt
#   then:      diff decode-correct-dense-baseline.txt decode-correct-dense-opt1.txt

set -uo pipefail

# ---- config -----------------------------------------------------------------
REPO="$HOME/llama.cpp.isc26"
BIN="$REPO/build/bin/llama-simple"
DENSE="/shared/llms/models/Qwen2.5-14B-Instruct-Q4_0.gguf"
MOE="/shared/llms/models/deepseek-moe-16b-base.Q4_K_M.gguf"
GPU_IDX=0
NGEN=256
PROMPT="The history of computing began"
# -----------------------------------------------------------------------------

TAG="${1:-run}"

ml gcc cuda
export CUDA_VISIBLE_DEVICES="$GPU_IDX"

echo "node=$(hostname) commit=$(git -C "$REPO" rev-parse --short HEAD) tag=$TAG" >&2

for entry in "dense:$DENSE" "moe:$MOE"; do
    name="${entry%%:*}"
    model="${entry#*:}"
    out="decode-correct-$name-$TAG.txt"     # written to CWD (launch dir), not the repo tree
    echo ">>> $name: $model -> $out" >&2
    # llama-simple: -m model, -ngl 99 (default), -n tokens, then the prompt (positional).
    # stdout = prompt + generated text (the artifact to diff); stderr (load/decode stats) -> slurm log.
    "$BIN" -m "$model" -ngl 99 -n "$NGEN" "$PROMPT" > "$out"
    echo "    -> $(wc -c < "$out") bytes in $out" >&2
done

echo "done" >&2
