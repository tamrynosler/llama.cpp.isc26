#!/bin/bash
#SBATCH --job-name=orig-ppl
#SBATCH --output=orig-ppl-%j.out.txt
#SBATCH --error=orig-ppl-%j.err.txt
#SBATCH --partition=workers
#SBATCH --nodes=1
#SBATCH --nodelist=xeon-cg2
#
# Baseline perplexity from the ORIGINAL, unmodified upstream llama.cpp.
#
# Runs stock llama-perplexity from a SEPARATE upstream clone (NOT this fork) on the exact same
# model + dataset + flags as our DP runs, so the fork's perplexity (baseline 28.91, DP 28.91)
# can be checked against the genuine "unoptimized llama.cpp" baseline the competition references.
# Single GPU, no data parallelism (upstream has none).
#
# One-time setup of the original repo (separate dir; does NOT touch this fork):
#   git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp-orig
#   cd ~/llama.cpp-orig
#   ml gcc; ml cuda
#   cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_BUILD_TYPE=Release
#   cmake --build build --target llama-perplexity -j$(nproc)
# (A100 = sm_80; use 90a on the H100 cluster and point --nodelist at cn001.)

set -euo pipefail

source /etc/profile.d/lmod.sh
module purge
module load gcc
module load cuda

# ---- perf-per-watt throttle (mirrors isc26 maxout): CPUs low + GPUs capped to 200 W ----
# Uses the isc26 power helpers (ssh-loop over $SLURM_JOB_NODELIST), reached via POWER_DIR (default
# $HOME/isc26/power). CPU governors are restored on any exit; the per-rank wrapper on the workload
# below re-raises the working rank's CPUs. GPU cap is left set on exit, same as maxout.
export POWER="${POWER_DIR:-$HOME/isc26/power}"
trap "$POWER/reset_nodes_cpus.sh" EXIT INT TERM
"$POWER/set_nodes_cpus_low.sh"
"$POWER/set_nodes_gpus_pl.sh" 200

ORIG="${ORIG_REPO:-$HOME/llama.cpp-orig}"
BIN="$ORIG/build/bin/llama-perplexity"
M="/shared/llms/models/Llama-3.1-Nemotron-70B-Instruct-HF-Q4_K_M.gguf"
DATA="${WIKITEXT:-/shared/llms/datasets/wikitext-2-raw/wiki.test.raw}"

export CUDA_VISIBLE_DEVICES=0

# same flags as the DP perplexity scripts (-c 512 chunk, -b 4096); PPL is independent of batching.
echo "################  ORIGINAL upstream llama.cpp perplexity (single GPU)  ################"
"$POWER/set_rank_cpus_high.sh" "$BIN" -m "$M" -f "$DATA" -ngl 99 -c 512 -b 4096
