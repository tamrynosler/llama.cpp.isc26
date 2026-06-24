#!/bin/bash
#SBATCH --job-name=tune-t1
#SBATCH --output=tune-t1-%j.out.txt
#SBATCH --error=tune-t1-%j.err.txt
#SBATCH --partition=workers
#SBATCH --nodes=1
#SBATCH --nodelist=xeon-cg2
#
# T-1 single-GPU parameter sweep: 70B Q4_K_M on one 80 GB A100. Workload 512 / 512 (decode-bound).
#
# Binding lever = B (parallelism, -npl). Decode is weight-bandwidth-bound at low batch, so batching
# sequences amortizes weight reads almost linearly until compute saturates, then S t/s plateaus.
# Find where it flattens (expected ~16-64) -- don't chase the memory ceiling (~107 f16).
#
# ctx fixed at 65536 = npl_max(64) x (npp+ntg)=1024. f16 KV (plenty of headroom at this tier).
# -ub/-b are not the lever here; held at 512/2048.
#
# Build (one 80 GB A100, sm_80), on feat/orchestrator2:
#   ml gcc; ml cuda
#   cmake -B build-a100 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-a100 --target llama-batched-bench -j$(nproc)

set -euo pipefail
source /etc/profile.d/lmod.sh
module purge
module load gcc
module load cuda

BIN="$HOME/llama.cpp.isc26/build-a100/bin/llama-batched-bench"
M="/shared/llms/models/Llama-3.1-Nemotron-70B-Instruct-HF-Q4_K_M.gguf"
export CUDA_VISIBLE_DEVICES=0

"$BIN" -m "$M" -fa on -ngl 999 -c 65536 -b 2048 -ub 512 \
    -npp 512 -ntg 512 -npl 1,2,4,8,16,32,64
