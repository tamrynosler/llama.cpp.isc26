#!/usr/bin/env bash
# Track 2 (decode latency) — optimised CUDA build, with CUDA graphs enabled.
#
# Mirrors the project's optimised cluster build flags, plus -DGGML_CUDA_GRAPHS=ON
# (this fork defaults CUDA graphs OFF, ggml/CMakeLists.txt:117-118, so it must be explicit).
# The orchestrator MPI flag (-DLLAMA_ORCH_MPI=ON) is intentionally omitted — Track-2 decode
# is single-GPU; add it back only if you also want the multi-node orchestrator in this build.
#
# GPU arch must match the card:
#   A100 (practice cluster, xeon-cg2) -> 80
#   H100 (main cluster)               -> 90a
#
# Usage:
#   ml gcc cuda
#   decode-temp/build-decode.sh 80        # A100 today
#   decode-temp/build-decode.sh 90a       # H100 tomorrow

set -uo pipefail

ARCH="${1:-80}"
REPO="${REPO:-$HOME/llama.cpp.isc26}"
BUILD="${BUILD:-build}"

cd "$REPO" || exit 1
echo "building in $REPO/$BUILD for CUDA arch sm_$ARCH (commit $(git rev-parse --short HEAD))"

cmake -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
  -DGGML_NATIVE=OFF -DGGML_AVX512=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON || exit 1

cmake --build "$BUILD" -j"$(nproc)"
