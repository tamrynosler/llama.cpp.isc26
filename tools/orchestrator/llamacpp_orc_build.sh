#!/bin/bash

# Phase 0: Foundation Build. We are only building llama.cpp.

set -euo pipefail

# Load in the required modules
ml purge
ml gcc cuda hpcx

# Set the directories
LLAMA_DIR=$HOME/llama.cpp.isc26
NVHPC_CUDA=/shared/nvhpc/Linux_x86_64/26.3/cuda/13.1
# Local cluster: A100 = sm_80. Competition H100 PCIe = sm_90a.
# Set CUDA_ARCH=90a when building on the competition cluster.
CUDA_ARCH="${CUDA_ARCH:-90a}" # Change to 90a for competition cluster

cd "$LLAMA_DIR"
rm -rf build CMakeCache.txt CMakeFiles/

echo "========================================"
echo "Phase 0 build  |  CUDA_ARCH=$CUDA_ARCH"
echo "========================================"

cmake --fresh -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_AVX512=ON \
  -DGGML_AVX2=ON \
  -DGGML_FMA=ON \
  -DGGML_F16C=ON \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DGGML_RPC=ON \
  -DLLAMA_ORCH_MPI=ON \
  -DGGML_NATIVE=OFF \
  -DCMAKE_CUDA_COMPILER=$NVHPC_CUDA/bin/nvcc \
  -DCMAKE_INSTALL_RPATH="/shared/nvhpc/Linux_x86_64/26.3/comm_libs/nccl/lib;$NVHPC_CUDA/lib64;\$ORIGIN" \
  -DNCCL_INCLUDE_DIR=/shared/nvhpc/Linux_x86_64/26.3/comm_libs/nccl/include \
  -DNCCL_LIBRARY=/shared/nvhpc/Linux_x86_64/26.3/comm_libs/nccl/lib/libnccl.so \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
  -DCMAKE_C_COMPILER=/shared/gcc/15.2/out/bin/gcc \
  -DCMAKE_CXX_COMPILER=/shared/gcc/15.2/out/bin/g++ \
  -DCMAKE_ASM_COMPILER=/shared/gcc/15.2/out/bin/gcc \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  2>&1 | tee build_cmake.log

# Everything else here is extra, only worry about the stuff above. Compilation Complete

# Capture which optional libs CMake actually found (this is the key signal).
echo ""
echo "---- CMake feature detection ----"
grep -E -i "NCCL|libibverbs|verbs|cuBLAS|P2P|peer" build_cmake.log || true
echo "---------------------------------"

# Export the custom NCCL lib path so the linker (ld) can resolve transitive dependencies
export LD_LIBRARY_PATH="/shared/nvhpc/Linux_x86_64/26.3/comm_libs/nccl/lib:$LD_LIBRARY_PATH"

# Now run the build
cmake --build build --config Release -j "$(nproc)" 2>&1 | tee build_make.log

echo ""
echo "========================================"
echo "VERIFICATION"
echo "========================================"
# Verify whether packages were actually installed and it's not a false positive
# This checks the build logs to make sure nothing was missed by accident
BIN=$LLAMA_DIR/build/bin
FAIL=0

check_link() {
  local binary=$1
  local lib=$2
  local required=$3   # "REQ" or "OPT"
  if ldd "$binary" 2>/dev/null | grep -q "$lib"; then
    printf "  %-20s %-15s OK\n" "$(basename "$binary")" "$lib"
  else
    if [[ "$required" == "REQ" ]]; then
      printf "  %-20s %-15s MISSING (REQUIRED)\n" "$(basename "$binary")" "$lib"
      FAIL=1
    else
      printf "  %-20s %-15s missing (optional)\n" "$(basename "$binary")" "$lib"
    fi
  fi
}

echo ""
echo "Binary presence:"
for b in llama-bench llama-cli llama-server llama-perplexity llama-quantize llama-speculative rpc-server; do
  if [[ -x "$BIN/$b" ]]; then
    printf "  %-25s OK\n" "$b"
  else
    printf "  %-25s MISSING\n" "$b"
    FAIL=1
  fi
done

echo ""
echo "Library linkage:"
check_link "$BIN/llama-bench"   "libcudart"       "REQ"
check_link "$BIN/llama-bench"   "libggml-cuda"    "REQ"
check_link "$BIN/llama-bench"   "libcublas"       "REQ"
check_link "$BIN/llama-bench"   "libnccl"         "OPT"   # needed for -sm tensor (Phase 3)
check_link "$BIN/rpc-server"    "libibverbs"      "OPT"   # needed for RDMA on multi-node

echo ""
echo "Help smoke test:"
"$BIN/llama-bench" --help > /dev/null && echo "  llama-bench --help          OK" || { echo "  llama-bench --help          FAIL"; FAIL=1; }
"$BIN/llama-perplexity" --help > /dev/null 2>&1 && echo "  llama-perplexity --help     OK" || echo "  llama-perplexity --help     (no --help, normal)"
"$BIN/rpc-server" --help > /dev/null 2>&1 && echo "  rpc-server --help           OK" || echo "  rpc-server --help           (no --help, normal)"

echo ""
if [[ $FAIL -eq 0 ]]; then
  echo "Phase 0 build: PASS"
else
  echo "Phase 0 build: FAIL — fix before proceeding"
  exit 1
fi
