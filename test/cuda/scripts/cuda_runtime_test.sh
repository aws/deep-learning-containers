#!/usr/bin/env bash
set -euo pipefail

# CUDA runtime sanity checks
# Usage: cuda_runtime_test.sh <cuda_version>
# Example: cuda_runtime_test.sh 12.8

CUDA_VERSION="${1:?Usage: cuda_runtime_test.sh <cuda_version>}"
FAILED=0

# --- nvidia-smi detects GPU(s) ---
if nvidia-smi &>/dev/null; then
  GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
  echo "PASS: nvidia-smi detected $GPU_COUNT GPU(s)"
else
  echo "FAIL: nvidia-smi not found or no GPUs detected"
  FAILED=1
fi

# --- CUDA_HOME is set ---
if [ -n "${CUDA_HOME:-}" ] && [ -d "$CUDA_HOME" ]; then
  echo "PASS: CUDA_HOME=$CUDA_HOME"
else
  echo "FAIL: CUDA_HOME is not set or does not exist"
  FAILED=1
fi

# --- LD_LIBRARY_PATH contains cuda lib64 ---
if [[ "${LD_LIBRARY_PATH:-}" == */usr/local/cuda/lib64* ]]; then
  echo "PASS: LD_LIBRARY_PATH contains /usr/local/cuda/lib64"
else
  echo "FAIL: LD_LIBRARY_PATH missing /usr/local/cuda/lib64"
  FAILED=1
fi

# --- CUDA runtime libs are loadable ---
if python3 -c "import ctypes; ctypes.CDLL('libcudart.so')" &>/dev/null; then
  echo "PASS: libcudart.so is loadable"
else
  echo "FAIL: libcudart.so is not loadable"
  FAILED=1
fi

# --- nvcc should NOT be present in runtime image ---
if command -v nvcc &>/dev/null; then
  echo "FAIL: nvcc found in runtime image (should not be present)"
  FAILED=1
else
  echo "PASS: nvcc not present (expected for runtime image)"
fi

exit $FAILED
