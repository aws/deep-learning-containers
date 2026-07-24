#!/usr/bin/env bash
set -euo pipefail

# CUDA runtime sanity checks
# Usage: cuda_runtime_test.sh <cuda_version>
# Example: cuda_runtime_test.sh 12.9

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
if ldconfig -p 2>/dev/null | grep -q libcudart; then
  echo "PASS: libcudart found in ldconfig"
else
  echo "FAIL: libcudart not found in ldconfig"
  FAILED=1
fi

# --- nvcc should NOT be present in runtime image ---
if command -v nvcc &>/dev/null; then
  echo "FAIL: nvcc found in runtime image (should not be present)"
  FAILED=1
else
  echo "PASS: nvcc not present (expected for runtime image)"
fi

# --- GPU compute: nvidia-smi query compute capability ---
if nvidia-smi &>/dev/null; then
  COMPUTE=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
  if [ -n "$COMPUTE" ]; then
    echo "PASS: GPU compute capability $COMPUTE"
  else
    echo "FAIL: could not query GPU compute capability"
    FAILED=1
  fi
fi

# --- GPU compute: cudaMalloc/cudaFree via CUDA runtime API ---
if nvidia-smi &>/dev/null; then
  if python3 -c "
import ctypes, ctypes.util
cudart = ctypes.CDLL(ctypes.util.find_library('cudart'))
ptr = ctypes.c_void_p()
rc = cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(1024))
assert rc == 0, f'cudaMalloc failed with rc={rc}'
rc = cudart.cudaFree(ptr)
assert rc == 0, f'cudaFree failed with rc={rc}'
" 2>/dev/null; then
    echo "PASS: cudaMalloc/cudaFree succeeded"
  else
    echo "FAIL: cudaMalloc/cudaFree failed"
    FAILED=1
  fi
fi

exit $FAILED
