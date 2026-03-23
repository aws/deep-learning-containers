#!/usr/bin/env bash
set -euo pipefail

# CUDA devel sanity checks
# Usage: cuda_devel_test.sh <cuda_version>
# Example: cuda_devel_test.sh 12.8

CUDA_VERSION="${1:?Usage: cuda_devel_test.sh <cuda_version>}"
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

# --- nvcc is present and reports correct version ---
if command -v nvcc &>/dev/null; then
  NVCC_OUT=$(nvcc --version 2>&1)
  if echo "$NVCC_OUT" | grep -q "release ${CUDA_VERSION}"; then
    echo "PASS: nvcc reports CUDA $CUDA_VERSION"
  else
    echo "FAIL: nvcc version mismatch (expected $CUDA_VERSION)"
    echo "$NVCC_OUT"
    FAILED=1
  fi
else
  echo "FAIL: nvcc not found in devel image"
  FAILED=1
fi

# --- CUDA headers exist ---
if [ -f /usr/local/cuda/include/cuda.h ]; then
  echo "PASS: CUDA headers found"
else
  echo "FAIL: /usr/local/cuda/include/cuda.h not found"
  FAILED=1
fi

# --- Compile and run cuda-samples (deviceQuery, vectorAdd) ---
SAMPLES_DIR=$(mktemp -d)
SAMPLES_TAG="v${CUDA_VERSION}"

echo "Cloning cuda-samples ${SAMPLES_TAG}..."
if git clone --depth 1 --branch "${SAMPLES_TAG}" \
  https://github.com/NVIDIA/cuda-samples.git "$SAMPLES_DIR" >/dev/null 2>&1; then

  # deviceQuery
  echo "Building deviceQuery..."
  BUILD_DIR="$SAMPLES_DIR/build/deviceQuery"
  if cmake -B "$BUILD_DIR" -S "$SAMPLES_DIR/Samples/1_Utilities/deviceQuery" >/dev/null 2>&1 \
    && cmake --build "$BUILD_DIR" -j"$(nproc)" >/dev/null 2>&1; then
    OUTPUT=$("$BUILD_DIR/deviceQuery" 2>&1)
    if echo "$OUTPUT" | grep -q "Result = PASS"; then
      echo "PASS: deviceQuery"
    else
      echo "FAIL: deviceQuery did not report PASS"
      echo "$OUTPUT" | tail -5
      FAILED=1
    fi
  else
    echo "FAIL: deviceQuery failed to compile"
    FAILED=1
  fi

  # vectorAdd
  echo "Building vectorAdd..."
  BUILD_DIR="$SAMPLES_DIR/build/vectorAdd"
  if cmake -B "$BUILD_DIR" -S "$SAMPLES_DIR/Samples/0_Introduction/vectorAdd" >/dev/null 2>&1 \
    && cmake --build "$BUILD_DIR" -j"$(nproc)" >/dev/null 2>&1; then
    OUTPUT=$("$BUILD_DIR/vectorAdd" 2>&1)
    if echo "$OUTPUT" | grep -q "PASSED"; then
      echo "PASS: vectorAdd"
    else
      echo "FAIL: vectorAdd did not report PASSED"
      echo "$OUTPUT" | tail -5
      FAILED=1
    fi
  else
    echo "FAIL: vectorAdd failed to compile"
    FAILED=1
  fi

  # matrixMul — tests CUDA kernel execution with shared memory
  echo "Building matrixMul..."
  BUILD_DIR="$SAMPLES_DIR/build/matrixMul"
  if cmake -B "$BUILD_DIR" -S "$SAMPLES_DIR/Samples/0_Introduction/matrixMul" >/dev/null 2>&1 \
    && cmake --build "$BUILD_DIR" -j"$(nproc)" >/dev/null 2>&1; then
    OUTPUT=$("$BUILD_DIR/matrixMul" 2>&1)
    if echo "$OUTPUT" | grep -q "Result = PASS"; then
      echo "PASS: matrixMul"
    else
      echo "FAIL: matrixMul did not report PASS"
      echo "$OUTPUT" | tail -5
      FAILED=1
    fi
  else
    echo "FAIL: matrixMul failed to compile"
    FAILED=1
  fi

  # simpleCUBLAS — tests cuBLAS library linkage and GPU GEMM
  echo "Building simpleCUBLAS..."
  BUILD_DIR="$SAMPLES_DIR/build/simpleCUBLAS"
  if cmake -B "$BUILD_DIR" -S "$SAMPLES_DIR/Samples/4_CUDA_Libraries/simpleCUBLAS" >/dev/null 2>&1 \
    && cmake --build "$BUILD_DIR" -j"$(nproc)" >/dev/null 2>&1; then
    OUTPUT=$("$BUILD_DIR/simpleCUBLAS" 2>&1)
    if echo "$OUTPUT" | grep -q "test passed"; then
      echo "PASS: simpleCUBLAS"
    else
      echo "FAIL: simpleCUBLAS did not report test passed"
      echo "$OUTPUT" | tail -5
      FAILED=1
    fi
  else
    echo "FAIL: simpleCUBLAS failed to compile"
    FAILED=1
  fi

  # simpleCUFFT — tests cuFFT library (FFT on GPU)
  echo "Building simpleCUFFT..."
  BUILD_DIR="$SAMPLES_DIR/build/simpleCUFFT"
  if cmake -B "$BUILD_DIR" -S "$SAMPLES_DIR/Samples/4_CUDA_Libraries/simpleCUFFT" >/dev/null 2>&1 \
    && cmake --build "$BUILD_DIR" -j"$(nproc)" >/dev/null 2>&1; then
    if "$BUILD_DIR/simpleCUFFT" >/dev/null 2>&1; then
      echo "PASS: simpleCUFFT"
    else
      echo "FAIL: simpleCUFFT exited with error"
      FAILED=1
    fi
  else
    echo "FAIL: simpleCUFFT failed to compile"
    FAILED=1
  fi

  # conjugateGradient — tests cuSPARSE library (sparse matrix solver)
  echo "Building conjugateGradient..."
  BUILD_DIR="$SAMPLES_DIR/build/conjugateGradient"
  if cmake -B "$BUILD_DIR" -S "$SAMPLES_DIR/Samples/4_CUDA_Libraries/conjugateGradient" >/dev/null 2>&1 \
    && cmake --build "$BUILD_DIR" -j"$(nproc)" >/dev/null 2>&1; then
    if "$BUILD_DIR/conjugateGradient" >/dev/null 2>&1; then
      echo "PASS: conjugateGradient"
    else
      echo "FAIL: conjugateGradient exited with error"
      FAILED=1
    fi
  else
    echo "FAIL: conjugateGradient failed to compile"
    FAILED=1
  fi
else
  echo "FAIL: could not clone cuda-samples ${SAMPLES_TAG}"
  FAILED=1
fi

rm -rf "$SAMPLES_DIR"

exit $FAILED
