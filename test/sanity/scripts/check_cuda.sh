#!/usr/bin/env bash
set -euo pipefail

# Check CUDA version matches expected
# Runs inside the container (requires GPU)
#
# Usage: check_cuda.sh <cuda-version>
#   cuda-version: cuXYZ format (e.g., cu129)

CU_RAW="${1:?Usage: check_cuda.sh <cuda-version>}"

CU_MAJOR="${CU_RAW:2:${#CU_RAW}-3}"
CU_MINOR="${CU_RAW: -1}"
EXPECTED="${CU_MAJOR}.${CU_MINOR}"

NVCC_OUTPUT=$(nvcc --version 2>&1 || true)
EXPECTED_FLAT=$(echo "$EXPECTED" | tr -d '.')
NVCC_FLAT=$(echo "$NVCC_OUTPUT" | tr -d '.')

echo "Expected CUDA: $EXPECTED"
if ! echo "$NVCC_FLAT" | grep -q "$EXPECTED_FLAT"; then
  echo "FAIL: Expected CUDA $EXPECTED not found in nvcc output:"
  echo "$NVCC_OUTPUT"
  exit 1
fi

echo "CUDA version check passed"
