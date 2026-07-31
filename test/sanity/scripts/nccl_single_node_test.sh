#!/usr/bin/env bash
set -uo pipefail

# Single-node NCCL functional check for base devel images. Requires GPU(s).
#
# Compiles and runs a minimal NCCL all-reduce against the image's own nccl.h /
# libnccl, which proves three things a file-existence check cannot:
#   1. nccl.h and libnccl are a matched, linkable pair (-lnccl resolves)
#   2. NCCL initialises a communicator against the installed CUDA runtime
#   3. Collective results are numerically correct
#
# Works with one GPU (ncclCommInitAll over a single device) and scales to all
# visible GPUs when the runner has more.
#
# Multi-node NCCL over EFA/Libfabric with GDRDMA is not covered here — that
# needs EFA hardware and lives in test/efa/test_efa.py, whose
# test/efa/scripts/nccl_allreduce.sh is a superset of this check (it also
# validates the EFA transport and all_reduce bandwidth). This script is the
# cheap always-on guard that runs on the existing single-GPU sanity runner.
#
# Usage: nccl_single_node_test.sh

FAILED=0
WORK_DIR=$(mktemp -d)
trap 'rm -rf "$WORK_DIR"' EXIT

if ! nvidia-smi &>/dev/null; then
  echo "FAIL: nvidia-smi not found or no GPUs detected"
  exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected ${GPU_COUNT} GPU(s)"

# Locate nccl.h — the devel image installs libnccl-devel into /usr/include.
NCCL_INCLUDE=""
for CANDIDATE in /usr/include /usr/local/cuda/include /usr/local/include; do
  if [ -f "${CANDIDATE}/nccl.h" ]; then
    NCCL_INCLUDE="$CANDIDATE"
    break
  fi
done
if [ -z "$NCCL_INCLUDE" ]; then
  echo "FAIL: nccl.h not found; cannot compile the NCCL test"
  exit 1
fi
echo "Using nccl.h from ${NCCL_INCLUDE}"

cat >"${WORK_DIR}/nccl_allreduce.cu" <<'EOF'
// Minimal single-process, multi-device NCCL all-reduce.
// Device i contributes (i + 1); every device must end up with the sum.
#include <cuda_runtime.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(cmd)                                                       \
  do {                                                                        \
    cudaError_t e = (cmd);                                                     \
    if (e != cudaSuccess) {                                                    \
      printf("CUDA error %s:%d '%s'\n", __FILE__, __LINE__,                     \
             cudaGetErrorString(e));                                           \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define NCCL_CHECK(cmd)                                                       \
  do {                                                                        \
    ncclResult_t r = (cmd);                                                    \
    if (r != ncclSuccess) {                                                    \
      printf("NCCL error %s:%d '%s'\n", __FILE__, __LINE__,                     \
             ncclGetErrorString(r));                                           \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main() {
  int version = 0;
  NCCL_CHECK(ncclGetVersion(&version));
  printf("NCCL runtime version code: %d\n", version);

  int num_devices = 0;
  CUDA_CHECK(cudaGetDeviceCount(&num_devices));
  if (num_devices < 1) {
    printf("no CUDA devices visible\n");
    return 1;
  }
  printf("using %d device(s)\n", num_devices);

  const size_t count = 1024 * 1024;  // 4 MiB of floats per device
  float expected = 0.0f;
  for (int i = 0; i < num_devices; i++) expected += (float)(i + 1);

  int *devices = (int *)malloc(num_devices * sizeof(int));
  float **send = (float **)malloc(num_devices * sizeof(float *));
  float **recv = (float **)malloc(num_devices * sizeof(float *));
  cudaStream_t *streams =
      (cudaStream_t *)malloc(num_devices * sizeof(cudaStream_t));

  for (int i = 0; i < num_devices; i++) {
    devices[i] = i;
    CUDA_CHECK(cudaSetDevice(i));
    CUDA_CHECK(cudaMalloc(&send[i], count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&recv[i], count * sizeof(float)));
    CUDA_CHECK(cudaStreamCreate(&streams[i]));
    // memset can only write byte patterns, so fill from the host instead.
    float *host = (float *)malloc(count * sizeof(float));
    for (size_t j = 0; j < count; j++) host[j] = (float)(i + 1);
    CUDA_CHECK(cudaMemcpy(send[i], host, count * sizeof(float),
                          cudaMemcpyHostToDevice));
    free(host);
  }

  ncclComm_t *comms = (ncclComm_t *)malloc(num_devices * sizeof(ncclComm_t));
  NCCL_CHECK(ncclCommInitAll(comms, num_devices, devices));
  printf("ncclCommInitAll succeeded\n");

  NCCL_CHECK(ncclGroupStart());
  for (int i = 0; i < num_devices; i++) {
    NCCL_CHECK(ncclAllReduce(send[i], recv[i], count, ncclFloat, ncclSum,
                             comms[i], streams[i]));
  }
  NCCL_CHECK(ncclGroupEnd());

  for (int i = 0; i < num_devices; i++) {
    CUDA_CHECK(cudaSetDevice(i));
    CUDA_CHECK(cudaStreamSynchronize(streams[i]));
  }
  printf("ncclAllReduce completed\n");

  // Verify every device's full buffer, not just the first element — a partial
  // or misconfigured collective can leave the tail untouched.
  int ok = 1;
  for (int i = 0; i < num_devices && ok; i++) {
    CUDA_CHECK(cudaSetDevice(i));
    float *host = (float *)malloc(count * sizeof(float));
    CUDA_CHECK(
        cudaMemcpy(host, recv[i], count * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t j = 0; j < count; j++) {
      if (host[j] != expected) {
        printf("device %d element %zu: got %f, expected %f\n", i, j, host[j],
               expected);
        ok = 0;
        break;
      }
    }
    free(host);
  }

  for (int i = 0; i < num_devices; i++) {
    CUDA_CHECK(cudaSetDevice(i));
    NCCL_CHECK(ncclCommDestroy(comms[i]));
    CUDA_CHECK(cudaStreamDestroy(streams[i]));
    CUDA_CHECK(cudaFree(send[i]));
    CUDA_CHECK(cudaFree(recv[i]));
  }

  if (!ok) {
    printf("all_reduce result mismatch\n");
    return 1;
  }
  printf("all_reduce result correct (expected %f on every device)\n", expected);
  printf("Result = PASS\n");
  return 0;
}
EOF

echo "Compiling NCCL all-reduce test..."
COMPILE_OUT=$(nvcc -I"${NCCL_INCLUDE}" -o "${WORK_DIR}/nccl_allreduce" \
  "${WORK_DIR}/nccl_allreduce.cu" -lnccl -lcudart 2>&1)
COMPILE_RC=$?
if [ $COMPILE_RC -ne 0 ]; then
  echo "FAIL: NCCL test failed to compile (nvcc -lnccl)"
  echo "$COMPILE_OUT"
  exit 1
fi
echo "PASS: compiled against nccl.h + libnccl (-lnccl resolved)"

# NCCL_DEBUG=WARN keeps the output small while still surfacing init problems.
echo "Running NCCL all-reduce..."
RUN_OUT=$(NCCL_DEBUG=WARN "${WORK_DIR}/nccl_allreduce" 2>&1)
RUN_RC=$?
echo "$RUN_OUT"
if [ $RUN_RC -eq 0 ] && echo "$RUN_OUT" | grep -q "Result = PASS"; then
  echo "PASS: NCCL all-reduce across ${GPU_COUNT} GPU(s)"
else
  echo "FAIL: NCCL all-reduce did not pass (exit=${RUN_RC})"
  FAILED=1
fi

exit $FAILED
