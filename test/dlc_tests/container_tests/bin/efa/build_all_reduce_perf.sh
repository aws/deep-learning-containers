#!/bin/bash

# For dockerfiles of PyTorch >= 2.0, CUDA_HOME is already set as an env, and is configured as /opt/conda
python -c "import torch; from packaging.version import Version; assert Version(torch.__version__) >= Version('2.0')"
TORCH_VERSION_2x=$?
if [ $TORCH_VERSION_2x -ne 0 ]; then
  CUDA_HOME=/usr/local/cuda
fi

set -e

# Pin nccl-tests to a known-good tag and build ONLY all_reduce_perf. A bare `make`
# builds every test binary, including comm_ops, whose recent revisions reference
# NCCL APIs (e.g. ncclCommGrow) and curand.h that are absent from the image's NCCL
# headers, which breaks the build. all_reduce_perf is the only binary this test needs.
NCCL_TESTS_VERSION=v2.20.0
echo "Building all_reduce_perf from nccl-tests ${NCCL_TESTS_VERSION}"
cd /tmp/
rm -rf nccl-tests/
git clone --branch ${NCCL_TESTS_VERSION} --depth 1 https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests/
make -C src BUILDDIR="$(pwd)/build" MPI=1 MPI_HOME=/opt/amazon/openmpi NCCL_HOME=/usr/local CUDA_HOME=${CUDA_HOME} "$(pwd)/build/all_reduce_perf"
cp build/all_reduce_perf /all_reduce_perf
cd /tmp/
rm -rf nccl-tests/
