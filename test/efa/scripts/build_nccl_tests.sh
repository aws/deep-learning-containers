#!/bin/bash
# Ported from V1: test/v2/ec2/efa/build_all_reduce_perf.sh
# Build all_reduce_perf from nccl-tests.
set -e

CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

# Runtime image has cuda-nvcc but not CUDA headers — install cuda-cudart-devel
CUDA_MAJOR_MINOR=$(nvcc --version | grep -oP 'V\K[0-9]+\.[0-9]+' | tr '.' '-')
dnf install -y -q cuda-cudart-devel-${CUDA_MAJOR_MINOR}

echo "Building all_reduce_perf from nccl-tests"
cd /tmp/
rm -rf nccl-tests/
# Download tarball instead of git clone — runtime image may not have git
curl -fsSL https://github.com/NVIDIA/nccl-tests/archive/refs/heads/master.tar.gz | tar xz
mv nccl-tests-master nccl-tests
cd nccl-tests/
make MPI=1 MPI_HOME=/opt/amazon/openmpi NCCL_HOME=/usr/local CUDA_HOME=${CUDA_HOME}
cp build/all_reduce_perf /all_reduce_perf
cd /tmp/
rm -rf nccl-tests/
echo "Built /all_reduce_perf successfully"
