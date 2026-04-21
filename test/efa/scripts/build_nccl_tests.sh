#!/bin/bash
# Ported from V1: test/v2/ec2/efa/build_all_reduce_perf.sh
# Build all_reduce_perf from nccl-tests.
set -e

CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

echo "Building all_reduce_perf from nccl-tests"
cd /tmp/
rm -rf nccl-tests/
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests/
make MPI=1 MPI_HOME=/opt/amazon/openmpi NCCL_HOME=/usr/local CUDA_HOME=${CUDA_HOME}
cp build/all_reduce_perf /all_reduce_perf
cd /tmp/
rm -rf nccl-tests/
echo "Built /all_reduce_perf successfully"
