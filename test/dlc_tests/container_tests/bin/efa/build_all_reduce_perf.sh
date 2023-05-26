#!/bin/bash

set -e

echo "Building all_reduce_perf from nccl-tests"
cd /tmp/
rm -rf nccl-tests/
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests/
make MPI=1 MPI_HOME=/opt/amazon/openmpi NCCL_HOME=/usr/local CUDA_HOME=/usr/local/cuda
cp build/all_reduce_perf /all_reduce_perf
cd /tmp/
rm -rf nccl-tests/
