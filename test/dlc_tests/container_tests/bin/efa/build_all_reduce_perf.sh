#!/bin/bash

# For dockerfiles of PyTorch >= 2.0, CUDA_HOME is already set as an env, and is configured as /opt/conda
python -c "import torch; from packaging.version import Version; assert Version(torch.__version__) >= Version('2.0')"
TORCH_VERSION_2x=$?
if [ $TORCH_VERSION_2x -ne 0 ]; then
  CUDA_HOME=/usr/local/cuda
fi

set -e

echo "Building all_reduce_perf from nccl-tests"
cd /tmp/
rm -rf nccl-tests/
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests/
make MPI=1 MPI_HOME=/opt/amazon/openmpi NCCL_HOME=/usr/local CUDA_HOME=${CUDA_HOME}
cp build/all_reduce_perf /all_reduce_perf
cd /tmp/
rm -rf nccl-tests/
