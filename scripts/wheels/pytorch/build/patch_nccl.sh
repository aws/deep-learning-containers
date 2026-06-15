#!/bin/bash
set -xeo pipefail

DESIRED_NCCL_TAG=$1

# delete existing nccl files
find /usr/local/cuda/include -follow \( -name "*nccl*" -type f -o -name "*nccl*" -type l \) -print -delete
find /usr/local/cuda/lib64 -follow \( -name "*nccl*" -type f -o -name "*nccl*" -type l \) -print -delete

# build nccl
BUILD_DIR=/tmp/nccl
git clone -b ${DESIRED_NCCL_TAG} --depth 1 https://github.com/NVIDIA/nccl.git $BUILD_DIR
cd $BUILD_DIR && make -j src.build
cp -a build/include/* /usr/local/cuda/include/
cp -a build/lib/* /usr/local/cuda/lib64/
rm -rf $BUILD_DIR

# print nccl files
find /usr/local/cuda/include -follow \( -name "*nccl*" -type f -o -name "*nccl*" -type l \) -print
find /usr/local/cuda/lib64 -follow \( -name "*nccl*" -type f -o -name "*nccl*" -type l \) -print
