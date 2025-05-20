#!/bin/bash

set -ex

# https://raw.githubusercontent.com/pytorch/pytorch/842d51500be144d53f4d046d31169e8f46c063f6/.ci/docker/common/install_cuda.sh

function prune_cuda {
    # Remove non-essential CUDA components to reduce image size:
    # - Documentation and manual pages
    # - Sample code, demos, and example projects
    # - IDE integration (Nsight Eclipse Edition)
    # - Debugging tools (compute-sanitizer, debugger)
    # - Profiling tools (Nsight Compute, Nsight Systems)
    # - Legacy tools (Visual Profiler)
    # This keeps only the essential runtime libraries, headers and development tools
    rm -rf /usr/local/cuda/compute-sanitizer/docs \
        /usr/local/cuda/nsight-compute-****.*.*/docs \
        /usr/local/cuda/nsight-systems-****.*.*/documentation \
        /usr/local/cuda/extras/demo_suite \
        /usr/local/cuda/extras/CUPTI/samples \
        /usr/local/cuda/nsight-compute-****.*.*/extras/samples \
        /usr/local/cuda/libnvvp \
        /usr/local/cuda/nsightee_plugins \
        /usr/local/cuda/compute-sanitizer \
        /usr/local/cuda/extras/Debugger \
        /usr/local/cuda/nsight-compute-****.*.* \
        /usr/local/cuda/nsight-systems-****.*.*
    rm -rf /usr/local/cuda/doc
    rm -rf /usr/local/cuda/samples
    rm -rf /usr/local/cuda/share/doc
}

function install_cuda118_stack {
    CUDNN_VERSION=9.1.0.70
    NCCL_VERSION=v2.21.5-1
    CUDA_HOME="/usr/local/cuda"
    
    # move cuda-compt and remove existing cuda dir from nvidia/cuda:**.*.*-base-*
    mv /usr/local/cuda/compat /usr/local
    rm -rf /usr/local/cuda-*
    rm -rf /usr/local/cuda
    
    # install cuda
    cd /tmp
    wget -q https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
    chmod +x cuda_11.8.0_520.61.05_linux.run
    ./cuda_11.8.0_520.61.05_linux.run --toolkit --silent
    rm -f cuda_11.8.0_520.61.05_linux.run
    ln -s /usr/local/cuda-11.8 /usr/local/cuda
    # bring back cuda-compat
    mv /usr/local/compat /usr/local/cuda/compat

    # install cudnn
    mkdir -p /tmp/cudnn
    cd /tmp/cudnn
    wget -q https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-${CUDNN_VERSION}_cuda11-archive.tar.xz -O cudnn-linux-x86_64-${CUDNN_VERSION}_cuda11-archive.tar.xz
    tar xf cudnn-linux-x86_64-${CUDNN_VERSION}_cuda11-archive.tar.xz
    cp -a cudnn-linux-x86_64-${CUDNN_VERSION}_cuda11-archive/include/* /usr/local/cuda/include/
    cp -a cudnn-linux-x86_64-${CUDNN_VERSION}_cuda11-archive/lib/* /usr/local/cuda/lib64/

    # install nccl
    mkdir -p /tmp/nccl
    cd /tmp/nccl
    git clone -b $NCCL_VERSION --depth 1 https://github.com/NVIDIA/nccl.git
    cd nccl 
    make -j src.build
    cp -a build/include/* /usr/local/cuda/include/
    cp -a build/lib/* /usr/local/cuda/lib64/

    prune_cuda
    ldconfig
}


function install_cuda124_stack {
    CUDNN_VERSION="9.1.0.70"
    NCCL_VERSION="v2.23.4-1"
    CUDA_HOME="/usr/local/cuda"
    
    # move cuda-compt and remove existing cuda dir from nvidia/cuda:**.*.*-base-*
    mv /usr/local/cuda/compat /usr/local
    rm -rf /usr/local/cuda-*
    rm -rf /usr/local/cuda
    
    # install cuda
    cd /tmp
    wget -q https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run
    chmod +x cuda_12.4.1_550.54.15_linux.run
    ./cuda_12.4.1_550.54.15_linux.run --toolkit --silent
    rm -f cuda_12.4.1_550.54.15_linux.run
    ln -s /usr/local/cuda-12.4 /usr/local/cuda
    # bring back cuda-compat
    mv /usr/local/compat /usr/local/cuda/compat

    # install cudnn
    mkdir -p /tmp/cudnn
    cd /tmp/cudnn
    wget -q https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive.tar.xz -O cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive.tar.xz
    tar xf cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive.tar.xz
    cp -a cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive/include/* /usr/local/cuda/include/
    cp -a cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive/lib/* /usr/local/cuda/lib64/

    # install nccl
    mkdir -p /tmp/nccl
    cd /tmp/nccl
    git clone -b $NCCL_VERSION --depth 1 https://github.com/NVIDIA/nccl.git
    cd nccl 
    make -j src.build
    cp -a build/include/* /usr/local/cuda/include/
    cp -a build/lib/* /usr/local/cuda/lib64/

    prune_cuda
    ldconfig
}


function install_cuda126_stack {
    CUDNN_VERSION="9.7.0.66"
    NCCL_VERSION="v2.23.4-1"
    CUDA_HOME="/usr/local/cuda"
    
    # move cuda-compt and remove existing cuda dir from nvidia/cuda:**.*.*-base-*
    mv /usr/local/cuda/compat /usr/local
    rm -rf /usr/local/cuda-*
    rm -rf /usr/local/cuda

    # install CUDA
    wget -q https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda_12.6.3_560.35.05_linux.run
    chmod +x cuda_12.6.3_560.35.05_linux.run
    ./cuda_12.6.3_560.35.05_linux.run --toolkit --silent
    rm -f cuda_12.6.3_560.35.05_linux.run
    ln -s /usr/local/cuda-12.6 /usr/local/cuda
    # bring back cuda-compat
    mv /usr/local/compat /usr/local/cuda/compat

    # install cudnn
    mkdir -p /tmp/cudnn
    cd /tmp/cudnn
    wget -q https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive.tar.xz -O cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive.tar.xz
    tar xf cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive.tar.xz
    cp -a cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive/include/* /usr/local/cuda/include/
    cp -a cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive/lib/* /usr/local/cuda/lib64/

    # install nccl
    mkdir -p /tmp/nccl
    cd /tmp/nccl
    git clone -b $NCCL_VERSION --depth 1 https://github.com/NVIDIA/nccl.git
    cd nccl 
    make -j src.build
    cp -a build/include/* /usr/local/cuda/include/
    cp -a build/lib/* /usr/local/cuda/lib64/

    prune_cuda
    ldconfig
}


function install_cuda128_stack {
    CUDNN_VERSION="9.8.0.87"
    NCCL_VERSION="v2.26.2-1"
    CUDA_HOME="/usr/local/cuda"
    
    # move cuda-compt and remove existing cuda dir from nvidia/cuda:**.*.*-base-*
    mv /usr/local/cuda/compat /usr/local
    rm -rf /usr/local/cuda-*
    rm -rf /usr/local/cuda

    # install CUDA
    wget -q https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
    chmod +x cuda_12.8.1_570.124.06_linux.run
    ./cuda_12.8.1_570.124.06_linux.run --toolkit --silent
    rm -f cuda_12.8.1_570.124.06_linux.run
    ln -s /usr/local/cuda-12.8 /usr/local/cuda
    # bring back cuda-compat
    mv /usr/local/compat /usr/local/cuda/compat

    # install cudnn
    mkdir -p /tmp/cudnn
    cd /tmp/cudnn
    wget -q https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive.tar.xz -O cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive.tar.xz
    tar xf cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive.tar.xz
    cp -a cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive/include/* /usr/local/cuda/include/
    cp -a cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive/lib/* /usr/local/cuda/lib64/

    # install nccl
    mkdir -p /tmp/nccl
    cd /tmp/nccl
    git clone -b $NCCL_VERSION --depth 1 https://github.com/NVIDIA/nccl.git
    cd nccl 
    make -j src.build
    cp -a build/include/* /usr/local/cuda/include/
    cp -a build/lib/* /usr/local/cuda/lib64/

    prune_cuda
    ldconfig
}

# idiomatic parameter and option handling in sh
while test $# -gt 0
do
    case "$1" in
    11.8) install_cuda118_stack; 
        ;;
    12.4) install_cuda124_stack; 
        ;;
    12.6) install_cuda126_stack; 
        ;;
    12.8) install_cuda128_stack;
        ;;
    *) echo "bad argument $1"; exit 1
        ;;
    esac
    shift
done
