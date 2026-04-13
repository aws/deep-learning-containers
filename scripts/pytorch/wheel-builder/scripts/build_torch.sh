#!/bin/bash
set -eo pipefail

PROCESSOR_TYPE=$1   # "cpu" or something like "cu124"
PYTORCH_VERSION=$2  # something like "2.6.0"
PYTHON_VERSION=$3   # something like "3.12"
ARCH_TYPE=$4        # "x86" or "arm64"

# clone pytorch repository and submodules
cd /
git clone https://github.com/pytorch/pytorch -b v${PYTORCH_VERSION} --recurse-submodules --quiet

# Apply SM 7.5 patch for arm64 cu130 builds (Graviton + T4 GPU support)
if [[ "$PYTORCH_VERSION" == "2.11.0" && "$ARCH_TYPE" == "arm64" && "$PROCESSOR_TYPE" == "cu130" ]]; then
    git -C /pytorch apply /scripts/patches/pytorch-2.11.0-sm75-aarch64.patch
fi

# inject telemetry
cat /scripts/aws_telemetry.py >> /pytorch/torch/__init__.py

# set common environment variables
export PYTORCH_ROOT="/pytorch"
export PACKAGE_TYPE="manywheel"
export DESIRED_CUDA=$PROCESSOR_TYPE
export DESIRED_PYTHON=$PYTHON_VERSION
export PYTORCH_BUILD_VERSION="${PYTORCH_VERSION}+${PROCESSOR_TYPE}"
export PYTORCH_FINAL_PACKAGE_DIR="/artifacts"
export MAX_JOBS=12
export SKIP_ALL_TESTS="1"

# build torch wheel
if [ "$ARCH_TYPE" == "x86" ]; then
    export USE_FBGEMM=1
    export ARCH="x86_64"
    export PYTORCH_EXTRA_INSTALL_REQUIREMENTS=""
    if [ $PROCESSOR_TYPE == "cpu" ]; then
        export GPU_ARCH_TYPE="cpu"
    else
        export GPU_ARCH_TYPE="cuda"
        # customize nccl version
        # nccl version for patch
        DESIRED_NCCL_VERSION=2.23.4-1
        cd /pytorch/third_party/nccl/nccl
        git checkout v${DESIRED_NCCL_VERSION}
        bash /scripts/patch_nccl.sh v${DESIRED_NCCL_VERSION}
    fi
else # arm64
    if [ $PROCESSOR_TYPE == "cpu" ]; then
        export GPU_ARCH_TYPE="cpu-aarch64"
    else
        export GPU_ARCH_TYPE="cuda-aarch64"
        export GPU_ARCH_VERSION=${PROCESSOR_TYPE:2:2}.${PROCESSOR_TYPE:4}
        # TORCH_CUDA_ARCH_LIST is set to "7.5;8.0;9.0;10.0;11.0;12.0" through patch file
        # Use PyPI NVIDIA packages
        export PYTORCH_EXTRA_INSTALL_REQUIREMENTS="nvidia-cuda-nvrtc==13.0.88; platform_system == 'Linux' | nvidia-cuda-runtime==13.0.96; platform_system == 'Linux' | nvidia-cuda-cupti==13.0.85; platform_system == 'Linux' | nvidia-cudnn-cu13==9.19.0.56; platform_system == 'Linux' | nvidia-cublas==13.1.0.3; platform_system == 'Linux' | nvidia-cufft==12.0.0.61; platform_system == 'Linux' | nvidia-curand==10.4.0.35; platform_system == 'Linux' | nvidia-cusolver==12.0.4.66; platform_system == 'Linux' | nvidia-cusparse==12.6.3.3; platform_system == 'Linux' | nvidia-cusparselt-cu13==0.8.0; platform_system == 'Linux' | nvidia-nccl-cu13==2.29.7; platform_system == 'Linux' | nvidia-nvshmem-cu13==3.4.5; platform_system == 'Linux' | nvidia-nvtx==13.0.85; platform_system == 'Linux' | nvidia-nvjitlink==13.0.88; platform_system == 'Linux' | nvidia-cufile==1.15.1.6; platform_system == 'Linux'"
    fi
fi

cd /
bash /pytorch/.ci/manywheel/build.sh
