#!/bin/bash
set -eo pipefail

PROCESSOR_TYPE=$1
TORCHVISION_VERSION=$2
TORCHAUDIO_VERSION=$3
TORCHTEXT_VERSION=$4
TORCHDATA_VERSION=$5
ARCH_TYPE=$6
PYTHON_VERSION=$7

# ===========================
# Install conda if not present
# ===========================
if [[ ! -e /opt/conda ]]; then
    if [ "$ARCH_TYPE" == "x86" ]; then
        MAMBA_FORGE_INSTALLER_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    else
        MAMBA_FORGE_INSTALLER_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh"
    fi
    MAMBA_FORGE_INSTALLER=miniforge.sh
    wget ${MAMBA_FORGE_INSTALLER_URL} -O ${MAMBA_FORGE_INSTALLER}
    bash ${MAMBA_FORGE_INSTALLER} -b -p /opt/conda
    rm ${MAMBA_FORGE_INSTALLER}
    /opt/conda/bin/conda init bash
fi

# Always initialize conda in current shell
eval "$(/opt/conda/bin/conda shell.bash hook)"
conda config --set channel_priority false

# ===========================
# Create build env
# arm64+CUDA: install cuda-toolkit via conda (provides nvcc + headers + stubs)
# x86+CUDA: system CUDA at /usr/local/cuda-X.Y is pre-installed in container
# ===========================
CONDA_ENV_NAME=build_tools_env

if [[ "$ARCH_TYPE" == "arm64" && "$PROCESSOR_TYPE" != "cpu" ]]; then
    CUDA_VERSION=${PROCESSOR_TYPE:2:2}.${PROCESSOR_TYPE:4}
    conda create --yes --quiet -n "${CONDA_ENV_NAME}" "python=${PYTHON_VERSION}" \
        "cmake<4" ninja pkg-config wheel \
        "cuda-toolkit=${CUDA_VERSION}" \
        -c nvidia
else
    conda create --yes --quiet -n "${CONDA_ENV_NAME}" "python=${PYTHON_VERSION}" \
        cmake ninja pkg-config wheel
fi

# ===========================
# Activate conda env
# ===========================
eval "$(/opt/conda/bin/conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"

# ===========================
# Set up CUDA environment
# ===========================
export CMAKE_SHARED_LINKER_FLAGS="-Wl,-z,max-page-size=0x10000"

if [ "$PROCESSOR_TYPE" != "cpu" ]; then
    CUDA_VERSION=${PROCESSOR_TYPE:2:2}.${PROCESSOR_TYPE:4}
    export FORCE_CUDA=1

    if [[ "$ARCH_TYPE" == "arm64" ]]; then
        # CUDA is in the conda env (no system CUDA in aarch64 manywheel container)
        export CUDA_HOME="${CONDA_PREFIX}"
        export PATH="${CONDA_PREFIX}/bin:${PATH}"
        export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/targets/sbsa-linux/lib:${LD_LIBRARY_PATH:-}"
        export LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/targets/sbsa-linux/lib:${LIBRARY_PATH:-}"
        export CPATH="${CONDA_PREFIX}/targets/sbsa-linux/include:${CPATH:-}"
        # Match the arch list from build_cuda.sh 13.0 case after aarch64 filtering
        # No +PTX: upstream dropped it for CUDA 13.0 to reduce binary size
        export TORCH_CUDA_ARCH_LIST="7.5;8.0;9.0;10.0;11.0;12.0"
    else
        export CUDA_HOME="/usr/local/cuda-${CUDA_VERSION}"
        export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0+PTX"
    fi
fi

# ===========================
# Install torch + nvidia PyPI CUDA packages
# On arm64+CUDA, torch wheel declares nvidia-* as pip deps but doesn't auto-install
# them. cmake's find_package(Torch)->find_package(CUDAToolkit) needs the .so files.
# Package names match exactly what build_torch.sh sets in PYTORCH_EXTRA_INSTALL_REQUIREMENTS
# ===========================
pip install /artifacts/*.whl

if [[ "$ARCH_TYPE" == "arm64" && "$PROCESSOR_TYPE" != "cpu" ]]; then
    pip install \
        "nvidia-cuda-nvrtc==13.0.88" \
        "nvidia-cuda-runtime==13.0.96" \
        "nvidia-cuda-cupti==13.0.85" \
        "nvidia-cudnn-cu13==9.19.0.56" \
        "nvidia-cublas==13.1.0.3" \
        "nvidia-cufft==12.0.0.61" \
        "nvidia-curand==10.4.0.35" \
        "nvidia-cusolver==12.0.4.66" \
        "nvidia-cusparse==12.6.3.3" \
        "nvidia-cusparselt-cu13==0.8.0" \
        "nvidia-nccl-cu13==2.29.7" \
        "nvidia-nvtx==13.0.85" \
        "nvidia-nvjitlink==13.0.88"

    # Extend library paths to include nvidia PyPI package libs
    # so cmake linker can find libcudart, libcublas, libcudnn etc.
    SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
    for pkg_dir in "${SITE_PACKAGES}"/nvidia/*/lib; do
        if [[ -d "$pkg_dir" ]]; then
            export LD_LIBRARY_PATH="${pkg_dir}:${LD_LIBRARY_PATH:-}"
            export LIBRARY_PATH="${pkg_dir}:${LIBRARY_PATH:-}"
        fi
    done
fi

export PYTORCH_VERSION=$(pip show torch | grep ^Version: | sed 's/Version: *//' | sed 's/+.\+//')

# ===========================
# Build TorchVision
# ===========================
echo "Building TorchVision wheel"
cd /
git clone https://github.com/pytorch/vision -b v${TORCHVISION_VERSION} --depth 1 --shallow-submodules
cd /vision
bash packaging/pre_build_script.sh
BUILD_VERSION=${TORCHVISION_VERSION}+${PROCESSOR_TYPE} python3 setup.py bdist_wheel
if [ "$PROCESSOR_TYPE" != "cpu" ]; then
    bash packaging/post_build_script.sh
fi
cp /vision/dist/* /artifacts/

# ===========================
# Build TorchAudio
# ===========================
echo "Building TorchAudio wheel"

# Upgrade setuptools to >= 72.2.0 for PathLike support in Extension sources
pip install "setuptools>=72.2.0"

cd /
git clone https://github.com/pytorch/audio -b v${TORCHAUDIO_VERSION} --depth 1 --shallow-submodules
cd /audio
# FFmpeg build (only if the build script exists - removed in torchaudio >= 2.10.0)
if [[ -f ".github/scripts/ffmpeg/build.sh" ]]; then
    export FFMPEG_VERSION="6.1.2"
    export FFMPEG_ROOT="/audio/third_party/ffmpeg"
    sed -i "s/curl -LsS -o/wget -O/g" .github/scripts/ffmpeg/build.sh
    for attempt in 1 2 3; do
        bash .github/scripts/ffmpeg/build.sh && break
        echo "ffmpeg attempt ${attempt} failed, retrying in 15s..."
        sleep 15
    done
    export USE_FFMPEG=1
fi

if [[ "$ARCH_TYPE" == "arm64" && "$PROCESSOR_TYPE" != "cpu" ]]; then
    # Symlink CUDA headers so cmake FindCUDA finds device_functions.h in standard path
    ln -sf "${CONDA_PREFIX}/targets/sbsa-linux/include/device_functions.h" "${CONDA_PREFIX}/include/device_functions.h"
    ln -sf "${CONDA_PREFIX}/targets/sbsa-linux/include/cuda_runtime.h" "${CONDA_PREFIX}/include/cuda_runtime.h"
    ln -sf "${CONDA_PREFIX}/targets/sbsa-linux/include/cuda.h" "${CONDA_PREFIX}/include/cuda.h"
    # Fix cmake CUDA args: remove spurious quotes, add CUDA_INCLUDE_DIRS
    sed -i "s|f\"-DCMAKE_CUDA_COMPILER='{CUDA_HOME}/bin/nvcc'\"|f\"-DCMAKE_CUDA_COMPILER={CUDA_HOME}/bin/nvcc\"|g" tools/setup_helpers/extension.py
    sed -i "s|f\"-DCUDA_TOOLKIT_ROOT_DIR='{CUDA_HOME}'\"|f\"-DCUDA_TOOLKIT_ROOT_DIR={CUDA_HOME}\"|g" tools/setup_helpers/extension.py
    sed -i '/DCUDA_TOOLKIT_ROOT_DIR/a\            cmake_args += [f"-DCUDA_INCLUDE_DIRS={CUDA_HOME}/targets/sbsa-linux/include"]' tools/setup_helpers/extension.py
    sed -i '/DCUDA_INCLUDE_DIRS/a\            cmake_args += ["-DCMAKE_CUDA_FLAGS=-Xcompiler=-fpermissive"]' tools/setup_helpers/extension.py
    # Patch CUB APIs removed in CUDA 13
    sed -i 's/cub::FpLimits<float>::Lowest()/-std::numeric_limits<float>::infinity()/g' src/libtorchaudio/cuctc/src/ctc_prefix_decoder_kernel_v2.cu
    sed -i 's/cub::Max()/cuda::maximum<scalar_t>{}/g' src/libtorchaudio/forced_align/gpu/compute.cu
    sed -i '/#include <cub\/cub.cuh>/a #include <cuda\/functional>' src/libtorchaudio/forced_align/gpu/compute.cu
fi

BUILD_VERSION=${TORCHAUDIO_VERSION}+${PROCESSOR_TYPE} python3 setup.py bdist_wheel
cp /audio/dist/* /artifacts/

# ===========================
# Build TorchText
# ===========================
echo "Building TorchText wheel"
cd /
git clone https://github.com/pytorch/text -b v${TORCHTEXT_VERSION} --depth 1 --shallow-submodules
cd /text
BUILD_VERSION=${TORCHTEXT_VERSION}+${PROCESSOR_TYPE} python3 setup.py bdist_wheel
cp /text/dist/* /artifacts/

# ===========================
# Build TorchData
# ===========================
echo "Building TorchData wheel"
cd /
git clone https://github.com/pytorch/data -b v${TORCHDATA_VERSION} --depth 1 --shallow-submodules
cd /data
BUILD_VERSION=${TORCHDATA_VERSION}+${PROCESSOR_TYPE} python3 setup.py bdist_wheel
cp /data/dist/* /artifacts/
