#!/bin/bash
set -eo pipefail

ARCH_TYPE=$1
PROCESSOR_TYPE=$2
PYTORCH_VERSION=$3
PYTHON_VERSION=$4

###############################################################################
# Install conda
###############################################################################

echo 'Installing conda-forge'

CONDA_ENV_NAME=${ARCH_TYPE}_${PROCESSOR_TYPE}_env

if [[ ! -e /opt/conda ]]; then
    if [ "$ARCH_TYPE" == "x86" ]; then
        MAMBA_FORGE_INSTALLER_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    else
        # comment out this part cause release/2.4 use an unavailable numpy, and release is not related to PYTORCH_VERSION
        # # run aarch64 setup script
        # cd /
        # git clone https://github.com/pytorch/builder -b release/${PYTORCH_VERSION%.*}
        # export DESIRED_PYTHON=$PYTHON_VERSION
        # bash /builder/aarch64_linux/aarch64_ci_setup.sh
        # CONDA_ENV_NAME=aarch64_env
        MAMBA_FORGE_INSTALLER_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh"
    fi
    MAMBA_FORGE_INSTALLER=miniforge.sh
    wget ${MAMBA_FORGE_INSTALLER_URL} -O ${MAMBA_FORGE_INSTALLER}
    bash ${MAMBA_FORGE_INSTALLER} -b -p /opt/conda
    rm -f ${MAMBA_FORGE_INSTALLER}
    /opt/conda/bin/conda init bash
    eval "$(/opt/conda/bin/conda shell.bash hook)"
    conda config --set channel_priority false
fi
echo 'Conda-forge is installed successfully'

conda create --yes --quiet -n "${CONDA_ENV_NAME}" "python=${PYTHON_VERSION}"
PATH=/opt/conda/bin:$PATH
LD_LIBRARY_PATH=/opt/conda/envs/${CONDA_ENV_NAME}/lib/:/opt/conda/lib:$LD_LIBRARY_PATH
source /opt/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_NAME}"


###############################################################################
# Install wheels
###############################################################################

echo 'Installing wheels'
for wheel in /wheels/*.whl; do
    pip install ${wheel}
done
echo 'Wheels are installed successfully'

if [[ "$PROCESSOR_TYPE" == cu* ]] && [ "$ARCH_TYPE" == "x86" ]; then
    pip install triton
fi