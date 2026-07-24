#!/bin/bash
set -eo pipefail

ARCH_TYPE=$1
PROCESSOR_TYPE=$2
PYTORCH_VERSION=$3

###############################################################################
# activate conda env
###############################################################################

# if [ "$ARCH_TYPE" == "x86" ]; then
#     CONDA_ENV_NAME=${ARCH_TYPE}_${PROCESSOR_TYPE}_env
# else
#     CONDA_ENV_NAME=aarch64_env
# fi

CONDA_ENV_NAME=${ARCH_TYPE}_${PROCESSOR_TYPE}_env

PATH=/opt/conda/bin:$PATH
LD_LIBRARY_PATH=/opt/conda/envs/${CONDA_ENV_NAME}/lib/:/opt/conda/lib:$LD_LIBRARY_PATH
source /opt/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_NAME}"


###############################################################################
# clone repo and install dependencies
###############################################################################

cd /
git clone https://github.com/pytorch/pytorch -b v${PYTORCH_VERSION} --depth 1 --shallow-submodules --recurse-submodules
cd /pytorch
pip install -r requirements.txt
pip install pytest scipy pytest-rerunfailures pytest-shard pytest-flakefinder pytest-xdist


# fix for missing libnvrtc-builtins.so.12.4
if [[ "$PROCESSOR_TYPE" == cu* ]]; then
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
fi

###############################################################################
# run unit tests
###############################################################################

echo "Running unit tests. This will take a while..."
cd /pytorch/
mkdir -p /reports
TIMESTAMP=$(date '+%Y-%m-%d-%H-%M-%S')
LOG_NAME=unit_tests_${TIMESTAMP}.log

if [ "$ARCH_TYPE" == "x86" ]; then

# CORE_TEST_LIST = [
#     "test_autograd",
#     "test_autograd_fallback",
#     "test_modules",
#     "test_nn",
#     "test_ops",
#     "test_ops_gradients",
#     "test_ops_fwd_gradients",
#     "test_ops_jit",
#     "test_torch",
# ]
    python test/run_test.py --keep-going --core -x doctests 2>&1 3>&1 | tee /reports/${LOG_NAME} || true
else
    tests=(
        test_modules
        test_mkldnn
        test_mkldnn_fusion
        test_openmp
        test_torch
        test_dynamic_shapes
        test_transformers
        test_multiprocessing
        test_numpy_interop
    )
    for test in ${tests[@]}; do
        python test/run_test.py --verbose --keep-going --include ${test} 2>&1 | tee -a /reports/${LOG_NAME} || true
    done
fi
echo "Unit Testing Complete. Please check the log for results..."