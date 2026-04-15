#!/bin/bash

set -xeo pipefail

while test $# -gt 0
do
    case "$1" in
        --arch_type) shift; ARCH_TYPE="$1"                      # "x86" or "arm64"
        ;;
        --processor_type) shift; PROCESSOR_TYPE="$1"            # "cpu" or something like "cu124" (only 12.4 is available currently for arm64)
        ;;
        --pytorch_version) shift; PYTORCH_VERSION="$1"          # e.g. "2.4.0"
        ;;
        --python_version) shift; PYTHON_VERSION="$1"            # e.g. "3.11"
        ;;
        --s3_binary_store) shift; S3_BINARY_STORE="$1"          # e.g. "s3://aws-pytorch-cicd/pipeline/pip-wheel-builder-arm64/binary_artifacts"
        ;;
        *) echo "bad argument $1"; exit 1
        ;;
    esac
    shift
done

function arg_check() {
    if [ -z "${!1}" ]; then
        echo "argument $1 is not set."
        exit 1
    else
        echo "argument ${1}=${!1}"
    fi
}

arg_check ARCH_TYPE
arg_check PROCESSOR_TYPE
arg_check PYTORCH_VERSION
arg_check PYTHON_VERSION
arg_check S3_BINARY_STORE

# =================== download wheels from s3 =====================
s3_uri="${S3_BINARY_STORE}/"
wheels_dir="pytorch${PYTORCH_VERSION}_python${PYTHON_VERSION}_${ARCH_TYPE}_${PROCESSOR_TYPE}_wheels"
mkdir -p ${wheels_dir}

# only copy wheels not logs
aws s3 sync ${s3_uri} ${wheels_dir} --exclude "*" --include "*.whl"


# ======== determine docker image based on processor type ===========
if [ "$ARCH_TYPE" == "x86" ]; then
    if [ "$PROCESSOR_TYPE" == "cpu" ]; then
        docker_image="pytorch/manylinux-builder:cpu-main"
    else
        CUDA_VERSION=${PROCESSOR_TYPE:2:2}.${PROCESSOR_TYPE:4} # in the form M.m (e.g. "12.4")
        docker_image="pytorch/manylinux-builder:cuda${CUDA_VERSION}-main"
    fi
else
    if [ "$PROCESSOR_TYPE" == "cpu" ]; then
        docker_image="pytorch/manylinuxaarch64-builder:cpu-aarch64-main"
    else
        CUDA_VERSION=${PROCESSOR_TYPE:2:2}.${PROCESSOR_TYPE:4} # in the form M.m (e.g. "12.4")
        docker_image="pytorch/manylinuxaarch64-builder:cuda${CUDA_VERSION}-main"
    fi
fi

# ===================== launch docker container =======================
docker pull ${docker_image}
if [[ "$PROCESSOR_TYPE" == cu* ]]; then
    container_id=$(docker run -td --gpus all ${docker_image})
else
    container_id=$(docker run -td ${docker_image})
fi

# ==================== copy scripts to container ======================
docker cp install_wheels.sh ${container_id}:/
docker cp run_tests.sh ${container_id}:/

# =========== copy wheels into container and install them ==============
docker exec ${container_id} mkdir -p /wheels/
docker cp ${wheels_dir}/. ${container_id}:/wheels/
docker exec -it ${container_id} bash -i /install_wheels.sh ${ARCH_TYPE} ${PROCESSOR_TYPE} ${PYTORCH_VERSION} ${PYTHON_VERSION}

# ========== run unit tests and generate reports in container ===========
docker exec -it ${container_id} bash -i /run_tests.sh ${ARCH_TYPE} ${PROCESSOR_TYPE} ${PYTORCH_VERSION}

# =================== copy reports and upload to s3 =====================
reports_dir="pytorch${PYTORCH_VERSION}_python${PYTHON_VERSION}_${ARCH_TYPE}_${PROCESSOR_TYPE}_reports"
mkdir -p ${reports_dir}
docker cp ${container_id}:/reports/. ${reports_dir}

echo "Reports generated: "
ls -A ${reports_dir}

aws s3 cp --recursive ${reports_dir} ${s3_uri}
echo "Reports uploaded to ${s3_uri}"

docker stop ${container_id}