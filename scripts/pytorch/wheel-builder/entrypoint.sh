#!/bin/bash
set -xeo pipefail

while test $# -gt 0
do
    case "$1" in
        --arch_type) shift; ARCH_TYPE="$1"                      # e.g. "x86" or "arm64"
        ;;
        --processor_type) shift; PROCESSOR_TYPE="$1"            # e.g. "cpu" or "cu124"
        ;;
        --pytorch_version) shift; PYTORCH_VERSION="$1"          # e.g. "2.6.0"
        ;;
        --python_version) shift; PYTHON_VERSION="$1"            # e.g. "3.12"
        ;;
        --torchvision_version) shift; TORCHVISION_VERSION="$1"  # e.g. "0.21.0"
        ;;
        --torchaudio_version) shift; TORCHAUDIO_VERSION="$1"    # e.g. "2.6.0"
        ;;
        --torchtext_version) shift; TORCHTEXT_VERSION="$1"      # e.g. "0.18.0"
        ;;
        --torchdata_version) shift; TORCHDATA_VERSION="$1"      # e.g. "0.10.1"
        ;;
        --s3_binary_store) shift; S3_BINARY_STORE="$1"          # e.g. "s3://asimov-codepipeline-infra-s3-source/pipeline/pip-wheel-builder-arm64/binary_artifacts"
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
arg_check TORCHVISION_VERSION
arg_check TORCHAUDIO_VERSION
arg_check TORCHTEXT_VERSION
arg_check TORCHDATA_VERSION
arg_check S3_BINARY_STORE

# ================= select container =================
if [ "$ARCH_TYPE" == "x86" ]; then
    if [ "$PROCESSOR_TYPE" == "cpu" ]; then
        DOCKER_IMAGE="pytorch/manylinux-builder:cpu-main"
    else
        CUDA_VERSION=${PROCESSOR_TYPE:2:2}.${PROCESSOR_TYPE:4} # in the form M.m (e.g. "12.4")
        DOCKER_IMAGE="pytorch/manylinux-builder:cuda${CUDA_VERSION}-main"
    fi
else # arm64
    if [ "$PROCESSOR_TYPE" == "cpu" ]; then
        DOCKER_IMAGE="pytorch/manylinuxaarch64-builder:cpu-aarch64-main"
    else
        CUDA_VERSION=${PROCESSOR_TYPE:2:2}.${PROCESSOR_TYPE:4} # in the form M.m (e.g. "12.4")
        DOCKER_IMAGE="pytorch/manylinuxaarch64-builder:cuda${CUDA_VERSION}-main"
    fi
fi

# =================== setup =====================
export WORKDIR=$(pwd)
export BUILDDIR="${WORKDIR}/build/manywheel-linux-${ARCH_TYPE}-${PROCESSOR_TYPE}-py${PYTHON_VERSION}-$(date +'%Y%m%d-%H%M%S')"
mkdir -p ${BUILDDIR}/artifacts                   # store pip wheels that are built
mkdir -p ${BUILDDIR}/scripts                     # store scripts that will be used in the container
cp -r ${WORKDIR}/scripts ${BUILDDIR}/            # copy scripts into the directory that'll be used in the container

# ================= build torch =================
CONTAINER_ID=$(docker run \
    --tty \
    --detach \
    -v "${BUILDDIR}/artifacts:/artifacts" \
    -v "${BUILDDIR}/scripts:/scripts" \
    "${DOCKER_IMAGE}")

docker exec -t "${CONTAINER_ID}" bash -c \
    "bash /scripts/build_torch.sh \
        ${PROCESSOR_TYPE} \
        ${PYTORCH_VERSION} \
        ${PYTHON_VERSION} \
        ${ARCH_TYPE}"

# ================= build tools =================
if [ "$ARCH_TYPE" == "x86" ]; then
    # for x86, we need to use a new container for building tools
    # becuase the devtools are removed by x86 torch build process
    docker kill ${CONTAINER_ID}
    CONTAINER_ID=$(docker run \
        --tty \
        --detach \
        -v "${BUILDDIR}/artifacts:/artifacts" \
        -v "${BUILDDIR}/scripts:/scripts" \
        "${DOCKER_IMAGE}")
fi

docker exec -t "${CONTAINER_ID}" bash -c \
    "bash /scripts/build_tools.sh \
        ${PROCESSOR_TYPE} \
        ${TORCHVISION_VERSION} \
        ${TORCHAUDIO_VERSION} \
        ${TORCHTEXT_VERSION} \
        ${TORCHDATA_VERSION} \
        ${ARCH_TYPE} \
        ${PYTHON_VERSION}"

# ================= upload wheels to s3 =================
S3_URI="${S3_BINARY_STORE}/pytorch${PYTORCH_VERSION}/${PROCESSOR_TYPE}"
aws s3 cp --recursive ${BUILDDIR}/artifacts ${S3_URI}

docker kill ${CONTAINER_ID}
