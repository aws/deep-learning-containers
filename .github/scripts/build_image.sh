#!/bin/bash
set -euo pipefail

# Build Docker Image Script
# This script builds and pushes Docker images for different frameworks and targets

# Required parameters
FRAMEWORK="${FRAMEWORK:?FRAMEWORK is required}"
TARGET="${TARGET:?TARGET is required}"
BASE_IMAGE="${BASE_IMAGE:?BASE_IMAGE is required}"
FRAMEWORK_VERSION="${FRAMEWORK_VERSION:?FRAMEWORK_VERSION is required}"
CONTAINER_TYPE="${CONTAINER_TYPE:?CONTAINER_TYPE is required}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:?AWS_ACCOUNT_ID is required}"
AWS_REGION="${AWS_REGION:?AWS_REGION is required}"
TAG_PR="${TAG_PR:?TAG_PR is required}"
DOCKERFILE_PATH="${DOCKERFILE_PATH:?DOCKERFILE_PATH is required}"

# Optional parameters with defaults
ARCH_TYPE="${ARCH_TYPE:-x86}"
DEVICE_TYPE="${DEVICE_TYPE:-gpu}"
CUDA_VERSION="${CUDA_VERSION:-}"
PYTHON_VERSION="${PYTHON_VERSION:-}"
OS_VERSION="${OS_VERSION:-}"
CONTRIBUTOR="${CONTRIBUTOR:-None}"
CUSTOMER_TYPE="${CUSTOMER_TYPE:-}"
INFERENCE_TOOLKIT_VERSION="${INFERENCE_TOOLKIT_VERSION:-}"
TORCHSERVE_VERSION="${TORCHSERVE_VERSION:-}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-}"
RUNTIME_BASE="${RUNTIME_BASE:-}"

# Resolve image URI
CI_IMAGE_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/ci:${TAG_PR}"
echo "Image URI to build: ${CI_IMAGE_URI}"

# Process label values
LABEL_FRAMEWORK=$(echo "${FRAMEWORK}" | tr '_' '-')
LABEL_FRAMEWORK_VERSION=$(echo "${FRAMEWORK_VERSION}" | tr '.' '-')
LABEL_ARCH="${ARCH_TYPE}"
LABEL_PYTHON_VERSION=$(echo "${PYTHON_VERSION}" | tr '.' '-')
LABEL_OS_VERSION=$(echo "${OS_VERSION}" | tr '.' '-')
LABEL_CONTRIBUTOR="${CONTRIBUTOR}"

# Build transformers label if version provided
LABEL_TRANSFORMERS=""
if [[ -n "${TRANSFORMERS_VERSION}" ]]; then
  LABEL_TRANSFORMERS=$(echo "${TRANSFORMERS_VERSION}" | tr '.' '-')
fi

# Build inference toolkit label if both versions provided
LABEL_INFERENCE_TOOLKIT=""
if [[ -n "${INFERENCE_TOOLKIT_VERSION}" && -n "${TORCHSERVE_VERSION}" ]]; then
  TOOLKIT_VER=$(echo "${INFERENCE_TOOLKIT_VERSION}" | tr '.' '-')
  TORCHSERVE_VER=$(echo "${TORCHSERVE_VERSION}" | tr '.' '-')
  LABEL_INFERENCE_TOOLKIT="${TOOLKIT_VER}.torchserve.${TORCHSERVE_VER}"
fi

# Construct device type label
LABEL_DEVICE_TYPE="${DEVICE_TYPE}"
if [[ "${DEVICE_TYPE}" == "gpu" && -n "${CUDA_VERSION}" ]]; then
  LABEL_DEVICE_TYPE="${DEVICE_TYPE}.${CUDA_VERSION}"
fi

# Build base command
BUILD_CMD="docker buildx build --progress plain \
  --build-arg CACHE_REFRESH=\"$(date +"%Y-%m-%d")\" \
  --build-arg BASE_IMAGE=\"${BASE_IMAGE}\" \
  --build-arg CONTAINER_TYPE=\"${CONTAINER_TYPE}\" \
  --build-arg FRAMEWORK=\"${FRAMEWORK}\" \
  --build-arg FRAMEWORK_VERSION=\"${FRAMEWORK_VERSION}\""

# Use pre-built runtime base if available (skips compile stages)
if [[ -n "${RUNTIME_BASE}" ]]; then
  echo "Using pre-built runtime base: ${RUNTIME_BASE}"
  BUILD_CMD="${BUILD_CMD} \
  --build-arg RUNTIME_BASE=\"${RUNTIME_BASE}\""
fi

# Enable sccache for compilation caching (BuildKit cache mount persists on same runner)
if [[ -n "${USE_SCCACHE:-}" ]]; then
  echo "Enabling sccache with BuildKit cache mount"
  BUILD_CMD="${BUILD_CMD} \
  --build-arg USE_SCCACHE=\"${USE_SCCACHE}\""
fi

# Add SageMaker labels if customer-type is 'sagemaker'
if [[ "${CUSTOMER_TYPE}" == "sagemaker" ]]; then
  BUILD_CMD="${BUILD_CMD} \
  --label \"com.amazonaws.ml.engines.sagemaker.dlc.arch.${LABEL_ARCH}=true\" \
  --label \"com.amazonaws.ml.engines.sagemaker.dlc.device.${LABEL_DEVICE_TYPE}=true\" \
  --label \"com.amazonaws.ml.engines.sagemaker.dlc.framework.${LABEL_FRAMEWORK}.${LABEL_FRAMEWORK_VERSION}=true\" \
  --label \"com.amazonaws.ml.engines.sagemaker.dlc.job.${CONTAINER_TYPE}=true\""

  # Add OS version label if provided
  if [[ -n "${OS_VERSION}" ]]; then
    BUILD_CMD="${BUILD_CMD} \
  --label \"com.amazonaws.ml.engines.sagemaker.dlc.os.${LABEL_OS_VERSION}=true\""
  fi

  # Add Python version label if provided
  if [[ -n "${PYTHON_VERSION}" ]]; then
    BUILD_CMD="${BUILD_CMD} \
  --label \"com.amazonaws.ml.engines.sagemaker.dlc.python.${LABEL_PYTHON_VERSION}=true\""
  fi

  # Add contributor label if provided
  if [[ -n "${LABEL_CONTRIBUTOR}" ]]; then
    BUILD_CMD="${BUILD_CMD} \
  --label \"com.amazonaws.ml.engines.sagemaker.dlc.contributor.${LABEL_CONTRIBUTOR}=true\""
  fi

  # Add transformers library label if provided
  if [[ -n "${LABEL_TRANSFORMERS}" ]]; then
    BUILD_CMD="${BUILD_CMD} \
  --label \"com.amazonaws.ml.engines.sagemaker.dlc.lib.transformers.${LABEL_TRANSFORMERS}=true\""
  fi

  # Add inference toolkit label if provided
  if [[ -n "${LABEL_INFERENCE_TOOLKIT}" ]]; then
    BUILD_CMD="${BUILD_CMD} \
  --label \"com.amazonaws.ml.engines.sagemaker.dlc.inference-toolkit.${LABEL_INFERENCE_TOOLKIT}=true\""
  fi
fi

# Ensure default docker builder is used
docker buildx use default 2>/dev/null || true

# Step 1: Extract wheel from build stage (if wheel-export target exists)
# This caches all layers including the build stage for reuse in step 2
WHEEL_DIR="/tmp/${FRAMEWORK}-wheels"
echo "Step 1: Extracting wheel from build stage..."
eval ${BUILD_CMD} \
  --target wheel-export \
  --output "type=local,dest=${WHEEL_DIR}" \
  -f ${DOCKERFILE_PATH} . 2>/dev/null \
  && echo "Wheels extracted: $(ls ${WHEEL_DIR}/wheels/*.whl 2>/dev/null)" \
  || echo "No wheel-export target — skipping wheel extraction"

# Step 2: Build and push final image (build stage is cached from step 1)
echo "Step 2: Building and pushing final image..."
eval ${BUILD_CMD} \
  --cache-to=type=inline \
  --cache-from=type=registry,ref=${CI_IMAGE_URI} \
  --tag ${CI_IMAGE_URI} \
  --push \
  --target ${TARGET} \
  -f ${DOCKERFILE_PATH} .

# Clean up local image (may not exist when using remote buildkitd)
docker rmi ${CI_IMAGE_URI} 2>/dev/null || true

echo "Build completed successfully!"
echo "CI_IMAGE_URI=${CI_IMAGE_URI}"
