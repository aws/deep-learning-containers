#!/usr/bin/env bash
# Print the dlc-cicd-wheels S3 prefix for a TensorFlow GPU config's source-built
# artifact (training -> TF wheel, inference -> tensorflow_model_server).
#
#   wheels/tensorflow/<training|inference>/<cuXYZ>/<source-hash>/
#
# The source hash covers every input that changes the compiled output: the
# upstream version and commit, CUDA/cuDNN (and NCCL/NVSHMEM for Serving), the
# GPU target list, Python, the Bazel version, the build recipe script, and the
# Dockerfile's source-build stage (base image, build tools). Any change produces
# a new prefix, so a stale artifact is never reused.
#
# Usage: cache_prefix.sh --config-file <path>
# Also prints, on stderr, the resolved inputs for the CI log.
set -euo pipefail

CONFIG_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-file) CONFIG_FILE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done
[[ -f "$CONFIG_FILE" ]] || { echo "ERROR: config file not found: $CONFIG_FILE" >&2; exit 1; }

JOB_TYPE=$(yq -r '.metadata.job_type' "$CONFIG_FILE")
CUDA_VERSION=$(yq -r '.build.cuda_version' "$CONFIG_FILE")
CUDNN_VERSION=$(yq -r '.build.hermetic_cudnn_version' "$CONFIG_FILE")
CAPS=$(yq -r '.build.cuda_compute_capabilities' "$CONFIG_FILE")
BAZEL_VERSION=$(yq -r '.build.bazel_version' "$CONFIG_FILE")
DOCKERFILE=$(yq -r '.build.dockerfile' "$CONFIG_FILE")

case "$JOB_TYPE" in
  training)
    VERSION=$(yq -r '.build.tf_version' "$CONFIG_FILE")
    COMMIT=$(yq -r '.build.tf_commit' "$CONFIG_FILE")
    PYTHON_VERSION=$(yq -r '.build.python_version' "$CONFIG_FILE")
    EXTRA="python:${PYTHON_VERSION}"
    RECIPE="scripts/docker/tensorflow/training/build_tf_wheel.sh"
    STAGE="tf-wheel-build"
    ;;
  inference)
    VERSION=$(yq -r '.build.tf_serving_version' "$CONFIG_FILE")
    COMMIT=$(yq -r '.build.tf_serving_commit' "$CONFIG_FILE")
    NCCL_VERSION=$(yq -r '.build.hermetic_nccl_version' "$CONFIG_FILE")
    NVSHMEM_VERSION=$(yq -r '.build.hermetic_nvshmem_version' "$CONFIG_FILE")
    EXTRA="nccl:${NCCL_VERSION} nvshmem:${NVSHMEM_VERSION}"
    RECIPE="scripts/docker/tensorflow/inference/build_tf_serving.sh"
    STAGE="tfs-build"
    ;;
  *) echo "ERROR: unsupported job_type: $JOB_TYPE" >&2; exit 1 ;;
esac

for v in VERSION COMMIT CUDA_VERSION CUDNN_VERSION CAPS BAZEL_VERSION DOCKERFILE; do
  [[ -n "${!v}" && "${!v}" != "null" ]] || { echo "ERROR: missing build key for ${v} in $CONFIG_FILE" >&2; exit 1; }
done

# The source-build stage of the Dockerfile, without comments or blank lines, so
# that a comment-only edit does not force a two-hour rebuild.
STAGE_TEXT=$(awk -v stage="$STAGE" '
  $0 ~ "^FROM .* AS " stage "$" { inside = 1; print; next }
  inside && /^FROM / { exit }
  inside && $0 !~ /^[[:space:]]*(#|$)/ { print }
' "$DOCKERFILE")
[[ -n "$STAGE_TEXT" ]] || { echo "ERROR: stage ${STAGE} not found in ${DOCKERFILE}" >&2; exit 1; }

HASH=$({
  echo "version:${VERSION}"
  echo "commit:${COMMIT}"
  echo "cuda:${CUDA_VERSION}"
  echo "cudnn:${CUDNN_VERSION}"
  echo "caps:${CAPS}"
  echo "bazel:${BAZEL_VERSION}"
  echo "${EXTRA}"
  cat "${RECIPE}"
  echo "${STAGE_TEXT}"
} | sha256sum | cut -c1-12)

CUDA_SHORT="cu$(echo "${CUDA_VERSION}" | cut -d. -f1-2 | tr -d '.')"
echo "TF ${JOB_TYPE} artifact: version=${VERSION} commit=${COMMIT} cuda=${CUDA_VERSION} cudnn=${CUDNN_VERSION} caps=${CAPS} bazel=${BAZEL_VERSION} ${EXTRA} recipe=${RECIPE} stage=${STAGE} hash=${HASH}" >&2
echo "wheels/tensorflow/${JOB_TYPE}/${CUDA_SHORT}/${HASH}/"
