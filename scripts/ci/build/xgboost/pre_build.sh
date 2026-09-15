#!/usr/bin/env bash
# Pre-build hook for XGBoost.
# Clones sagemaker-xgboost-container, builds a wheel, and places it
# in the Docker build context so the Dockerfile can COPY it in.
#
# Usage:
#   bash scripts/ci/build/xgboost/pre_build.sh --config-file <path>
#
# Inputs:
#   --config-file - config file path (reads build.xgboost_container_branch)
#
# Side effects:
#   Places wheel at docker/xgboost/prebuilt.whl

set -euo pipefail

CONFIG_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-file) CONFIG_FILE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

[[ -n "$CONFIG_FILE" ]] || { echo "ERROR: --config-file is required" >&2; exit 1; }
[[ -f "$CONFIG_FILE" ]] || { echo "ERROR: Config file not found: $CONFIG_FILE" >&2; exit 1; }

# Repo root (CI runs this hook from GITHUB_WORKSPACE). Anchor all build-context
# writes on it so they stay correct regardless of later `cd` calls.
REPO_ROOT="$(pwd)"

FRAMEWORK_VERSION=$(yq '.metadata.framework_version' "$CONFIG_FILE")
DOCKERFILE=$(yq '.build.dockerfile' "$CONFIG_FILE")
DOCKER_DIR=$(dirname "$DOCKERFILE")

# Fetch the DLC-maintained (patched) multi-model-server wheel into the build
# context. Required for every XGBoost version (3.0.x and 3.2.x), so this runs
# before the 3.0.x early-exit below.
echo "Fetching multi-model-server wheel..."
MMS_WHEEL_S3_URI="${MMS_WHEEL_S3_URI:-s3://dlc-cicd-wheels/wheels/mme-wheel/multi_model_server-1.1.2-py2.py3-none-any.whl}"
aws s3 cp "${MMS_WHEEL_S3_URI}" "${REPO_ROOT}/${DOCKER_DIR}/multi_model_server-1.1.2-py2.py3-none-any.whl"
echo "multi-model-server wheel ready: $(ls ${REPO_ROOT}/${DOCKER_DIR}/multi_model_server-1.1.2-py2.py3-none-any.whl)"

# 3.0.x builds the xgboost wheel inside its Dockerfile multi-stage, so no wheel pre-build is needed here.
if [[ "$FRAMEWORK_VERSION" == 3.0* ]]; then
  echo "XGBoost ${FRAMEWORK_VERSION} builds its wheel internally; skipping xgboost wheel pre-build"
  exit 0
fi

XGBOOST_CONTAINER_BRANCH=$(yq '.build.xgboost_container_branch // "master"' "$CONFIG_FILE")
XGBOOST_CONTAINER_REPO="https://github.com/aws/sagemaker-xgboost-container.git"

echo "Cloning sagemaker-xgboost-container (branch: ${XGBOOST_CONTAINER_BRANCH})..."
rm -rf /tmp/xgboost-wheel
git clone --depth 1 --branch "${XGBOOST_CONTAINER_BRANCH}" "${XGBOOST_CONTAINER_REPO}" /tmp/xgboost-wheel

echo "Building wheel..."
cd /tmp/xgboost-wheel
uv build --wheel --out-dir dist
echo "Placing wheel in build context (${DOCKER_DIR}/prebuilt.whl)..."
cp /tmp/xgboost-wheel/dist/*.whl "${REPO_ROOT}/${DOCKER_DIR}/prebuilt.whl"
cd "${REPO_ROOT}"

echo "XGBoost wheel ready: $(ls ${DOCKER_DIR}/prebuilt.whl)"

echo "Cleaning up build artifacts..."
rm -rf /tmp/xgboost-wheel
