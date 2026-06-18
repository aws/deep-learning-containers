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

XGBOOST_CONTAINER_BRANCH=$(yq '.build.xgboost_container_branch // "master"' "$CONFIG_FILE")
XGBOOST_CONTAINER_REPO="https://github.com/aws/sagemaker-xgboost-container.git"
DOCKERFILE=$(yq '.build.dockerfile' "$CONFIG_FILE")
DOCKER_DIR=$(dirname "$DOCKERFILE")

echo "Cloning sagemaker-xgboost-container (branch: ${XGBOOST_CONTAINER_BRANCH})..."
rm -rf /tmp/xgboost-wheel
git clone --depth 1 --branch "${XGBOOST_CONTAINER_BRANCH}" "${XGBOOST_CONTAINER_REPO}" /tmp/xgboost-wheel

echo "Building wheel..."
cd /tmp/xgboost-wheel
pip install setuptools wheel -q
python setup.py bdist_wheel --universal

echo "Placing wheel in build context (${DOCKER_DIR}/prebuilt.whl)..."
cp /tmp/xgboost-wheel/dist/*.whl "${OLDPWD}/${DOCKER_DIR}/prebuilt.whl"
cd "${OLDPWD}"

echo "XGBoost wheel ready: $(ls ${DOCKER_DIR}/prebuilt.whl)"
