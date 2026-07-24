#!/usr/bin/env bash
# Pre-build hook for Scikit-learn.
# Clones sagemaker-scikit-learn-container, builds a wheel, and places it
# in the Docker build context so the Dockerfile can COPY it in.
#
# Usage:
#   bash scripts/ci/build/sklearn/pre_build.sh --config-file <path>
#
# Inputs:
#   --config-file - config file path (reads build.sklearn_container_branch)
#
# Side effects:
#   Places wheel at ${DOCKER_DIR}/prebuilt.whl (e.g. docker/sklearn/1.4-2-py312/prebuilt.whl)

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

# Capture the repo root explicitly — using $OLDPWD is fragile under `set -u`
# (unset if no prior cd) and can misfire in non-interactive shells.
REPO_ROOT=$(pwd)

DOCKERFILE=$(yq '.build.dockerfile' "$CONFIG_FILE")
DOCKER_DIR=$(dirname "$DOCKERFILE")
SKLEARN_CONTAINER_BRANCH=$(yq '.build.sklearn_container_branch // "master"' "$CONFIG_FILE")
SKLEARN_CONTAINER_REPO="https://github.com/aws/sagemaker-scikit-learn-container.git"

echo "Cloning sagemaker-scikit-learn-container (branch: ${SKLEARN_CONTAINER_BRANCH})..."
rm -rf /tmp/sklearn-wheel
git clone --depth 1 --branch "${SKLEARN_CONTAINER_BRANCH}" "${SKLEARN_CONTAINER_REPO}" /tmp/sklearn-wheel

echo "Building wheel..."
cd /tmp/sklearn-wheel
uv build --wheel --out-dir dist

echo "Placing wheel in build context (${DOCKER_DIR}/prebuilt.whl)..."
cp /tmp/sklearn-wheel/dist/*.whl "${REPO_ROOT}/${DOCKER_DIR}/prebuilt.whl"

echo "Scikit-learn wheel ready: $(ls ${REPO_ROOT}/${DOCKER_DIR}/prebuilt.whl)"

echo "Cleaning up build artifacts..."
rm -rf /tmp/sklearn-wheel
