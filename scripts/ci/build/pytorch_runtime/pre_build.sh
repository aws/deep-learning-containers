#!/usr/bin/env bash
# Pre-build hook for PyTorch.
# Fetches cached wheels from S3 to avoid recompilation.
#
# Usage:
#   bash .github/scripts/build/pytorch/pre_build.sh --config-file <path>
#
# Inputs:
#   --config-file      - config file path
#   WHEELS_BUCKET - S3 bucket for wheel cache (env var, from workflow vars)
#
# Outputs (env vars written to $GITHUB_ENV):
#   WHEEL_CACHE_HIT    - "true" if all wheels found, "false" otherwise
#
# Side effects:
#   Places wheels in docker/pytorch/{version}/wheels/ if cache hit

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-file) CONFIG_FILE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

[[ -n "$CONFIG_FILE" ]] || { echo "ERROR: --config-file is required" >&2; exit 1; }
[[ -f "$CONFIG_FILE" ]] || { echo "ERROR: Config file not found: $CONFIG_FILE" >&2; exit 1; }

CUDA_VERSION=$(yq '.build.cuda_version' "$CONFIG_FILE")
FLASH_ATTN_VERSION=$(yq '.build.flash_attn_version // ""' "$CONFIG_FILE")
TRANSFORMER_ENGINE_VERSION=$(yq '.build.transformer_engine_version // ""' "$CONFIG_FILE")
FRAMEWORK_VERSION=$(yq '.metadata.framework_version' "$CONFIG_FILE")
PYTORCH_SHORT=$(echo "$FRAMEWORK_VERSION" | cut -d. -f1,2)

# Pass short version as build-arg for Dockerfile COPY paths
echo "FRAMEWORK_SHORT_VERSION=${PYTORCH_SHORT}" >> "${GITHUB_ENV:-/dev/null}"

DEST="docker/pytorch/${PYTORCH_SHORT}/wheels"
mkdir -p "$DEST"

PACKAGES=()
[[ -n "$FLASH_ATTN_VERSION" ]] && PACKAGES+=("flash-attn:${FLASH_ATTN_VERSION}")
[[ -n "$TRANSFORMER_ENGINE_VERSION" ]] && PACKAGES+=("transformer-engine-torch:${TRANSFORMER_ENGINE_VERSION}")

PACKAGES_STR=$(IFS=','; echo "${PACKAGES[*]}")
echo "===DIAG-PRE-BUILD=== config-file: $CONFIG_FILE"
echo "===DIAG-PRE-BUILD=== FRAMEWORK_VERSION=$FRAMEWORK_VERSION"
echo "===DIAG-PRE-BUILD=== CUDA_VERSION=$CUDA_VERSION"
echo "===DIAG-PRE-BUILD=== FLASH_ATTN_VERSION=$FLASH_ATTN_VERSION"
echo "===DIAG-PRE-BUILD=== TRANSFORMER_ENGINE_VERSION=$TRANSFORMER_ENGINE_VERSION"
echo "===DIAG-PRE-BUILD=== target DEST: $DEST"
echo "===DIAG-PRE-BUILD=== packages string: $PACKAGES_STR"
echo "Fetching cached wheels: ${PACKAGES_STR:-none}"
if bash "$SCRIPT_DIR/lib/fetch_wheels.sh" --dest-dir "$DEST" --bucket "${WHEELS_BUCKET:-dlc-cicd-wheels}" \
    --cuda-version "$CUDA_VERSION" --torch-version "$FRAMEWORK_VERSION" --packages "$PACKAGES_STR"; then
  echo "WHEEL_CACHE_HIT=true" >> "${GITHUB_ENV:-/dev/null}"
  echo "Wheel cache hit"
else
  echo "WHEEL_CACHE_HIT=false" >> "${GITHUB_ENV:-/dev/null}"
  echo "Wheel cache miss — will build from source"
fi
