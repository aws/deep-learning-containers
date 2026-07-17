#!/usr/bin/env bash
# Post-build hook for PyTorch.
# Uploads built wheels to S3 cache for future builds.
#
# Usage:
#   bash .github/scripts/build/pytorch/post_build.sh --config-file <path>
#
# Inputs:
#   --config-file      - config file path
#   WHEEL_CACHE_HIT    - "true" to skip upload (env var from pre_build)
#   WHEELS_BUCKET - S3 bucket (env var)
#   CI_IMAGE_URI       - built image URI (env var)
#
# Outputs: none (uploads to S3)

set -euo pipefail

if [[ "${WHEEL_CACHE_HIT:-}" == "true" ]]; then
  echo "Wheel cache hit — skipping upload"
  exit 0
fi

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
FRAMEWORK_VERSION=$(yq '.metadata.framework_version' "$CONFIG_FILE")
FLASH_ATTN_VERSION=$(yq '.build.flash_attn_version // ""' "$CONFIG_FILE")
TRANSFORMER_ENGINE_VERSION=$(yq '.build.transformer_engine_version // ""' "$CONFIG_FILE")
DOCKERFILE=$(yq '.build.dockerfile' "$CONFIG_FILE")

PACKAGES=()
[[ -n "$FLASH_ATTN_VERSION" ]] && PACKAGES+=("flash-attn:${FLASH_ATTN_VERSION}")
[[ -n "$TRANSFORMER_ENGINE_VERSION" ]] && PACKAGES+=("transformer-engine-torch:${TRANSFORMER_ENGINE_VERSION}")

PACKAGES_STR=$(IFS=','; echo "${PACKAGES[*]}")
echo "Uploading wheels to cache: ${PACKAGES_STR:-none}"
bash "$SCRIPT_DIR/lib/upload_wheels.sh" --bucket "${WHEELS_BUCKET:-dlc-cicd-wheels}" \
  --cuda-version "$CUDA_VERSION" --torch-version "$FRAMEWORK_VERSION" \
  --image-uri "${CI_IMAGE_URI}" --dockerfile "$DOCKERFILE" --packages "$PACKAGES_STR" || true
