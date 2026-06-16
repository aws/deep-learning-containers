#!/usr/bin/env bash
# Pre-build hook for vLLM (and vLLM-Omni via symlink).
# Fetches cached wheel from S3 and syncs sccache.
#
# Usage:
#   bash .github/scripts/build/vllm/pre_build.sh --config-file <path>
#
# Inputs:
#   --config-file      - config file path
#   WHEEL_CACHE_BUCKET - S3 bucket (env var, default: dlc-cicd-wheels)
#
# Outputs (env vars written to $GITHUB_ENV):
#   WHEEL_CACHE_HIT    - "true" if wheel found, "false" otherwise
#   EXPORT_TARGETS     - set on cache miss for build_image.sh to export stages
#   USE_SCCACHE        - "1" if sccache enabled
#   USE_PREBUILT_WHEEL - "1" if wheel found
#
# Side effects:
#   Places wheel in docker/vllm/prebuilt_wheels/ if cache hit
#   Populates docker/vllm/sccache-cache/ from S3

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

BUCKET="${WHEEL_CACHE_BUCKET:-dlc-cicd-wheels}"

CUDA_VERSION=$(yq '.build.cuda_version' "$CONFIG_FILE")
VLLM_REF=$(yq '.build.vllm_ref' "$CONFIG_FILE")
FRAMEWORK_VERSION=$(yq '.metadata.framework_version' "$CONFIG_FILE")
USE_SCCACHE=$(yq '.build.use_sccache // "false"' "$CONFIG_FILE")

# Prepare build context directories
mkdir -p docker/vllm/prebuilt_wheels docker/vllm/sccache-cache

# Fetch cached wheel
WHEEL_HIT="false"
echo "Fetching cached vLLM wheel..."
if bash "$SCRIPT_DIR/lib/fetch_wheels.sh" --cuda-version "$CUDA_VERSION" --vllm-ref "$VLLM_REF" --framework-version "$FRAMEWORK_VERSION" --bucket "$BUCKET"; then
  WHEEL_HIT="true"
  echo "Wheel cache hit"
else
  echo "Wheel cache miss"
fi

echo "WHEEL_CACHE_HIT=${WHEEL_HIT}" >> "${GITHUB_ENV:-/dev/null}"

if [[ "$WHEEL_HIT" == "true" ]]; then
  echo "USE_PREBUILT_WHEEL=1" >> "${GITHUB_ENV:-/dev/null}"
else
  echo "USE_PREBUILT_WHEEL=0" >> "${GITHUB_ENV:-/dev/null}"
  echo "EXPORT_TARGETS=wheel-export:/tmp/vllm-wheels,sccache-export:docker/vllm/sccache-cache" >> "${GITHUB_ENV:-/dev/null}"
fi

# Sync sccache from S3
if [[ "$USE_SCCACHE" == "true" ]]; then
  echo "Syncing sccache from S3..."
  bash "$SCRIPT_DIR/lib/sync_sccache.sh" --action pull --framework vllm --bucket "$BUCKET" || echo "sccache pull failed (cold cache okay)"
fi
