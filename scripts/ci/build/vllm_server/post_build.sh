#!/usr/bin/env bash
# Post-build hook for vLLM (and vLLM-Omni via symlink).
# Uploads built wheel to S3 and syncs sccache.
#
# Usage:
#   bash .github/scripts/build/vllm/post_build.sh --config-file <path>
#
# Inputs:
#   --config-file      - config file path
#   WHEEL_CACHE_HIT    - "true" to skip upload (env var from pre_build)
#   WHEELS_BUCKET - S3 bucket (env var, default: dlc-cicd-wheels)
#
# Outputs: none (uploads to S3)

set -euo pipefail

if [[ "${WHEEL_CACHE_HIT:-}" == "true" ]]; then
  echo "Wheel cache hit — skipping upload and sccache push"
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

BUCKET="${WHEELS_BUCKET:-dlc-cicd-wheels}"

CUDA_VERSION=$(yq '.build.cuda_version' "$CONFIG_FILE")
VLLM_REF=$(yq '.build.vllm_ref' "$CONFIG_FILE")
FRAMEWORK_VERSION=$(yq '.metadata.framework_version' "$CONFIG_FILE")
USE_SCCACHE=$(yq '.build.use_sccache // "false"' "$CONFIG_FILE")

# Upload wheel
echo "Uploading vLLM wheel to cache..."
bash "$SCRIPT_DIR/lib/upload_wheels.sh" --cuda-version "$CUDA_VERSION" --vllm-ref "$VLLM_REF" --framework-version "$FRAMEWORK_VERSION" --bucket "$BUCKET" || true

# Push sccache
if [[ "$USE_SCCACHE" == "true" ]]; then
  echo "Syncing sccache to S3..."
  bash "$SCRIPT_DIR/lib/sync_sccache.sh" --action push --framework vllm --bucket "$BUCKET" || true
fi
