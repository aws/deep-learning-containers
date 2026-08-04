#!/usr/bin/env bash
# Post-build hook for Lambda images. Only acts for *vllm* targets: uploads the
# freshly source-built vLLM wheel to the S3 cache and pushes the sccache compiler
# cache, so subsequent builds hit the cache instead of recompiling (~85 min).
# No-op for base/cupy/pytorch/sglang/preview targets.
#
# Usage:  bash scripts/ci/build/lambda/post_build.sh --config-file <path>
#
# Reads (env vars from pre_build via $GITHUB_ENV):
#   WHEEL_CACHE_HIT - "true" to skip upload (wheel already in cache)
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

TARGET=$(yq '.build.target' "$CONFIG_FILE")
[[ "$TARGET" == *vllm* ]] || { echo "Non-vLLM target — no post-build actions."; exit 0; }

if [[ "${WHEEL_CACHE_HIT:-}" == "true" ]]; then
  echo "vLLM wheel cache hit — skipping wheel upload and sccache push."
  exit 0
fi

BUCKET="${WHEELS_BUCKET:-dlc-cicd-wheels}"
CUDA_VERSION=$(yq '.build.cuda_version' "$CONFIG_FILE")
VLLM_REF=$(yq '.build.vllm_ref' "$CONFIG_FILE")
VLLM_VERSION=$(yq '.build.vllm_version' "$CONFIG_FILE")
ARCH_LIST=$(yq '.build.torch_cuda_arch_list // "8.0 8.6 8.9 12.0"' "$CONFIG_FILE")
USE_SCCACHE=$(yq '.build.use_sccache // "false"' "$CONFIG_FILE")

echo "Uploading Lambda vLLM wheel to cache..."
bash "$SCRIPT_DIR/lib/upload_wheels.sh" \
  --cuda-version "$CUDA_VERSION" --vllm-ref "$VLLM_REF" \
  --vllm-version "$VLLM_VERSION" --arch-list "$ARCH_LIST" --bucket "$BUCKET" || true

if [[ "$USE_SCCACHE" == "true" ]]; then
  echo "Pushing sccache to S3..."
  bash "$SCRIPT_DIR/lib/sync_sccache.sh" --action push --bucket "$BUCKET" || true
fi
