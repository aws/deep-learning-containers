#!/usr/bin/env bash
# fetch_cached_wheels.sh — Download pre-built vLLM wheel from S3 into the build context.
#
# Usage: fetch_cached_wheels.sh <cuda_version> <vllm_ref> <vllm_version> [bucket]
# Exit code: 0 always. Prints "cache-hit=true" or "cache-hit=false" for CI.
#
# S3 layout: s3://<bucket>/wheels/vllm/<cuda>/<source_hash>/vllm-*.whl
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CUDA="$1"; VLLM_REF="$2"; VLLM_VERSION="$3"; BUCKET="${4:-dlc-cicd-wheels}"
DEST_DIR="docker/vllm/prebuilt_wheels"

SOURCE_HASH=$("${SCRIPT_DIR}/vllm_source_hash.sh" "${VLLM_REF}" "${VLLM_VERSION}")
PREFIX="wheels/vllm/${CUDA}/${SOURCE_HASH}/"

mkdir -p "${DEST_DIR}"

echo "⬇️  Looking for vLLM wheel (${CUDA}, src:${SOURCE_HASH}) in s3://${BUCKET}/${PREFIX} ..."
aws s3 cp "s3://${BUCKET}/${PREFIX}" "${DEST_DIR}/" \
  --recursive --exclude "*" --include "vllm-${VLLM_VERSION}*.whl" 2>/dev/null || true

if ls "${DEST_DIR}"/*.whl >/dev/null 2>&1; then
  echo "✅ Cache hit (src:${SOURCE_HASH})"
  echo "cache-hit=true"
else
  echo "⚠️  Cache miss (src:${SOURCE_HASH}) — will build from source"
  echo "cache-hit=false"
fi
