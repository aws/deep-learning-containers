#!/usr/bin/env bash
# fetch_cached_wheels.sh — Download pre-built vLLM wheel from S3 into the build context.
#
# Usage: fetch_cached_wheels.sh <cuda_version> <vllm_ref> [bucket]
#
# Run BEFORE docker build. Wheels go into docker/vllm/prebuilt_wheels/
# which the Dockerfile COPYs. If no wheel found, the dir stays empty
# and the Dockerfile falls back to building from source.
#
# S3 layout: s3://<bucket>/wheels/vllm/<cuda>/<source_hash>/vllm-*.whl
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CUDA="$1"; VLLM_REF="$2"; BUCKET="${3:-dlc-cicd-wheels}"
DEST_DIR="docker/vllm/prebuilt_wheels"

SOURCE_HASH=$("${SCRIPT_DIR}/vllm_source_hash.sh" "${VLLM_REF}")
PREFIX="wheels/vllm/${CUDA}/${SOURCE_HASH}/"

mkdir -p "${DEST_DIR}"

echo "⬇️  Looking for vLLM wheel (${CUDA}, src:${SOURCE_HASH}) in s3://${BUCKET}/${PREFIX} ..."
aws s3 cp "s3://${BUCKET}/${PREFIX}" "${DEST_DIR}/" \
  --recursive --exclude "*" --include "*.whl" 2>/dev/null \
  && echo "✅ Cache hit (src:${SOURCE_HASH})" \
  || echo "⚠️  Cache miss (src:${SOURCE_HASH}) — will build from source"
