#!/usr/bin/env bash
# upload_cached_wheels.sh — Extract vLLM wheel via wheel-export stage and upload to S3.
#
# Usage: upload_cached_wheels.sh <cuda_version> <vllm_ref> [bucket]
#
# Requires /tmp/docker-build-base.sh from build_image.sh (contains all build args).
# S3 layout: s3://<bucket>/wheels/vllm/<cuda>/<source_hash>/vllm-*.whl
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CUDA="$1"; VLLM_REF="$2"; BUCKET="${3:-dlc-cicd-wheels}"

SOURCE_HASH=$("${SCRIPT_DIR}/vllm_source_hash.sh" "${VLLM_REF}")

if [ ! -f /tmp/docker-build-base.sh ]; then
  echo "⚠️  /tmp/docker-build-base.sh not found — skipping wheel upload"
  exit 0
fi

EXPORT_DIR=$(mktemp -d)
echo "📦 Extracting wheel from wheel-export stage..."
EXPORT_CMD=$(cat /tmp/docker-build-base.sh)
eval "${EXPORT_CMD} --target wheel-export --output type=local,dest=${EXPORT_DIR}" 2>/dev/null \
  || { echo "⚠️  Failed to extract wheel-export stage"; rm -rf "${EXPORT_DIR}"; exit 0; }

for WHL in "${EXPORT_DIR}"/wheels/*.whl; do
  [ -f "${WHL}" ] || continue
  FNAME=$(basename "${WHL}")
  S3_KEY="wheels/vllm/${CUDA}/${SOURCE_HASH}/${FNAME}"

  if aws s3 ls "s3://${BUCKET}/${S3_KEY}" &>/dev/null; then
    echo "✅ Already cached: ${S3_KEY}"
    continue
  fi

  echo "⬆️  Uploading ${FNAME} → s3://${BUCKET}/${S3_KEY}"
  aws s3 cp "${WHL}" "s3://${BUCKET}/${S3_KEY}" \
    && echo "✅ Uploaded (src:${SOURCE_HASH})" \
    || echo "⚠️  Upload failed (non-fatal)"
done

rm -rf "${EXPORT_DIR}"
