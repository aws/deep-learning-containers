#!/usr/bin/env bash
# upload_cached_wheels.sh — Upload vLLM wheel to S3.
#
# Usage: upload_cached_wheels.sh <cuda_version> <vllm_ref> [bucket]
#
# Reads wheels from /tmp/vllm-wheels/wheels/ (exported by build_image.sh).
# S3 layout: s3://<bucket>/wheels/vllm/<cuda>/<source_hash>/vllm-*.whl
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CUDA="$1"; VLLM_REF="$2"; BUCKET="${3:-dlc-cicd-wheels}"
WHEEL_DIR="/tmp/vllm-wheels/wheels"

SOURCE_HASH=$("${SCRIPT_DIR}/vllm_source_hash.sh" "${VLLM_REF}")

for WHL in "${WHEEL_DIR}"/*.whl; do
  [ -f "${WHL}" ] || { echo "⚠️  No wheels found in ${WHEEL_DIR}"; exit 0; }
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
