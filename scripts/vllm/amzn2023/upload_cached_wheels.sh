#!/usr/bin/env bash
# upload_cached_wheels.sh — Extract built vLLM wheel from Docker and upload to S3.
#
# Usage: upload_cached_wheels.sh <cuda_version> <vllm_ref> <image_uri> [bucket]
#
# S3 layout: s3://<bucket>/wheels/vllm/<cuda>/<source_hash>/vllm-*.whl
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CUDA="$1"; VLLM_REF="$2"; IMAGE="$3"; BUCKET="${4:-dlc-cicd-wheels}"

SOURCE_HASH=$("${SCRIPT_DIR}/vllm_source_hash.sh" "${VLLM_REF}")

EXPORT_DIR=$(mktemp -d)

# Image may only exist in registry after --push build
docker pull "${IMAGE}" 2>/dev/null || true
CID=$(docker create "${IMAGE}" /bin/true 2>/dev/null) || { echo "⚠️  Cannot create container from ${IMAGE}"; exit 0; }
docker cp "${CID}:/workspace/vllm/dist/." "${EXPORT_DIR}/" 2>/dev/null || true
docker rm "${CID}" &>/dev/null || true

for WHL in "${EXPORT_DIR}"/*.whl; do
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
