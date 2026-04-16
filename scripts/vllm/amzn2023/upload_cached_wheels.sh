#!/usr/bin/env bash
# upload_cached_wheels.sh — Extract built vLLM wheel from the build stage and upload to S3.
#
# Usage: upload_cached_wheels.sh <cuda_version> <vllm_ref> [bucket]
#
# Must be run from the repo root after a successful docker build.
# Uses --target build to extract the wheel from the intermediate build stage.
#
# S3 layout: s3://<bucket>/wheels/vllm/<cuda>/<source_hash>/vllm-*.whl
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CUDA="$1"; VLLM_REF="$2"; BUCKET="${3:-dlc-cicd-wheels}"

SOURCE_HASH=$("${SCRIPT_DIR}/vllm_source_hash.sh" "${VLLM_REF}")

# Extract wheel from the cached build stage (no recompilation — uses Docker layer cache)
EXPORT_DIR=$(mktemp -d)
echo "📦 Extracting wheel from build stage..."
docker buildx build --progress=plain --target build \
  --output "type=local,dest=${EXPORT_DIR}" \
  -f docker/vllm/Dockerfile.amzn2023 . 2>/dev/null \
  || { echo "⚠️  Failed to extract build stage"; rm -rf "${EXPORT_DIR}"; exit 0; }

for WHL in "${EXPORT_DIR}"/workspace/vllm/dist/*.whl; do
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
