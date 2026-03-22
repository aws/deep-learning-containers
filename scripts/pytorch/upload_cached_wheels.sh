#!/usr/bin/env bash
# upload_cached_wheels.sh — Extract built wheels from Docker builder stages and upload to S3.
#
# Usage: upload_cached_wheels.sh <bucket> <cuda> <torch> <python> <image_uri> <pkg:ver> [...]
#
# Wheels are extracted from /tmp/built_wheels/ inside the builder stages.
# The image must have been built with --target runtime (which includes builder stages in cache).
set -euo pipefail

BUCKET="$1"; CUDA="$2"; TORCH="$3"; PYTHON="$4"; IMAGE="$5"
shift 5

if [ -z "${BUCKET}" ]; then
  echo "⚠️  No wheel cache bucket configured — skipping upload"
  exit 0
fi

CUDA_SHORT=$(echo "${CUDA}" | cut -d. -f1,2 | tr -d '.')
PY_TAG="cp$(echo "${PYTHON}" | tr -d '.')"

for spec in "$@"; do
  PKG="${spec%%:*}"
  VER="${spec##*:}"
  S3_KEY="wheels/${PKG}-${VER}-cu${CUDA_SHORT}-torch${TORCH}-${PY_TAG}.whl"

  # Skip if already cached
  if aws s3 ls "s3://${BUCKET}/${S3_KEY}" &>/dev/null; then
    echo "✅ Already cached: ${PKG}==${VER}"
    continue
  fi

  # Extract wheel from the image's /tmp/built_wheels/
  PKG_UNDER="${PKG//-/_}"
  echo "📦 Extracting ${PKG}==${VER} wheel from image..."
  CID=$(docker create "${IMAGE}" /bin/true)
  docker cp "${CID}:/tmp/built_wheels/" /tmp/extract_wheels/ 2>/dev/null || true
  docker rm "${CID}" &>/dev/null || true

  WHL=$(find /tmp/extract_wheels -name "${PKG_UNDER}*.whl" 2>/dev/null | head -1)
  if [ -n "${WHL}" ]; then
    echo "⬆️  Uploading $(basename "${WHL}") → s3://${BUCKET}/${S3_KEY}"
    aws s3 cp "${WHL}" "s3://${BUCKET}/${S3_KEY}" \
      && echo "✅ Uploaded ${PKG}==${VER}" \
      || echo "⚠️  Upload failed (non-fatal)"
  else
    echo "⚠️  No wheel found for ${PKG}==${VER} in image"
  fi
  rm -rf /tmp/extract_wheels
done
