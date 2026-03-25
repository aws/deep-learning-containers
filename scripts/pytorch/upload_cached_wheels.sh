#!/usr/bin/env bash
# upload_cached_wheels.sh — Extract built wheels from Docker wheel-export stage and upload to S3.
#
# Usage: upload_cached_wheels.sh <bucket> <cuda> <torch> <python> <image_uri> <pkg:ver> [...]
set -euo pipefail

BUCKET="$1"; CUDA="$2"; IMAGE="$5"
shift 5

if [ -z "${BUCKET}" ]; then
  echo "⚠️  No wheel cache bucket configured — skipping upload"
  exit 0
fi

# Build the wheel-export stage and extract to local dir
EXPORT_DIR=$(mktemp -d)
docker buildx build --progress=plain --target wheel-export --output "type=local,dest=${EXPORT_DIR}" \
  -f docker/pytorch/Dockerfile . 2>/dev/null || {
  echo "⚠️  wheel-export stage not available — extracting from runtime image"
  CID=$(docker create "${IMAGE}" /bin/true)
  docker cp "${CID}:/tmp/built_wheels/" "${EXPORT_DIR}/wheels/" 2>/dev/null || true
  docker rm "${CID}" &>/dev/null || true
}

for spec in "$@"; do
  PKG="${spec%%:*}"
  PKG_UNDER="${PKG//-/_}"

  WHL=$(find "${EXPORT_DIR}" -name "${PKG_UNDER}*.whl" 2>/dev/null | head -1)
  if [ -z "${WHL}" ]; then
    echo "⚠️  No wheel found for ${PKG}"
    continue
  fi

  # Embed CUDA version in wheel filename: pkg-ver-cpXY-cpXY-plat.whl → pkg-ver-cuXYZ-cpXY-cpXY-plat.whl
  FNAME=$(basename "${WHL}")
  S3_KEY="wheels/${CUDA}/${PKG_UNDER}/${FNAME}"

  if aws s3 ls "s3://${BUCKET}/${S3_KEY}" &>/dev/null; then
    echo "✅ Already cached: ${S3_KEY}"
    continue
  fi

  echo "⬆️  Uploading ${FNAME} → s3://${BUCKET}/${S3_KEY}"
  aws s3 cp "${WHL}" "s3://${BUCKET}/${S3_KEY}" \
    && echo "✅ Uploaded ${PKG}" \
    || echo "⚠️  Upload failed (non-fatal)"
done

rm -rf "${EXPORT_DIR}"
