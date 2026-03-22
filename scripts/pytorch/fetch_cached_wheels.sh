#!/usr/bin/env bash
# fetch_cached_wheels.sh — Download pre-built wheels from S3 into the build context.
#
# Usage: fetch_cached_wheels.sh <dest_dir> <bucket> <cuda> <torch> <python> <pkg:ver> [...]
#
# Cache key: s3://<bucket>/wheels/<pkg>-<ver>-cu<cuda_short>-torch<torch>-cp<py>.whl
# Missing wheels are silently skipped — the Dockerfile falls back to source build.
set -euo pipefail

DEST_DIR="$1"; BUCKET="$2"; CUDA="$3"; TORCH="$4"; PYTHON="$5"
shift 5

if [ -z "${BUCKET}" ]; then
  echo "⚠️  No wheel cache bucket configured — skipping"
  exit 0
fi

CUDA_SHORT=$(echo "${CUDA}" | cut -d. -f1,2 | tr -d '.')  # 12.9.1 → 129
PY_TAG="cp$(echo "${PYTHON}" | tr -d '.')"                 # 3.12 → cp312

mkdir -p "${DEST_DIR}"

for spec in "$@"; do
  PKG="${spec%%:*}"
  VER="${spec##*:}"
  KEY="wheels/${PKG}-${VER}-cu${CUDA_SHORT}-torch${TORCH}-${PY_TAG}.whl"
  echo "⬇️  s3://${BUCKET}/${KEY} ..."
  aws s3 cp "s3://${BUCKET}/${KEY}" "${DEST_DIR}/" 2>/dev/null \
    && echo "✅ Cache hit: ${PKG}==${VER}" \
    || echo "⚠️  Cache miss: ${PKG}==${VER}"
done
