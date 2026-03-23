#!/usr/bin/env bash
# fetch_cached_wheels.sh — Download pre-built wheels from S3 into the build context.
#
# Usage: fetch_cached_wheels.sh <dest_dir> <bucket> <cuda> <torch> <python> <pkg:ver> [...]
#
# Cache key: s3://<bucket>/wheels/<pkg_under>/<original_wheel_filename>
# Missing wheels are silently skipped — the Dockerfile falls back to source build.
set -euo pipefail

DEST_DIR="$1"; BUCKET="$2"; CUDA="$3"; TORCH="$4"; PYTHON="$5"
shift 5

if [ -z "${BUCKET}" ]; then
  echo "⚠️  No wheel cache bucket configured — skipping"
  exit 0
fi

mkdir -p "${DEST_DIR}"

for spec in "$@"; do
  PKG="${spec%%:*}"
  VER="${spec##*:}"
  PKG_UNDER="${PKG//-/_}"
  PREFIX="wheels/${PKG_UNDER}/"
  echo "⬇️  Looking for ${PKG}==${VER} in s3://${BUCKET}/${PREFIX} ..."
  # Download any .whl matching the package name — preserves original filename
  aws s3 cp "s3://${BUCKET}/${PREFIX}" "${DEST_DIR}/" --recursive --exclude "*" --include "*.whl" 2>/dev/null \
    && echo "✅ Cache hit: ${PKG}==${VER}" \
    || echo "⚠️  Cache miss: ${PKG}==${VER}"
done
