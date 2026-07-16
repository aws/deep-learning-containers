#!/usr/bin/env bash
# Download pre-built wheels from S3 into the build context.
# Shared per-package cache: s3://<bucket>/wheels/<pkg_underscore>/<cuda_short>/<file>.
# Filtered by PYTHON_TAG (e.g. cp313) so we never pull another framework's cp312 wheel.

set -euo pipefail

DEST_DIR=""
BUCKET="dlc-cicd-wheels"
CUDA=""
PACKAGES=""
PYTHON_TAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest-dir)     DEST_DIR="$2"; shift 2 ;;
    --bucket)       BUCKET="$2"; shift 2 ;;
    --cuda-version) CUDA="$2"; shift 2 ;;
    --packages)     PACKAGES="$2"; shift 2 ;;
    --python-tag)   PYTHON_TAG="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

[[ -n "$DEST_DIR" ]]   || { echo "ERROR: --dest-dir is required" >&2; exit 1; }
[[ -n "$CUDA" ]]       || { echo "ERROR: --cuda-version is required" >&2; exit 1; }
[[ -n "$PYTHON_TAG" ]] || { echo "ERROR: --python-tag is required (e.g. cp313)" >&2; exit 1; }

if [[ -z "$BUCKET" ]]; then
  echo "No wheel cache bucket configured — skipping"
  exit 1
fi

# Derive short CUDA string: 13.0.2 → cu130
CUDA_MAJOR=$(echo "$CUDA" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA" | cut -d. -f2)
CUDA_SHORT="cu${CUDA_MAJOR}${CUDA_MINOR}"

mkdir -p "${DEST_DIR}"

ALL_HIT=true
IFS=',' read -ra SPECS <<< "$PACKAGES"
for spec in "${SPECS[@]}"; do
  [[ -z "$spec" ]] && continue
  PKG="${spec%%:*}"
  VER="${spec##*:}"
  PKG_UNDER="${PKG//-/_}"
  PREFIX="wheels/${PKG_UNDER}/${CUDA_SHORT}/"

  # Filter on version AND python tag so we never pull another framework's cp312 wheel.
  echo "Looking for ${PKG}==${VER} (${CUDA_SHORT}, ${PYTHON_TAG}) in s3://${BUCKET}/${PREFIX} ..."
  aws s3 cp "s3://${BUCKET}/${PREFIX}" "${DEST_DIR}/" --recursive --exclude "*/*" --include "${PKG_UNDER}-${VER}-${PYTHON_TAG}-*.whl" 2>/dev/null || true
  if ls "${DEST_DIR}"/${PKG_UNDER}*.whl >/dev/null 2>&1; then
    echo "Cache hit: ${PKG}==${VER}"
  else
    echo "Cache miss: ${PKG}==${VER}"
    ALL_HIT=false
  fi
done

if [[ "$ALL_HIT" == "true" ]]; then
  exit 0
else
  exit 1
fi
