#!/usr/bin/env bash
# Download pre-built wheels from S3 into the build context.
#
# Usage:
#   bash fetch_wheels.sh --dest-dir <dir> --bucket <bucket> --cuda-version <ver> \
#     --torch-version <ver> --packages "flash-attn:2.8.3,transformer-engine-torch:2.12.0"
#
# Exit code: 0 if all wheels found, 1 if any cache miss.
# Cache key: s3://<bucket>/wheels/<pkg_underscore>/<cuda_short>/<torch_short>/<wheel_filename>
# Example: s3://dlc-cicd-wheels/wheels/flash_attn/cu130/torch213/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl

set -euo pipefail

DEST_DIR=""
BUCKET="dlc-cicd-wheels"
CUDA=""
TORCH_VERSION=""
PACKAGES=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest-dir)      DEST_DIR="$2"; shift 2 ;;
    --bucket)        BUCKET="$2"; shift 2 ;;
    --cuda-version)  CUDA="$2"; shift 2 ;;
    --torch-version) TORCH_VERSION="$2"; shift 2 ;;
    --packages)      PACKAGES="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

[[ -n "$DEST_DIR" ]]      || { echo "ERROR: --dest-dir is required" >&2; exit 1; }
[[ -n "$CUDA" ]]          || { echo "ERROR: --cuda-version is required" >&2; exit 1; }
[[ -n "$TORCH_VERSION" ]] || { echo "ERROR: --torch-version is required" >&2; exit 1; }

if [[ -z "$BUCKET" ]]; then
  echo "No wheel cache bucket configured — skipping"
  exit 1
fi

# Derive short CUDA string: 13.0.2 → cu130
CUDA_MAJOR=$(echo "$CUDA" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA" | cut -d. -f2)
CUDA_SHORT="cu${CUDA_MAJOR}${CUDA_MINOR}"

# Derive short torch string: 2.13.0 → torch213
TORCH_MAJOR=$(echo "$TORCH_VERSION" | cut -d. -f1)
TORCH_MINOR=$(echo "$TORCH_VERSION" | cut -d. -f2)
TORCH_SHORT="torch${TORCH_MAJOR}${TORCH_MINOR}"

mkdir -p "${DEST_DIR}"

ALL_HIT=true
IFS=',' read -ra SPECS <<< "$PACKAGES"
for spec in "${SPECS[@]}"; do
  [[ -z "$spec" ]] && continue
  PKG="${spec%%:*}"
  VER="${spec##*:}"
  PKG_UNDER="${PKG//-/_}"
  PREFIX="wheels/${PKG_UNDER}/${CUDA_SHORT}/${TORCH_SHORT}/"

  echo "Looking for ${PKG}==${VER} (${CUDA_SHORT}/${TORCH_SHORT}) in s3://${BUCKET}/${PREFIX} ..."
  aws s3 cp "s3://${BUCKET}/${PREFIX}" "${DEST_DIR}/" --recursive --exclude "*" --include "${PKG_UNDER}-${VER}*.whl" 2>/dev/null || true
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
