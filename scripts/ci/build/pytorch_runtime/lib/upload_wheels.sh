#!/usr/bin/env bash
# Extract built wheels from Docker wheel-export stage and upload to S3.
#
# Usage:
#   bash upload_wheels.sh --bucket <bucket> --cuda-version <ver> --image-uri <uri> \
#     --dockerfile <path> --packages "flash-attn:2.8.3,transformer-engine-torch:2.12.0"

set -euo pipefail

BUCKET="dlc-cicd-wheels"
CUDA=""
IMAGE=""
DOCKERFILE=""
PACKAGES=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bucket)         BUCKET="$2"; shift 2 ;;
    --cuda-version)   CUDA="$2"; shift 2 ;;
    --image-uri)      IMAGE="$2"; shift 2 ;;
    --dockerfile)     DOCKERFILE="$2"; shift 2 ;;
    --packages)       PACKAGES="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

[[ -n "$BUCKET" ]]     || { echo "ERROR: --bucket is required" >&2; exit 1; }
[[ -n "$CUDA" ]]       || { echo "ERROR: --cuda-version is required" >&2; exit 1; }
[[ -n "$IMAGE" ]]      || { echo "ERROR: --image-uri is required" >&2; exit 1; }
[[ -n "$DOCKERFILE" ]] || { echo "ERROR: --dockerfile is required" >&2; exit 1; }

# Derive short CUDA string: 13.0.2 → cu130
CUDA_MAJOR=$(echo "$CUDA" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA" | cut -d. -f2)
CUDA_SHORT="cu${CUDA_MAJOR}${CUDA_MINOR}"

EXPORT_DIR=$(mktemp -d)

# Forward the same --build-args the primary build passed, so BuildKit's
# cache key matches and the wheel-export stage hits cached layers instead
# of recompiling flash-attn + TE (which OOMs the runner — exit 137).
# EXTRA_BUILD_ARGS is the space-separated list of arg names written by
# .github/actions/build-image/resolve_build_args.py.
BUILD_ARG_FLAGS=()
if [[ -n "${EXTRA_BUILD_ARGS:-}" ]]; then
  for arg_name in $EXTRA_BUILD_ARGS; do
    if [[ -n "${!arg_name:-}" ]]; then
      BUILD_ARG_FLAGS+=(--build-arg "${arg_name}=${!arg_name}")
    fi
  done
fi

docker buildx build "${BUILD_ARG_FLAGS[@]}" --progress=plain --target wheel-export --output "type=local,dest=${EXPORT_DIR}" \
  -f "${DOCKERFILE}" . 2>/dev/null || {
  echo "wheel-export stage not available — extracting from runtime image"
  CID=$(docker create "${IMAGE}" /bin/true)
  docker cp "${CID}:/tmp/built_wheels/" "${EXPORT_DIR}/wheels/" 2>/dev/null || true
  docker rm "${CID}" &>/dev/null || true
}

IFS=',' read -ra SPECS <<< "$PACKAGES"
for spec in "${SPECS[@]}"; do
  [[ -z "$spec" ]] && continue
  PKG="${spec%%:*}"
  PKG_UNDER="${PKG//-/_}"

  WHL=$(find "${EXPORT_DIR}" -name "${PKG_UNDER}*.whl" 2>/dev/null | head -1)
  if [[ -z "${WHL}" ]]; then
    echo "No wheel found for ${PKG}"
    continue
  fi

  FNAME=$(basename "${WHL}")
  S3_KEY="wheels/${PKG_UNDER}/${CUDA_SHORT}/${FNAME}"

  if aws s3 ls "s3://${BUCKET}/${S3_KEY}" &>/dev/null; then
    echo "Already cached: ${S3_KEY}"
    continue
  fi

  echo "Uploading ${FNAME} -> s3://${BUCKET}/${S3_KEY}"
  aws s3 cp "${WHL}" "s3://${BUCKET}/${S3_KEY}" || echo "Upload failed (non-fatal)"
done

rm -rf "${EXPORT_DIR}"
