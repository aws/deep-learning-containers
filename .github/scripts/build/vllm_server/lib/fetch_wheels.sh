#!/usr/bin/env bash
# Download pre-built vLLM wheel from S3 into the build context.
#
# Usage:
#   bash fetch_wheels.sh --cuda-version <ver> --vllm-ref <ref> --framework-version <ver> [--bucket <bucket>]
#
# Exit code: 0 if wheel found, 1 if cache miss.
# S3 layout: s3://<bucket>/wheels/vllm/<cuda>/<source_hash>/vllm-*.whl

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CUDA=""
VLLM_REF=""
FRAMEWORK_VERSION=""
BUCKET="dlc-cicd-wheels"
DEST_DIR="docker/vllm/prebuilt_wheels"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda-version)      CUDA="$2"; shift 2 ;;
    --vllm-ref)          VLLM_REF="$2"; shift 2 ;;
    --framework-version) FRAMEWORK_VERSION="$2"; shift 2 ;;
    --bucket)            BUCKET="$2"; shift 2 ;;
    --dest-dir)          DEST_DIR="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

[[ -n "$CUDA" ]]              || { echo "ERROR: --cuda-version is required" >&2; exit 1; }
[[ -n "$VLLM_REF" ]]         || { echo "ERROR: --vllm-ref is required" >&2; exit 1; }
[[ -n "$FRAMEWORK_VERSION" ]] || { echo "ERROR: --framework-version is required" >&2; exit 1; }

SOURCE_HASH=$("${SCRIPT_DIR}/source_hash.sh" --ref "${VLLM_REF}" --version "${FRAMEWORK_VERSION}")
PREFIX="wheels/vllm/${CUDA}/${SOURCE_HASH}/"

mkdir -p "${DEST_DIR}"

echo "Looking for vLLM wheel (${CUDA}, src:${SOURCE_HASH}) in s3://${BUCKET}/${PREFIX} ..."
aws s3 cp "s3://${BUCKET}/${PREFIX}" "${DEST_DIR}/" \
  --recursive --exclude "*" --include "vllm-${FRAMEWORK_VERSION}*.whl" 2>/dev/null || true

if ls "${DEST_DIR}"/*.whl >/dev/null 2>&1; then
  echo "Cache hit (src:${SOURCE_HASH})"
  exit 0
else
  echo "Cache miss (src:${SOURCE_HASH})"
  exit 1
fi
