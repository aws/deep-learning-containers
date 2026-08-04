#!/usr/bin/env bash
# Download a pre-built Lambda vLLM wheel from S3 into the build context.
#
# Usage:
#   bash fetch_wheels.sh --cuda-version <ver> --vllm-ref <ref> --vllm-version <ver> --arch-list <arches> [--bucket <bucket>]
#
# Exit code: 0 if wheel found, 1 if cache miss.
# S3 layout: s3://<bucket>/wheels/lambda-vllm/<cuda>/<source_hash>/vllm-*.whl
#
# Namespaced under lambda-vllm/ (NOT the vllm_server wheels/vllm/ path): the Lambda
# wheel is built in a different environment (py3.13 base, cu130) and must not be
# cross-served with the amzn2023 wheel even when the vllm ref matches.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CUDA=""
VLLM_REF=""
VLLM_VERSION=""
ARCH_LIST=""
BUCKET="dlc-cicd-wheels"
DEST_DIR="docker/lambda/vllm/prebuilt_wheels"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda-version) CUDA="$2"; shift 2 ;;
    --vllm-ref)     VLLM_REF="$2"; shift 2 ;;
    --vllm-version) VLLM_VERSION="$2"; shift 2 ;;
    --arch-list)    ARCH_LIST="$2"; shift 2 ;;
    --bucket)       BUCKET="$2"; shift 2 ;;
    --dest-dir)     DEST_DIR="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

[[ -n "$CUDA" ]]         || { echo "ERROR: --cuda-version is required" >&2; exit 1; }
[[ -n "$VLLM_REF" ]]     || { echo "ERROR: --vllm-ref is required" >&2; exit 1; }
[[ -n "$VLLM_VERSION" ]] || { echo "ERROR: --vllm-version is required" >&2; exit 1; }
[[ -n "$ARCH_LIST" ]]    || { echo "ERROR: --arch-list is required" >&2; exit 1; }

SOURCE_HASH=$("${SCRIPT_DIR}/source_hash.sh" --ref "${VLLM_REF}" --version "${VLLM_VERSION}" --arch-list "${ARCH_LIST}")
CUDA_SHORT="cu$(echo "${CUDA}" | cut -d. -f1)$(echo "${CUDA}" | cut -d. -f2)"
PREFIX="wheels/lambda-vllm/${CUDA_SHORT}/${SOURCE_HASH}/"

mkdir -p "${DEST_DIR}"

echo "Looking for Lambda vLLM wheel (${CUDA}, src:${SOURCE_HASH}) in s3://${BUCKET}/${PREFIX} ..."
aws s3 cp "s3://${BUCKET}/${PREFIX}" "${DEST_DIR}/" \
  --recursive --exclude "*" --include "vllm-${VLLM_VERSION}*.whl" 2>/dev/null || true

if ls "${DEST_DIR}"/*.whl >/dev/null 2>&1; then
  echo "Cache hit (src:${SOURCE_HASH})"
  exit 0
else
  echo "Cache miss (src:${SOURCE_HASH})"
  exit 1
fi
