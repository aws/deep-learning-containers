#!/usr/bin/env bash
# Upload vLLM wheel to S3 cache.
#
# Usage:
#   bash upload_wheels.sh --cuda-version <ver> --vllm-ref <ref> --framework-version <ver> [--bucket <bucket>]
#
# Reads wheels from /tmp/vllm-wheels/wheels/ (exported by build_image.sh EXPORT_TARGETS).
# S3 layout: s3://<bucket>/wheels/vllm/<cuda>/<source_hash>/vllm-*.whl

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CUDA=""
VLLM_REF=""
FRAMEWORK_VERSION=""
BUCKET="dlc-cicd-wheels"
WHEEL_DIR="/tmp/vllm-wheels/wheels"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda-version)      CUDA="$2"; shift 2 ;;
    --vllm-ref)          VLLM_REF="$2"; shift 2 ;;
    --framework-version) FRAMEWORK_VERSION="$2"; shift 2 ;;
    --bucket)            BUCKET="$2"; shift 2 ;;
    --wheel-dir)         WHEEL_DIR="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

[[ -n "$CUDA" ]]              || { echo "ERROR: --cuda-version is required" >&2; exit 1; }
[[ -n "$VLLM_REF" ]]         || { echo "ERROR: --vllm-ref is required" >&2; exit 1; }
[[ -n "$FRAMEWORK_VERSION" ]] || { echo "ERROR: --framework-version is required" >&2; exit 1; }

SOURCE_HASH=$("${SCRIPT_DIR}/source_hash.sh" --ref "${VLLM_REF}" --version "${FRAMEWORK_VERSION}")

for WHL in "${WHEEL_DIR}"/*.whl; do
  [[ -f "${WHL}" ]] || { echo "No wheels found in ${WHEEL_DIR}"; exit 0; }
  FNAME=$(basename "${WHL}")
  S3_KEY="wheels/vllm/${CUDA}/${SOURCE_HASH}/${FNAME}"

  if aws s3 ls "s3://${BUCKET}/${S3_KEY}" &>/dev/null; then
    echo "Already cached: ${S3_KEY}"
    continue
  fi

  echo "Uploading ${FNAME} -> s3://${BUCKET}/${S3_KEY}"
  aws s3 cp "${WHL}" "s3://${BUCKET}/${S3_KEY}" || echo "Upload failed (non-fatal)"
done
