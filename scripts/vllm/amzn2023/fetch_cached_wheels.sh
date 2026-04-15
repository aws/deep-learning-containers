#!/usr/bin/env bash
# fetch_cached_wheels.sh — Check if a pre-built vLLM wheel exists in S3 and output the URI.
#
# Usage: fetch_cached_wheels.sh <cuda_version> <vllm_ref> [bucket]
# Output: S3 URI prefix if wheel exists, empty string if not.
#
# Pass the output as --build-arg VLLM_WHEEL_S3_URI=<uri> to docker build.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CUDA="$1"; VLLM_REF="$2"; BUCKET="${3:-dlc-cicd-wheels}"

SOURCE_HASH=$("${SCRIPT_DIR}/vllm_source_hash.sh" "${VLLM_REF}")
S3_URI="s3://${BUCKET}/wheels/vllm/${CUDA}/${SOURCE_HASH}/"

if aws s3 ls "${S3_URI}" 2>/dev/null | grep -q '\.whl$'; then
  echo "✅ Cache hit (src:${SOURCE_HASH})" >&2
  echo "${S3_URI}"
else
  echo "⚠️  Cache miss (src:${SOURCE_HASH}) — will build from source" >&2
  echo ""
fi
