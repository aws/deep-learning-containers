#!/usr/bin/env bash
# vllm_source_hash.sh — Compute a deterministic hash of vLLM compilation inputs.
#
# Usage: vllm_source_hash.sh <vllm_ref> <vllm_version> [patches_dir]
# Output: 12-char hex hash to stdout
set -euo pipefail

VLLM_REF="$1"
VLLM_VERSION="$2"
PATCHES_DIR="${3:-scripts/vllm/amzn2023/patches}"

{
  echo "ref:${VLLM_REF}"
  echo "version:${VLLM_VERSION}"
  find "${PATCHES_DIR}" -name '*.patch' -type f 2>/dev/null | sort | while read -r p; do
    echo "patch:$(basename "$p")"
    cat "$p"
  done
} | sha256sum | cut -c1-12
