#!/usr/bin/env bash
# Compute a deterministic hash of vLLM compilation inputs.
#
# Usage:
#   bash source_hash.sh --ref <vllm_ref> --version <vllm_version> [--patches-dir <dir>]
#
# Output: 12-char hex hash to stdout

set -euo pipefail

REF=""
VERSION=""
PATCHES_DIR="scripts/vllm/amzn2023/patches"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ref)         REF="$2"; shift 2 ;;
    --version)     VERSION="$2"; shift 2 ;;
    --patches-dir) PATCHES_DIR="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

[[ -n "$REF" ]]     || { echo "ERROR: --ref is required" >&2; exit 1; }
[[ -n "$VERSION" ]] || { echo "ERROR: --version is required" >&2; exit 1; }

{
  echo "ref:${REF}"
  echo "version:${VERSION}"
  find "${PATCHES_DIR}" -name '*.patch' -type f 2>/dev/null | sort | while read -r p; do
    echo "patch:$(basename "$p")"
    cat "$p"
  done
} | sha256sum | cut -c1-12
