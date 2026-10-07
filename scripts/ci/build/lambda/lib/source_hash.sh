#!/usr/bin/env bash
# Compute a deterministic hash of the Lambda vLLM wheel's compilation inputs.
#
# Usage:
#   bash source_hash.sh --ref <vllm_ref> --version <vllm_version> --arch-list <"8.0 8.6 8.9 12.0">
#
# Output: 12-char hex hash to stdout.
#
# The Lambda vLLM wheel is an abi3 wheel with CUDA kernels compiled for a specific
# set of GPU arches, so the arch list is part of the cache key (a wheel built for
# different arches is NOT interchangeable). vLLM ref + version pin the source.
set -euo pipefail

REF=""
VERSION=""
ARCH_LIST=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ref)       REF="$2"; shift 2 ;;
    --version)   VERSION="$2"; shift 2 ;;
    --arch-list) ARCH_LIST="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

[[ -n "$REF" ]]       || { echo "ERROR: --ref is required" >&2; exit 1; }
[[ -n "$VERSION" ]]   || { echo "ERROR: --version is required" >&2; exit 1; }
[[ -n "$ARCH_LIST" ]] || { echo "ERROR: --arch-list is required" >&2; exit 1; }

{
  echo "ref:${REF}"
  echo "version:${VERSION}"
  echo "arch:${ARCH_LIST}"
} | sha256sum | cut -c1-12
