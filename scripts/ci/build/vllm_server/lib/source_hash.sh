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
# Default matches the path the Dockerfiles COPY patches from
# (docker/vllm/amzn2023/patches). The earlier "scripts/vllm/amzn2023/patches"
# default did not exist, so `find` below exited non-zero under `set -e` and
# silently killed the fetch/upload hooks that capture this hash — every build
# then missed the wheel cache and recompiled. Keep this in sync with the
# Dockerfile COPY source.
PATCHES_DIR="scripts/docker/vllm/amzn2023/patches"

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
  # Guard the dir check so a missing patches dir contributes nothing to the
  # hash rather than exiting non-zero under `set -e` (which previously killed
  # the callers that capture this hash). The hash stays stable whether the dir
  # is absent or empty.
  if [[ -d "${PATCHES_DIR}" ]]; then
    find "${PATCHES_DIR}" -name '*.patch' -type f 2>/dev/null | sort | while read -r p; do
      echo "patch:$(basename "$p")"
      cat "$p"
    done
  fi
} | sha256sum | cut -c1-12
