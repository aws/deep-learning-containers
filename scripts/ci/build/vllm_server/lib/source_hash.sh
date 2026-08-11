#!/usr/bin/env bash
# Compute a deterministic hash of vLLM compilation inputs.
#
# Usage:
#   bash source_hash.sh --ref <vllm_ref> --version <vllm_version> [--arch-list <list>] [--patches-dir <dir>]
#
# Output: 12-char hex hash to stdout

set -euo pipefail

REF=""
VERSION=""
# The CUDA architectures the wheel is compiled for. This MUST be part of the hash:
# the cached wheel's cubins are only valid for the arches it was built with, and
# nothing else in the key reflects them. Omitting it meant adding sm_103 (B300) to
# the arch list did not change the key, so the build reused a wheel compiled without
# sm_103, skipped compilation entirely, and shipped an image that could not run on
# B300 -- with a green CI and a 2-minute "cache hit" build as the only symptom.
ARCH_LIST=""
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
    --arch-list)   ARCH_LIST="$2"; shift 2 ;;
    --patches-dir) PATCHES_DIR="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

[[ -n "$REF" ]]     || { echo "ERROR: --ref is required" >&2; exit 1; }
[[ -n "$VERSION" ]] || { echo "ERROR: --version is required" >&2; exit 1; }

{
  echo "ref:${REF}"
  echo "version:${VERSION}"
  # Normalized (whitespace-collapsed, sorted) so that reordering or reformatting the
  # same set of arches does not needlessly invalidate the cache. Emitted only when
  # non-empty, so configs that do not pin an arch list keep their existing hash
  # instead of every framework's cache being invalidated by this change.
  if [[ -n "${ARCH_LIST}" ]]; then
    echo "arch:$(echo "${ARCH_LIST}" | tr ' ;' '\n\n' | grep -v '^$' | sort | tr '\n' ' ' | sed 's/ $//')"
  fi
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
