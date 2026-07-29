#!/usr/bin/env bash
# ray pre-build: fetch cached flash-attn/TE wheels; no-op for Ray Serve (shares framework=ray).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-file) CONFIG_FILE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

[[ -n "$CONFIG_FILE" ]] || { echo "ERROR: --config-file is required" >&2; exit 1; }
[[ -f "$CONFIG_FILE" ]] || { echo "ERROR: Config file not found: $CONFIG_FILE" >&2; exit 1; }

FLASH_ATTN_VERSION=$(yq '.build.flash_attn_version // ""' "$CONFIG_FILE")
TRANSFORMER_ENGINE_VERSION=$(yq '.build.transformer_engine_version // ""' "$CONFIG_FILE")

# Ray Serve (no compiled wheels) — nothing to do.
if [[ -z "$FLASH_ATTN_VERSION" && -z "$TRANSFORMER_ENGINE_VERSION" ]]; then
  echo "No flash_attn/transformer_engine version in config — no wheel cache needed (Ray Serve)."
  exit 0
fi

CUDA_VERSION=$(yq '.build.cuda_version' "$CONFIG_FILE")
# Derive the CPython tag from python_version (3.13.12 → cp313) to scope the shared cache.
PYTHON_VERSION=$(yq '.build.python_version' "$CONFIG_FILE")
PYTHON_TAG="cp$(echo "$PYTHON_VERSION" | cut -d. -f1,2 | tr -d '.')"
DEST="docker/ray-train/wheels"
mkdir -p "$DEST"

PACKAGES=()
[[ -n "$FLASH_ATTN_VERSION" ]] && PACKAGES+=("flash-attn:${FLASH_ATTN_VERSION}")
[[ -n "$TRANSFORMER_ENGINE_VERSION" ]] && PACKAGES+=("transformer-engine-torch:${TRANSFORMER_ENGINE_VERSION}")

PACKAGES_STR=$(IFS=','; echo "${PACKAGES[*]}")
echo "Fetching cached wheels: ${PACKAGES_STR:-none}"
if bash "$SCRIPT_DIR/lib/fetch_wheels.sh" --dest-dir "$DEST" --bucket "${WHEELS_BUCKET:-dlc-cicd-wheels}" \
    --cuda-version "$CUDA_VERSION" --python-tag "$PYTHON_TAG" --packages "$PACKAGES_STR"; then
  echo "WHEEL_CACHE_HIT=true" >> "${GITHUB_ENV:-/dev/null}"
  echo "Wheel cache hit"
else
  echo "WHEEL_CACHE_HIT=false" >> "${GITHUB_ENV:-/dev/null}"
  echo "Wheel cache miss — will build from source"
fi
