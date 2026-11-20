#!/usr/bin/env bash
# Pre-build hook for Miles: stage the locked release wheels into the build
# context from the CI wheels bucket, mirroring them from upstream on a miss.
#
# Usage:
#   bash scripts/ci/build/miles/pre_build.sh --config-file <path>
#
# Inputs:
#   --config-file  - config file path
#   WHEELS_BUCKET  - S3 bucket for the wheel mirror (env var). Unset for local
#                    builds, which fetch from the upstream release instead.
#
# Side effects:
#   Places the wheels in docker/miles/v<miles_version>/wheels/

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

MILES_VERSION=$(yq -r '.build.miles_version' "$CONFIG_FILE")
WHEELS_REPO=$(yq -r '.build.miles_wheels_repo' "$CONFIG_FILE")
WHEELS_TAG=$(yq -r '.build.miles_wheels_tag' "$CONFIG_FILE")

bash "$SCRIPT_DIR/lib/stage_wheels.sh" \
  --version-dir "docker/miles/v${MILES_VERSION}" \
  --wheels-url "https://github.com/${WHEELS_REPO}/releases/download/${WHEELS_TAG}" \
  --bucket "${WHEELS_BUCKET:-}"
