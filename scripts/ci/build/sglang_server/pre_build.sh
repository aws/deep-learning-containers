#!/usr/bin/env bash
# Pre-build hook for the SGLang AL2023 images (framework sglang_server).
# Downloads private patches for the config's sglang_ref from the CI models bucket into
# scripts/docker/sglang/amzn2023/patches/, where the Dockerfile applies them. The CI
# runner has read access to dlc-cicd-models (same account), so no AWS creds enter the
# docker build itself.
#
# S3 patches are fetched for pull_request builds from branches of this repository and for
# builds of main, which include every release, so they ship in released images while
# staying out of the repository. Fork PRs, other branches and local runs skip them.
# Patches are keyed by sglang_ref, so they stop applying once the ref moves.
#
# Usage:
#   bash scripts/ci/build/sglang_server/pre_build.sh --config-file <path>
#
# Inputs:
#   --config-file            - config file path
#   SGLANG_PATCHES_S3_PREFIX - S3 prefix (env var, default:
#                              s3://dlc-cicd-models/build-patches/sglang_server)
#
# Side effects:
#   Copies <prefix>/<sglang_ref>/*.patch into scripts/docker/sglang/amzn2023/patches/
set -euo pipefail

CONFIG_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-file) CONFIG_FILE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

[[ -n "$CONFIG_FILE" ]] || { echo "ERROR: --config-file is required" >&2; exit 1; }
[[ -f "$CONFIG_FILE" ]] || { echo "ERROR: Config file not found: $CONFIG_FILE" >&2; exit 1; }

case "${GITHUB_EVENT_NAME:-}" in
  "")
    echo "Not a CI build: skipping S3 patches"
    exit 0
    ;;
  pull_request)
    HEAD_REPO=$(python3 -c 'import json, os
pr = json.load(open(os.environ["GITHUB_EVENT_PATH"]))["pull_request"]
print((pr["head"].get("repo") or {}).get("full_name", ""))')
    if [[ "$HEAD_REPO" != "${GITHUB_REPOSITORY:-}" ]]; then
      echo "Pull request from another repository (${HEAD_REPO:-unknown}): skipping S3 patches"
      exit 0
    fi
    ;;
  *)
    if [[ "${GITHUB_REF:-}" != "refs/heads/main" ]]; then
      echo "Not a main build (${GITHUB_REF:-unknown ref}): skipping S3 patches"
      exit 0
    fi
    ;;
esac

SGLANG_REF=$(yq -r '.build.sglang_ref // ""' "$CONFIG_FILE")
[[ -n "$SGLANG_REF" ]] || { echo "No build.sglang_ref in $CONFIG_FILE: skipping S3 patches"; exit 0; }

PREFIX="${SGLANG_PATCHES_S3_PREFIX:-s3://dlc-cicd-models/build-patches/sglang_server}"
DEST="scripts/docker/sglang/amzn2023/patches"
mkdir -p "$DEST"

aws s3 cp --recursive --only-show-errors --exclude "*" --include "*.patch" \
  "${PREFIX}/${SGLANG_REF}/" "${DEST}/"
echo "Patches to apply for ${SGLANG_REF}: $(find "$DEST" -maxdepth 1 -name '*.patch' | wc -l)"
