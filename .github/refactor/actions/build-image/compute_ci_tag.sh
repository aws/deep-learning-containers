#!/usr/bin/env bash
# Compute the CI image tag from a config file.
#
# Usage:
#   bash scripts/ci/compute_ci_tag.sh --config-file <path> --tag-suffix <suffix>
#
# Example:
#   bash scripts/ci/compute_ci_tag.sh \
#     --config-file .github/config/image/vllm/ec2-amzn2023.yml \
#     --tag-suffix "$GITHUB_RUN_ID"
#
# Output: writes ci-tag=<value> to $GITHUB_OUTPUT
#
# Requires: yq

set -euo pipefail

CONFIG_FILE=""
TAG_SUFFIX=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-file) CONFIG_FILE="$2"; shift 2 ;;
    --tag-suffix)  TAG_SUFFIX="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

[[ -n "$CONFIG_FILE" ]] || { echo "ERROR: --config-file is required" >&2; exit 1; }
[[ -n "$TAG_SUFFIX" ]]  || { echo "ERROR: --tag-suffix is required" >&2; exit 1; }
[[ -f "$CONFIG_FILE" ]] || { echo "ERROR: Config file not found: $CONFIG_FILE" >&2; exit 1; }

FRAMEWORK=$(yq '.metadata.framework' "$CONFIG_FILE")
FRAMEWORK_VERSION=$(yq '.metadata.framework_version' "$CONFIG_FILE")
DEVICE_TYPE=$(yq '.metadata.device_type' "$CONFIG_FILE")
OS_VERSION=$(yq '.metadata.os_version' "$CONFIG_FILE")
CUSTOMER_TYPE=$(yq '.metadata.customer_type' "$CONFIG_FILE")

# Derive short python version (3.12 → py312, 3.13.12 → py313)
PYTHON_RAW=$(yq '.build.python_version // ""' "$CONFIG_FILE")
if [[ -n "$PYTHON_RAW" ]]; then
  MAJOR=$(echo "$PYTHON_RAW" | cut -d. -f1)
  MINOR=$(echo "$PYTHON_RAW" | cut -d. -f2)
  PYTHON_VERSION="py${MAJOR}${MINOR}"
else
  PYTHON_VERSION=""
fi

# Derive short cuda version (12.9.1 → cu129, 13.0.2 → cu130)
CUDA_RAW=$(yq '.build.cuda_version // ""' "$CONFIG_FILE")
if [[ -n "$CUDA_RAW" ]]; then
  CUDA_MAJOR=$(echo "$CUDA_RAW" | cut -d. -f1)
  CUDA_MINOR=$(echo "$CUDA_RAW" | cut -d. -f2)
  CUDA_VERSION="cu${CUDA_MAJOR}${CUDA_MINOR}"
else
  CUDA_VERSION=""
fi

# Build tag from segments, skipping empty ones
SEGMENTS=("${FRAMEWORK}" "${FRAMEWORK_VERSION}" "${DEVICE_TYPE}")
[[ -n "${PYTHON_VERSION}" ]] && SEGMENTS+=("${PYTHON_VERSION}")
[[ -n "${CUDA_VERSION}" ]] && SEGMENTS+=("${CUDA_VERSION}")
SEGMENTS+=("${OS_VERSION}" "${CUSTOMER_TYPE}" "${TAG_SUFFIX}")

# Join with hyphens
CI_TAG=$(IFS='-'; echo "${SEGMENTS[*]}")

# Replace + with . for OCI tag compliance
CI_TAG="${CI_TAG//+/.}"

echo "CI_TAG=${CI_TAG}"

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  echo "ci-tag=${CI_TAG}" >> "$GITHUB_OUTPUT"
fi
