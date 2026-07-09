#!/usr/bin/env bash
# Compute the CI image tag from a config file.
#
# Format: {image.name}-{framework_version}-{device_type}-{py_short}-{cu_short}-{tag_suffix}
# Example: sglang-ec2-amzn2023-0.5.12.dlc1-gpu-py312-cu130-pr-123
#
# Usage:
#   bash compute_ci_tag.sh --config-file <path> --tag-suffix <suffix>
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

IMAGE_NAME=$(yq '.image.name' "$CONFIG_FILE")
FRAMEWORK_VERSION=$(yq '.metadata.framework_version' "$CONFIG_FILE")
DEVICE_TYPE=$(yq '.metadata.device_type' "$CONFIG_FILE")

PYTHON_RAW=$(yq '.build.python_version // ""' "$CONFIG_FILE")
if [[ -n "$PYTHON_RAW" ]]; then
  MAJOR=$(echo "$PYTHON_RAW" | cut -d. -f1)
  MINOR=$(echo "$PYTHON_RAW" | cut -d. -f2)
  PYTHON_SHORT="py${MAJOR}${MINOR}"
else
  PYTHON_SHORT=""
fi

CUDA_RAW=$(yq '.build.cuda_version // ""' "$CONFIG_FILE")
if [[ -n "$CUDA_RAW" ]]; then
  CUDA_MAJOR=$(echo "$CUDA_RAW" | cut -d. -f1)
  CUDA_MINOR=$(echo "$CUDA_RAW" | cut -d. -f2)
  CUDA_SHORT="cu${CUDA_MAJOR}${CUDA_MINOR}"
else
  CUDA_SHORT=""
fi

SEGMENTS=("${IMAGE_NAME}" "${FRAMEWORK_VERSION}" "${DEVICE_TYPE}")
[[ -n "${PYTHON_SHORT}" ]] && SEGMENTS+=("${PYTHON_SHORT}")
[[ -n "${CUDA_SHORT}" ]] && SEGMENTS+=("${CUDA_SHORT}")
SEGMENTS+=("${TAG_SUFFIX}")

CI_TAG=$(IFS='-'; echo "${SEGMENTS[*]}")

# Replace + with . for OCI tag compliance
CI_TAG="${CI_TAG//+/.}"

echo "CI_TAG=${CI_TAG}"

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  echo "ci-tag=${CI_TAG}" >> "$GITHUB_OUTPUT"
fi
