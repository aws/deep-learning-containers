#!/usr/bin/env bash
# Generate a release specification YAML from an image config file.
#
# Usage:
#   bash generate_release_spec.sh --config-file <path>
#
# Output: writes release-spec=<yaml> to $GITHUB_OUTPUT
#
# The release spec is consumed by reusable-release-image.yml to determine
# which registries to push to, what tags to apply, and release settings.
#
# Requires: yq

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

# Required fields
FRAMEWORK=$(yq '.metadata.framework' "$CONFIG_FILE")
VERSION=$(yq '.metadata.framework_version' "$CONFIG_FILE")

[[ -n "$FRAMEWORK" && "$FRAMEWORK" != "null" ]] || { echo "ERROR: metadata.framework is required" >&2; exit 1; }
[[ -n "$VERSION" && "$VERSION" != "null" ]] || { echo "ERROR: metadata.framework_version is required" >&2; exit 1; }

# Optional metadata fields
ARCH_TYPE=$(yq '.metadata.arch_type // ""' "$CONFIG_FILE")
JOB_TYPE=$(yq '.metadata.job_type // ""' "$CONFIG_FILE")
DEVICE_TYPE=$(yq '.metadata.device_type // ""' "$CONFIG_FILE")
OS_VERSION=$(yq '.metadata.os_version // ""' "$CONFIG_FILE")
CUSTOMER_TYPE=$(yq '.metadata.customer_type // ""' "$CONFIG_FILE")
PLATFORM=$(yq '.metadata.platform // ""' "$CONFIG_FILE")

# Derived fields
PYTHON_RAW=$(yq '.build.python_version // ""' "$CONFIG_FILE")
if [[ -n "$PYTHON_RAW" ]]; then
  MAJOR=$(echo "$PYTHON_RAW" | cut -d. -f1)
  MINOR=$(echo "$PYTHON_RAW" | cut -d. -f2)
  PYTHON_VERSION="py${MAJOR}${MINOR}"
else
  PYTHON_VERSION=""
fi

CUDA_RAW=$(yq '.build.cuda_version // ""' "$CONFIG_FILE")
if [[ -n "$CUDA_RAW" ]]; then
  CUDA_MAJOR=$(echo "$CUDA_RAW" | cut -d. -f1)
  CUDA_MINOR=$(echo "$CUDA_RAW" | cut -d. -f2)
  CUDA_VERSION="cu${CUDA_MAJOR}${CUDA_MINOR}"
else
  CUDA_VERSION=""
fi

TRANSFORMERS_VERSION=$(yq '.build.transformers_version // ""' "$CONFIG_FILE")

# Release flags
FORCE_RELEASE=$(yq '.release.force_release // ""' "$CONFIG_FILE")
PUBLIC_REGISTRY=$(yq '.release.public_registry // ""' "$CONFIG_FILE")
PRIVATE_REGISTRY=$(yq '.release.private_registry // ""' "$CONFIG_FILE")
ENABLE_SOCI=$(yq '.release.enable_soci // ""' "$CONFIG_FILE")
echo "Generating release spec:"
echo "  Framework: ${FRAMEWORK}"
echo "  Version: ${VERSION}"

# Generate release spec YAML
SPEC=""
SPEC+="framework: \"${FRAMEWORK}\""$'\n'
SPEC+="version: \"${VERSION}\""$'\n'

[[ -n "$ARCH_TYPE" ]]       && SPEC+="arch_type: \"${ARCH_TYPE}\""$'\n'
[[ -n "$JOB_TYPE" ]]        && SPEC+="job_type: \"${JOB_TYPE}\""$'\n'
[[ -n "$DEVICE_TYPE" ]]     && SPEC+="device_type: \"${DEVICE_TYPE}\""$'\n'
[[ -n "$PYTHON_VERSION" ]]  && SPEC+="python_version: \"${PYTHON_VERSION}\""$'\n'
[[ -n "$OS_VERSION" ]]      && SPEC+="os_version: \"${OS_VERSION}\""$'\n'
[[ -n "$CUSTOMER_TYPE" ]]   && SPEC+="customer_type: \"${CUSTOMER_TYPE}\""$'\n'
[[ -n "$CUDA_VERSION" ]]    && SPEC+="cuda_version: \"${CUDA_VERSION}\""$'\n'
[[ -n "$TRANSFORMERS_VERSION" ]] && SPEC+="transformers_version: \"${TRANSFORMERS_VERSION}\""$'\n'
[[ -n "$PLATFORM" ]]        && SPEC+="platform: \"${PLATFORM}\""$'\n'
[[ -n "$FORCE_RELEASE" ]]   && SPEC+="force_release: ${FORCE_RELEASE}"$'\n'
[[ -n "$PUBLIC_REGISTRY" ]] && SPEC+="public_registry: ${PUBLIC_REGISTRY}"$'\n'
[[ -n "$PRIVATE_REGISTRY" ]] && SPEC+="private_registry: ${PRIVATE_REGISTRY}"$'\n'
[[ -n "$ENABLE_SOCI" ]]    && SPEC+="enable_soci: ${ENABLE_SOCI}"$'\n'

echo "$SPEC"

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  {
    echo "release-spec<<EOF"
    echo "$SPEC"
    echo "EOF"
  } >> "$GITHUB_OUTPUT"
fi
