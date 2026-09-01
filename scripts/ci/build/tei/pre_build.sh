#!/usr/bin/env bash
# Pre-build hook for TEI (Text Embeddings Inference).
# Clones huggingface/text-embeddings-inference at tag v${framework_version}
# into the Docker build context so the Dockerfile can COPY backends/core/router/Cargo.*
# from a stable path.
#
# Usage:
#   bash scripts/ci/build/tei/pre_build.sh --config-file <path>
#
# Inputs:
#   --config-file - config file path (reads metadata.framework_version)
#
# Side effects:
#   Places TEI upstream source at ${DOCKER_DIR}/tei-src/
#   (e.g. docker/tei/1.9.3/cpu/tei-src/ and docker/tei/1.9.3/gpu/tei-src/)

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

REPO_ROOT=$(pwd)

DOCKERFILE=$(yq '.build.dockerfile' "$CONFIG_FILE")
DOCKER_DIR=$(dirname "$DOCKERFILE")
FRAMEWORK_VERSION=$(yq '.metadata.framework_version' "$CONFIG_FILE")
TEI_TAG="v${FRAMEWORK_VERSION}"
TEI_REPO="https://github.com/huggingface/text-embeddings-inference.git"

echo "Cloning text-embeddings-inference (tag: ${TEI_TAG})..."
rm -rf /tmp/tei-src
git clone --depth 1 --branch "${TEI_TAG}" "${TEI_REPO}" /tmp/tei-src

echo "Placing TEI source in build context (${DOCKER_DIR}/tei-src)..."
DEST="${REPO_ROOT}/${DOCKER_DIR}/tei-src"
rm -rf "${DEST}"
mkdir -p "${DEST}"
cp -r /tmp/tei-src/backends "${DEST}/"
cp -r /tmp/tei-src/core "${DEST}/"
cp -r /tmp/tei-src/router "${DEST}/"
cp /tmp/tei-src/Cargo.toml "${DEST}/"
cp /tmp/tei-src/Cargo.lock "${DEST}/"
cp /tmp/tei-src/rust-toolchain.toml "${DEST}/"

echo "TEI source ready:"
ls "${DEST}/"

echo "Cleaning up clone..."
rm -rf /tmp/tei-src
