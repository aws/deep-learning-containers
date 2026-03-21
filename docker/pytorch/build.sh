#!/usr/bin/env bash
# Build PyTorch AL2023 DLC images (base + eks targets).
# Usage: ./build.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# shellcheck source=versions.env
source "$SCRIPT_DIR/versions.env"

TAG_PREFIX="pytorch-al2023"
CUDA_SHORT="cu${CUDA_VERSION%%.*}${CUDA_VERSION#*.}"
CUDA_SHORT="${CUDA_SHORT//.}"  # 12.8.1 -> cu1281 -> cu1281, strip remaining dots
# We want cu128 (major.minor without patch)
CUDA_SHORT="cu$(echo "$CUDA_VERSION" | cut -d. -f1,2 | tr -d '.')"
BASE_TAG="${TORCH_VERSION}-${CUDA_SHORT}-al2023"

echo "==> Building base image: ${TAG_PREFIX}:${BASE_TAG}"
docker build --progress=plain --target base \
    -t "${TAG_PREFIX}:${BASE_TAG}-${BUILD_DATE}" \
    -t "${TAG_PREFIX}:${BASE_TAG}" \
    -t "${TAG_PREFIX}:latest" \
    -f "$SCRIPT_DIR/Dockerfile" "$REPO_ROOT"

echo "==> Building EKS image: ${TAG_PREFIX}:${BASE_TAG}-eks"
docker build --progress=plain --target eks \
    -t "${TAG_PREFIX}:${BASE_TAG}-eks-${BUILD_DATE}" \
    -t "${TAG_PREFIX}:${BASE_TAG}-eks" \
    -f "$SCRIPT_DIR/Dockerfile" "$REPO_ROOT"

echo "==> Done. Images:"
docker images --filter "reference=${TAG_PREFIX}:*" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
