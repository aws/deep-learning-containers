#!/usr/bin/env bash
# Push PyTorch AL2023 DLC images to an ECR registry.
# Usage: ./push.sh <registry-uri> [--dry-run]
#   e.g. ./push.sh 123456789.dkr.ecr.us-west-2.amazonaws.com/pytorch-al2023
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=versions.env
source "$SCRIPT_DIR/versions.env"

REGISTRY="${1:?Usage: $0 <registry-uri> [--dry-run]}"
DRY_RUN="${2:-}"

TAG_PREFIX="pytorch-al2023"
CUDA_SHORT="cu$(echo "$CUDA_VERSION" | cut -d. -f1,2 | tr -d '.')"
BASE_TAG="${TORCH_VERSION}-${CUDA_SHORT}-al2023"

TAGS=(
    "${BASE_TAG}-${BUILD_DATE}"
    "${BASE_TAG}"
    "latest"
    "${BASE_TAG}-eks-${BUILD_DATE}"
    "${BASE_TAG}-eks"
)

# Login to ECR if registry looks like an ECR URI
if [[ "$REGISTRY" == *".dkr.ecr."* ]]; then
    REGION="$(echo "$REGISTRY" | grep -oP '(?<=ecr\.)[^.]+' || true)"
    if [[ -n "$REGION" ]]; then
        echo "==> Logging in to ECR ($REGION)"
        aws ecr get-login-password --region "$REGION" | \
            docker login --username AWS --password-stdin "${REGISTRY%%/*}"
    fi
fi

for tag in "${TAGS[@]}"; do
    local_image="${TAG_PREFIX}:${tag}"
    remote_image="${REGISTRY}:${tag}"

    if ! docker image inspect "$local_image" &>/dev/null; then
        echo "SKIP $local_image (not found locally)"
        continue
    fi

    echo "==> ${local_image} -> ${remote_image}"
    if [[ "$DRY_RUN" == "--dry-run" ]]; then
        echo "    (dry-run, skipping)"
    else
        docker tag "$local_image" "$remote_image"
        docker push "$remote_image"
    fi
done

echo "==> Done."
