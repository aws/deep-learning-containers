#!/usr/bin/env bash
# Pre-build hook for Lambda images. Two independent responsibilities, each gated on
# the config's build.target (all Lambda images share framework=lambda, so this one
# hook runs for base/cupy/pytorch/sglang/vllm — it must act only where relevant):
#
#   1. *preview* targets: download the multi-mode concurrency RIC tarball from S3
#      into docker/lambda/artifacts/.
#   2. *vllm* targets: fetch a cached vLLM wheel from S3 (skip the ~85-min source
#      compile) and pull the sccache compiler cache. Mirrors scripts/ci/build/
#      vllm_server/pre_build.sh, but namespaced under lambda-vllm/ (py3.13 / cu130
#      build env is distinct from the amzn2023 wheel).
#
# Usage:  bash scripts/ci/build/lambda/pre_build.sh --config-file <path>
#
# Outputs (written to $GITHUB_ENV, consumed by the build-image action):
#   USE_PREBUILT_WHEEL - "1" if a cached vLLM wheel was fetched, else "0"
#   WHEEL_CACHE_HIT    - "true"/"false" (post_build reads this to skip re-upload)
#   EXPORT_TARGETS     - on cache miss, tells build-image to export the wheel +
#                        sccache scratch stages for post_build to upload
#
# Side effects:
#   docker/lambda/artifacts/                 (RIC tarball, preview targets)
#   docker/lambda/vllm/prebuilt_wheels/      (cached wheel, vllm targets on hit)
#   docker/lambda/vllm/sccache-cache/        (sccache, vllm targets)
#
# Versioning: awslambdaric_version (e.g. 3.1.1) is the Python package version and
# may repeat across RIC releases, so it alone cannot identify a build.
# awslambdaric_release (e.g. 2.0.0.0) is the RIC release version and is the
# provenance key: it selects the S3 path so each image traces to exactly one build
# and rollback is a one-field change. It is required for RIC targets.

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

BUCKET="${WHEELS_BUCKET:-dlc-cicd-wheels}"
TARGET=$(yq '.build.target' "$CONFIG_FILE")

# ---------------------------------------------------------------------------
# 1. RIC tarball (preview targets)
# ---------------------------------------------------------------------------
if [[ "$TARGET" == *preview* ]]; then
  AWSLAMBDARIC_VERSION=$(yq '.build.awslambdaric_version // "3.1.1"' "$CONFIG_FILE")
  AWSLAMBDARIC_RELEASE=$(yq '.build.awslambdaric_release // ""' "$CONFIG_FILE")
  [[ -n "$AWSLAMBDARIC_RELEASE" ]] || { echo "ERROR: awslambdaric_release is required for RIC targets" >&2; exit 1; }
  echo "RIC target detected — downloading RIC tarball (release ${AWSLAMBDARIC_RELEASE}, version ${AWSLAMBDARIC_VERSION})..."
  mkdir -p docker/lambda/artifacts
  aws s3 cp "s3://${BUCKET}/lambda-ric/${AWSLAMBDARIC_RELEASE}/awslambdaric-${AWSLAMBDARIC_VERSION}.tar.gz" \
    "docker/lambda/artifacts/awslambdaric-${AWSLAMBDARIC_VERSION}.tar.gz" --region us-west-2
  echo "RIC tarball downloaded."
fi

# ---------------------------------------------------------------------------
# 2. vLLM wheel cache + sccache (vllm targets only)
# ---------------------------------------------------------------------------
if [[ "$TARGET" == *vllm* ]]; then
  CUDA_VERSION=$(yq '.build.cuda_version' "$CONFIG_FILE")
  VLLM_REF=$(yq '.build.vllm_ref' "$CONFIG_FILE")
  VLLM_VERSION=$(yq '.build.vllm_version' "$CONFIG_FILE")
  ARCH_LIST=$(yq '.build.torch_cuda_arch_list // "8.0 8.6 8.9 12.0"' "$CONFIG_FILE")
  USE_SCCACHE=$(yq '.build.use_sccache // "false"' "$CONFIG_FILE")

  # build context dirs (empty by default so the Dockerfile COPYs always succeed)
  mkdir -p docker/lambda/vllm/prebuilt_wheels docker/lambda/vllm/sccache-cache

  WHEEL_HIT="false"
  echo "Fetching cached Lambda vLLM wheel..."
  if bash "$SCRIPT_DIR/lib/fetch_wheels.sh" \
      --cuda-version "$CUDA_VERSION" --vllm-ref "$VLLM_REF" \
      --vllm-version "$VLLM_VERSION" --arch-list "$ARCH_LIST" --bucket "$BUCKET"; then
    WHEEL_HIT="true"
  fi

  echo "WHEEL_CACHE_HIT=${WHEEL_HIT}" >> "${GITHUB_ENV:-/dev/null}"
  if [[ "$WHEEL_HIT" == "true" ]]; then
    echo "USE_PREBUILT_WHEEL=1" >> "${GITHUB_ENV:-/dev/null}"
  else
    echo "USE_PREBUILT_WHEEL=0" >> "${GITHUB_ENV:-/dev/null}"
    echo "EXPORT_TARGETS=vllm-wheel-export:docker/lambda/vllm/prebuilt_wheels,vllm-sccache-export:docker/lambda/vllm/sccache-cache" >> "${GITHUB_ENV:-/dev/null}"
  fi

  if [[ "$USE_SCCACHE" == "true" ]]; then
    echo "Syncing sccache from S3..."
    bash "$SCRIPT_DIR/lib/sync_sccache.sh" --action pull --bucket "$BUCKET" || echo "sccache pull failed (cold cache okay)"
  fi
fi

if [[ "$TARGET" != *preview* && "$TARGET" != *vllm* ]]; then
  echo "Non-RIC, non-vLLM target — no pre-build actions."
fi
