#!/usr/bin/env bash
# Extract built wheels from Docker wheel-export stage and upload to S3.
#
# Usage:
#   bash upload_wheels.sh --bucket <bucket> --cuda-version <ver> --torch-version <ver> \
#     --image-uri <uri> --dockerfile <path> --packages "flash-attn:2.8.3,transformer-engine-torch:2.12.0"

set -euo pipefail

BUCKET="dlc-cicd-wheels"
CUDA=""
TORCH_VERSION=""
IMAGE=""
DOCKERFILE=""
PACKAGES=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bucket)         BUCKET="$2"; shift 2 ;;
    --cuda-version)   CUDA="$2"; shift 2 ;;
    --torch-version)  TORCH_VERSION="$2"; shift 2 ;;
    --image-uri)      IMAGE="$2"; shift 2 ;;
    --dockerfile)     DOCKERFILE="$2"; shift 2 ;;
    --packages)       PACKAGES="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

[[ -n "$BUCKET" ]]        || { echo "ERROR: --bucket is required" >&2; exit 1; }
[[ -n "$CUDA" ]]          || { echo "ERROR: --cuda-version is required" >&2; exit 1; }
[[ -n "$TORCH_VERSION" ]] || { echo "ERROR: --torch-version is required" >&2; exit 1; }
[[ -n "$IMAGE" ]]         || { echo "ERROR: --image-uri is required" >&2; exit 1; }
[[ -n "$DOCKERFILE" ]]    || { echo "ERROR: --dockerfile is required" >&2; exit 1; }

# Derive short CUDA string: 13.0.2 → cu130
CUDA_MAJOR=$(echo "$CUDA" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA" | cut -d. -f2)
CUDA_SHORT="cu${CUDA_MAJOR}${CUDA_MINOR}"

# Derive short torch string: 2.13.0 → torch213
TORCH_MAJOR=$(echo "$TORCH_VERSION" | cut -d. -f1)
TORCH_MINOR=$(echo "$TORCH_VERSION" | cut -d. -f2)
TORCH_SHORT="torch${TORCH_MAJOR}${TORCH_MINOR}"

EXPORT_DIR=$(mktemp -d)

# Forward EXTRA_BUILD_ARGS (populated by resolve_build_args.py + exported to GITHUB_ENV
# by the build-image action) so BuildKit derives the same stage-cache key as the primary
# build. Without this, wheel-export resolves to a stale cached stage (e.g. builder-flash-attn
# built against a previous torch ABI) and poisons the uploaded wheel.
BUILD_ARGS=()
for var in ${EXTRA_BUILD_ARGS:-}; do
  if [[ -n "${!var:-}" ]]; then
    BUILD_ARGS+=("--build-arg" "${var}=${!var}")
  fi
done

# Match the extra build-args the primary build passes (see .github/actions/build-image/action.yml).
# These are hardcoded in action.yml, not in EXTRA_BUILD_ARGS, so they must be replicated here
# to keep the wheel-export invocation's stage-cache key aligned with the primary build.
[[ -n "${FRAMEWORK:-}" ]]              && BUILD_ARGS+=("--build-arg" "FRAMEWORK=${FRAMEWORK}")
[[ -n "${FRAMEWORK_VERSION:-}" ]]      && BUILD_ARGS+=("--build-arg" "FRAMEWORK_VERSION=${FRAMEWORK_VERSION}")
[[ -n "${JOB_TYPE:-}" ]]               && BUILD_ARGS+=("--build-arg" "CONTAINER_TYPE=${JOB_TYPE}")
[[ -n "${FRAMEWORK_SHORT_VERSION:-}" ]] && BUILD_ARGS+=("--build-arg" "FRAMEWORK_SHORT_VERSION=${FRAMEWORK_SHORT_VERSION}")
[[ -n "${USE_PREBUILT_WHEEL:-}" ]]     && BUILD_ARGS+=("--build-arg" "USE_PREBUILT_WHEEL=${USE_PREBUILT_WHEEL}")
if [[ -n "${PYTHON_VERSION:-}" ]]; then
  PYTHON_SHORT_VERSION=$(echo "${PYTHON_VERSION}" | cut -d. -f1,2)
  BUILD_ARGS+=("--build-arg" "PYTHON_SHORT_VERSION=${PYTHON_SHORT_VERSION}")
fi
BUILD_ARGS+=("--build-arg" "CACHE_REFRESH=$(date +%Y-%m-%d)")

docker buildx build --progress=plain --target wheel-export --output "type=local,dest=${EXPORT_DIR}" \
  "${BUILD_ARGS[@]}" \
  -f "${DOCKERFILE}" . 2>/dev/null || {
  echo "wheel-export stage not available — extracting from runtime image"
  CID=$(docker create "${IMAGE}" /bin/true)
  docker cp "${CID}:/tmp/built_wheels/" "${EXPORT_DIR}/wheels/" 2>/dev/null || true
  docker rm "${CID}" &>/dev/null || true
}

IFS=',' read -ra SPECS <<< "$PACKAGES"
for spec in "${SPECS[@]}"; do
  [[ -z "$spec" ]] && continue
  PKG="${spec%%:*}"
  PKG_UNDER="${PKG//-/_}"

  WHL=$(find "${EXPORT_DIR}" -name "${PKG_UNDER}*.whl" 2>/dev/null | head -1)
  if [[ -z "${WHL}" ]]; then
    echo "No wheel found for ${PKG}"
    continue
  fi

  # Verify flash-attn wheel ABI before uploading — prevent torch-ABI cache poisoning.
  # If BuildKit resolved a stale cached wheel-export stage (e.g. built against torch 2.12),
  # its compiled .so will reference `c10::impl::cow::materialize_cow_storage`, which was
  # renamed to `materialize_cow` in torch 2.13. Refuse to upload such a wheel.
  if [[ "${PKG_UNDER}" == "flash_attn" ]]; then
    TMP_EXTRACT=$(mktemp -d)
    if unzip -qo "${WHL}" -d "${TMP_EXTRACT}" 2>/dev/null; then
      SO=$(find "${TMP_EXTRACT}" -name "*.so" | head -1)
      if [[ -n "${SO}" ]]; then
        if nm --undefined-only --demangle "${SO}" 2>/dev/null | grep -q "materialize_cow_storage"; then
          echo "ERROR: flash-attn wheel references torch-2.12 symbol 'c10::impl::cow::materialize_cow_storage'"
          echo "       (torch 2.13 renamed this to 'materialize_cow'; the wheel appears built against a stale torch)"
          echo "       Refusing to upload — this would poison the S3 wheel-cache."
          rm -rf "${TMP_EXTRACT}"
          exit 1
        else
          echo "flash-attn wheel ABI verified: no stale torch-2.12 symbols"
        fi
      fi
    fi
    rm -rf "${TMP_EXTRACT}"
  fi

  FNAME=$(basename "${WHL}")
  S3_KEY="wheels/${PKG_UNDER}/${CUDA_SHORT}/${TORCH_SHORT}/${FNAME}"

  if aws s3 ls "s3://${BUCKET}/${S3_KEY}" &>/dev/null; then
    echo "Overwriting existing S3 object at: ${S3_KEY}"
  fi

  echo "Uploading ${FNAME} -> s3://${BUCKET}/${S3_KEY}"
  aws s3 cp "${WHL}" "s3://${BUCKET}/${S3_KEY}" || echo "Upload failed (non-fatal)"
done

rm -rf "${EXPORT_DIR}"
