#!/usr/bin/env bash
# Post-build hook for TensorFlow GPU images: on a cache miss, upload the
# source-built artifact exported by build-image (EXPORT_TARGETS) to the
# dlc-cicd-wheels cache so later builds skip the Bazel compile.
#
# pre_build.sh points the export at a fresh, empty directory (TF_EXPORT_DIR),
# so anything found there was written by this job's build. The export must hold
# exactly one artifact of this job type plus its .sha256 sidecar (written by the
# Dockerfile after the GPU-target check), and the artifact must match it:
# build-image treats a failed export as non-fatal, so a missing or truncated
# export fails here instead of being uploaded. The artifact is uploaded first and
# the checksum last, so a reader never sees a checksum for a partial object.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/checksum.sh
source "$SCRIPT_DIR/lib/checksum.sh"

if [[ "${TF_CACHE_HIT:-}" != "false" ]]; then
  echo "TF artifact cache hit or non-GPU build: nothing to upload"
  exit 0
fi

BUCKET="${WHEELS_BUCKET:-dlc-cicd-wheels}"
PREFIX="${TF_ARTIFACT_PREFIX:?TF_ARTIFACT_PREFIX not set by pre_build.sh}"
EXPORT_DIR="${TF_EXPORT_DIR:?TF_EXPORT_DIR not set by pre_build.sh}"
SRC_DIR="${EXPORT_DIR}/artifacts"
trap 'rm -rf "${EXPORT_DIR}"' EXIT

CONFIG_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-file) CONFIG_FILE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done
[[ -f "$CONFIG_FILE" ]] || { echo "ERROR: config file not found: $CONFIG_FILE" >&2; exit 1; }

case "$(yq -r '.metadata.job_type' "$CONFIG_FILE")" in
  training) PATTERN="tensorflow-*.whl" ;;
  inference) PATTERN="tensorflow_model_server" ;;
  *) echo "ERROR: unsupported job_type in $CONFIG_FILE" >&2; exit 1 ;;
esac

FILES=()
shopt -s nullglob
for f in "$SRC_DIR"/${PATTERN}; do
  if [[ -f "$f" ]]; then FILES+=("$f"); fi
done
shopt -u nullglob
if [[ ${#FILES[@]} -ne 1 ]]; then
  echo "::error::Expected one exported ${PATTERN} in ${SRC_DIR}, found ${#FILES[@]}: the tf-artifact-export export failed (see the build step log)"
  exit 1
fi

f="${FILES[0]}"
name="$(basename "$f")"
if ! artifact_verified "$f"; then
  echo "::error::Exported ${name} is missing ${name}.sha256 or does not match it (truncated export?): not uploading"
  exit 1
fi
digest=$(tr -d '[:space:]' < "${f}.sha256")

key="${PREFIX}${name}"
# Skip only when S3 already holds this exact artifact: its sidecar matches the
# digest just built. A cached object without a sidecar, or with a different
# one, failed verification in pre_build.sh and is overwritten here.
remote_sum="${EXPORT_DIR}/remote.sha256"
if aws s3 cp --only-show-errors "s3://${BUCKET}/${key}.sha256" "$remote_sum" >/dev/null 2>&1 \
    && [[ "$(tr -d '[:space:]' < "$remote_sum")" == "$digest" ]] \
    && aws s3 ls "s3://${BUCKET}/${key}" >/dev/null 2>&1; then
  echo "Already cached: ${key} (sha256 ${digest})"
  exit 0
fi

echo "Uploading ${name} (sha256 ${digest}) -> s3://${BUCKET}/${key}"
if aws s3 cp --only-show-errors "$f" "s3://${BUCKET}/${key}" \
    && aws s3 cp --only-show-errors "${f}.sha256" "s3://${BUCKET}/${key}.sha256"; then
  echo "Uploaded ${key} and its checksum"
else
  echo "::warning::Uploading ${name} to s3://${BUCKET}/${key} failed: the next build of this key compiles from source again"
fi
