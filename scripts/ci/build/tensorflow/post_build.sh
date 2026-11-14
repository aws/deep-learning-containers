#!/usr/bin/env bash
# Post-build hook for TensorFlow GPU images: on a cache miss, upload the
# source-built artifact exported by build-image (EXPORT_TARGETS) to the
# dlc-cicd-wheels cache so later builds skip the Bazel compile.
set -euo pipefail

if [[ "${TF_CACHE_HIT:-}" != "false" ]]; then
  echo "TF artifact cache hit or non-GPU build: nothing to upload"
  exit 0
fi

BUCKET="${WHEELS_BUCKET:-dlc-cicd-wheels}"
SRC_DIR="/tmp/tf-artifacts/artifacts"
PREFIX="${TF_ARTIFACT_PREFIX:?TF_ARTIFACT_PREFIX not set by pre_build.sh}"

CONFIG_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-file) CONFIG_FILE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done
[[ -f "$CONFIG_FILE" ]] || { echo "ERROR: config file not found: $CONFIG_FILE" >&2; exit 1; }

# Upload only what this job type builds. The export directory lives on the
# runner host, which the fleet reuses across jobs, so a previous job's
# artifact can still be sitting there.
shopt -s nullglob
case "$(yq -r '.metadata.job_type' "$CONFIG_FILE")" in
  training) FILES=("$SRC_DIR"/tensorflow-*.whl) ;;
  inference) FILES=("$SRC_DIR"/tensorflow_model_server) ;;
  *) echo "ERROR: unsupported job_type in $CONFIG_FILE" >&2; exit 1 ;;
esac
if [[ ${#FILES[@]} -eq 0 || ! -f "${FILES[0]}" ]]; then
  echo "No TF artifact exported to ${SRC_DIR}; skipping upload"
  exit 0
fi

for f in "${FILES[@]}"; do
  key="${PREFIX}$(basename "$f")"
  if aws s3 ls "s3://${BUCKET}/${key}" >/dev/null 2>&1; then
    echo "Already cached: ${key}"
    continue
  fi
  echo "Uploading $(basename "$f") -> s3://${BUCKET}/${key}"
  aws s3 cp --only-show-errors "$f" "s3://${BUCKET}/${key}" || echo "Upload failed (non-fatal)"
done
