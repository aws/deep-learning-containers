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

shopt -s nullglob
FILES=("$SRC_DIR"/tensorflow-*.whl "$SRC_DIR"/tensorflow_model_server)
if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "No TF artifact exported to ${SRC_DIR}; skipping upload"
  exit 0
fi

for f in "${FILES[@]}"; do
  [[ -f "$f" ]] || continue  # the literal model-server path survives nullglob
  key="${PREFIX}$(basename "$f")"
  if aws s3 ls "s3://${BUCKET}/${key}" >/dev/null 2>&1; then
    echo "Already cached: ${key}"
    continue
  fi
  echo "Uploading $(basename "$f") -> s3://${BUCKET}/${key}"
  aws s3 cp --only-show-errors "$f" "s3://${BUCKET}/${key}" || echo "Upload failed (non-fatal)"
done
