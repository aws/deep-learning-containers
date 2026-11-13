#!/usr/bin/env bash
# Pre-build hook for TensorFlow GPU images (training wheel / inference model server).
#
# Fetches the source-built artifact from the dlc-cicd-wheels cache into the
# image's prebuilt/ context dir. On a hit the Dockerfile skips the Bazel build;
# on a miss it compiles from source and build-image exports the artifact stage
# for post_build.sh to upload.
#
# Env exported via GITHUB_ENV:
#   USE_PREBUILT_WHEEL  - "1" on cache hit, "0" on miss
#   TF_ARTIFACT_PREFIX  - S3 prefix used by post_build.sh
#   TF_CACHE_HIT        - "true" / "false"
#   EXPORT_TARGETS      - on miss: tf-artifact-export:/tmp/tf-artifacts
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-file) CONFIG_FILE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done
[[ -f "$CONFIG_FILE" ]] || { echo "ERROR: config file not found: $CONFIG_FILE" >&2; exit 1; }

# Only GPU configs that build TF from source set cuda_compute_capabilities;
# CPU and older GPU images use upstream binaries.
if [[ "$(yq -r '.metadata.device_type' "$CONFIG_FILE")" != "gpu" ]] \
  || [[ "$(yq -r '.build.cuda_compute_capabilities // ""' "$CONFIG_FILE")" == "" ]]; then
  echo "No source build for $CONFIG_FILE: upstream binaries, skipping TF artifact cache"
  exit 0
fi

BUCKET="${WHEELS_BUCKET:-dlc-cicd-wheels}"
PREFIX=$(bash "$SCRIPT_DIR/lib/cache_prefix.sh" --config-file "$CONFIG_FILE")
DEST="$(dirname "$(yq -r '.build.dockerfile' "$CONFIG_FILE")")/prebuilt"
mkdir -p "$DEST"

echo "Looking for TF artifact in s3://${BUCKET}/${PREFIX} ..."
aws s3 cp --recursive --only-show-errors "s3://${BUCKET}/${PREFIX}" "$DEST/" 2>/dev/null || true

# Training caches a wheel, inference a model server binary: either one is a hit.
if compgen -G "$DEST/tensorflow-*.whl" >/dev/null || [[ -f "$DEST/tensorflow_model_server" ]]; then
  echo "Cache hit: $(ls "$DEST" | grep -v '^\.gitkeep$' | tr '\n' ' ')"
  {
    echo "TF_CACHE_HIT=true"
    echo "USE_PREBUILT_WHEEL=1"
  } >> "${GITHUB_ENV:-/dev/null}"
else
  echo "Cache miss: building from source"
  {
    echo "TF_CACHE_HIT=false"
    echo "USE_PREBUILT_WHEEL=0"
    echo "EXPORT_TARGETS=tf-artifact-export:/tmp/tf-artifacts"
  } >> "${GITHUB_ENV:-/dev/null}"
fi
echo "TF_ARTIFACT_PREFIX=${PREFIX}" >> "${GITHUB_ENV:-/dev/null}"
