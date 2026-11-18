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
#   EXPORT_TARGETS      - on miss: tf-artifact-export:<TF_EXPORT_DIR>
#   TF_EXPORT_DIR       - on miss: fresh, empty export directory for this job
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
# Start from an empty prebuilt/ (except .gitkeep): a file left by an earlier job
# on a reused host would otherwise count as a hit for this job's key.
find "$DEST" -mindepth 1 -maxdepth 1 ! -name .gitkeep -exec rm -rf {} +

# The artifact this job type needs: training caches the TF wheel, inference the
# model server binary. Only that artifact counts as a hit, so a stray file of
# the other type in the prefix can never pass for a cached build.
case "$(yq -r '.metadata.job_type' "$CONFIG_FILE")" in
  training) PATTERN="tensorflow-*.whl" ;;
  inference) PATTERN="tensorflow_model_server" ;;
  *) echo "ERROR: unsupported job_type in $CONFIG_FILE" >&2; exit 1 ;;
esac

echo "Looking for TF artifact s3://${BUCKET}/${PREFIX}${PATTERN} ..."
if ! aws s3 cp --recursive --only-show-errors --exclude "*" --include "${PATTERN}" \
    "s3://${BUCKET}/${PREFIX}" "$DEST/"; then
  echo "WARNING: S3 download failed; treating as a cache miss" >&2
fi

# Only real files count. A literal pattern (tensorflow_model_server) survives
# nullglob even when nothing matched, so check each entry with -f.
FOUND=()
shopt -s nullglob
for f in "$DEST"/${PATTERN}; do
  if [[ -f "$f" ]]; then FOUND+=("$f"); fi
done
shopt -u nullglob
if [[ ${#FOUND[@]} -gt 1 ]]; then
  echo "ERROR: ${#FOUND[@]} artifacts match ${PATTERN} in s3://${BUCKET}/${PREFIX}: expected one" >&2
  exit 1
fi

if [[ ${#FOUND[@]} -eq 1 ]]; then
  echo "Cache hit: $(basename "${FOUND[0]}")"
  {
    echo "TF_CACHE_HIT=true"
    echo "USE_PREBUILT_WHEEL=1"
  } >> "${GITHUB_ENV:-/dev/null}"
else
  # Runner hosts are reused across jobs and the BuildKit local exporter does not
  # clear its destination, so export into a new empty directory: post_build.sh
  # then only ever sees what this job's build wrote.
  EXPORT_DIR=$(mktemp -d /tmp/tf-artifacts.XXXXXX)
  echo "Cache miss: building from source (export dir ${EXPORT_DIR})"
  {
    echo "TF_CACHE_HIT=false"
    echo "USE_PREBUILT_WHEEL=0"
    echo "TF_EXPORT_DIR=${EXPORT_DIR}"
    echo "EXPORT_TARGETS=tf-artifact-export:${EXPORT_DIR}"
  } >> "${GITHUB_ENV:-/dev/null}"
fi
echo "TF_ARTIFACT_PREFIX=${PREFIX}" >> "${GITHUB_ENV:-/dev/null}"
