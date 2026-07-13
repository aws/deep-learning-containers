#!/usr/bin/env bash
# Pre-build hook for WhisperX.
# Downloads the pyannote speaker-diarization weights tarball from the CI models
# bucket into the Docker build context so the Dockerfile can COPY it in. Mirrors
# the S3 step in the standalone whisperx-docker/scripts/build.sh. The CI runner
# has read access to dlc-cicd-models (same account), so no AWS creds enter the
# docker build itself.
#
# Usage:
#   bash scripts/ci/build/whisperx/pre_build.sh --config-file <path>
#
# Side effects:
#   Places the tarball at docker/whisperx/pyannote-diarization.tar.gz
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
DEST="${REPO_ROOT}/docker/whisperx/pyannote-diarization.tar.gz"
PYANNOTE_S3_URI="${PYANNOTE_S3_URI:-s3://dlc-cicd-models/whisperx-models/speaker-diarization-community-1.tar.gz}"

echo "Downloading pyannote weights: ${PYANNOTE_S3_URI}"
aws s3 cp "${PYANNOTE_S3_URI}" "${DEST}"
echo "pyannote weights ready: $(ls -la "${DEST}")"
