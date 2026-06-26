#!/usr/bin/env bash
# Pre-build hook for Lambda images.
# Downloads the thread-mode RIC preview tarball from S3 for preview targets.
#
# Usage:
#   bash scripts/ci/build/lambda/pre_build.sh --config-file <path>
#
# Side effects:
#   Places awslambdaric tarball in docker/lambda/artifacts/ (preview targets only)

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

TARGET=$(yq '.build.target' "$CONFIG_FILE")
AWSLAMBDARIC_VERSION=$(yq '.build.awslambdaric_version // "3.1.1"' "$CONFIG_FILE")

if [[ "$TARGET" == *preview* ]]; then
  echo "Preview target detected — downloading RIC tarball..."
  mkdir -p docker/lambda/artifacts
  aws s3 cp "s3://dlc-cicd-wheels/lambda-ric/awslambdaric-${AWSLAMBDARIC_VERSION}.tar.gz" \
    "docker/lambda/artifacts/awslambdaric-${AWSLAMBDARIC_VERSION}.tar.gz" --region us-west-2
  echo "RIC tarball downloaded."
else
  echo "Non-preview target — skipping RIC tarball download."
fi
