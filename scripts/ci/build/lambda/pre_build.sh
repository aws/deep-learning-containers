#!/usr/bin/env bash
# Pre-build hook for Lambda images.
# Downloads the multi-mode concurrency RIC tarball from S3 for targets that need it.
#
# Usage:
#   bash scripts/ci/build/lambda/pre_build.sh --config-file <path>
#
# Side effects:
#   Places the awslambdaric tarball in docker/lambda/artifacts/ (RIC targets only)
#
# Versioning: awslambdaric_version (e.g. 3.1.1) is the Python package version and
# may repeat across RIC releases, so it alone cannot identify a build.
# awslambdaric_release (e.g. 2.0.0.0) is the RIC release version and is the
# provenance key: it selects the S3 path so each image traces to exactly one build
# and rollback is a one-field change. It is required for RIC targets.

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
AWSLAMBDARIC_RELEASE=$(yq '.build.awslambdaric_release // ""' "$CONFIG_FILE")

if [[ "$TARGET" == *preview* ]]; then
  [[ -n "$AWSLAMBDARIC_RELEASE" ]] || { echo "ERROR: awslambdaric_release is required for RIC targets" >&2; exit 1; }
  echo "RIC target detected — downloading RIC tarball (release ${AWSLAMBDARIC_RELEASE}, version ${AWSLAMBDARIC_VERSION})..."
  mkdir -p docker/lambda/artifacts
  aws s3 cp "s3://dlc-cicd-wheels/lambda-ric/${AWSLAMBDARIC_RELEASE}/awslambdaric-${AWSLAMBDARIC_VERSION}.tar.gz" \
    "docker/lambda/artifacts/awslambdaric-${AWSLAMBDARIC_VERSION}.tar.gz" --region us-west-2
  echo "RIC tarball downloaded."
else
  echo "Non-RIC target — skipping RIC tarball download."
fi
