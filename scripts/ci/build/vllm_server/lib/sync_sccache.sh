#!/usr/bin/env bash
# Sync sccache cache to/from S3.
#
# Usage:
#   bash sync_sccache.sh --action pull --framework vllm [--bucket <bucket>]
#   bash sync_sccache.sh --action push --framework vllm [--bucket <bucket>]
#
# Local: docker/<framework>/sccache-cache/
# S3:    s3://<bucket>/sccache/<framework>/

set -euo pipefail

ACTION=""
FRAMEWORK=""
BUCKET="dlc-cicd-wheels"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --action)    ACTION="$2"; shift 2 ;;
    --framework) FRAMEWORK="$2"; shift 2 ;;
    --bucket)    BUCKET="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

[[ -n "$ACTION" ]]    || { echo "ERROR: --action is required (pull|push)" >&2; exit 1; }
[[ -n "$FRAMEWORK" ]] || { echo "ERROR: --framework is required" >&2; exit 1; }

LOCAL_DIR="docker/${FRAMEWORK}/sccache-cache"
S3_PREFIX="s3://${BUCKET}/sccache/${FRAMEWORK}/"

mkdir -p "${LOCAL_DIR}"

case "${ACTION}" in
  pull)
    echo "Syncing sccache cache from ${S3_PREFIX} ..."
    aws s3 sync "${S3_PREFIX}" "${LOCAL_DIR}/" --quiet 2>/dev/null \
      && echo "sccache cache synced ($(du -sh "${LOCAL_DIR}" | cut -f1))" \
      || echo "sccache cache sync failed (non-fatal, cold cache)"
    ;;
  push)
    echo "Syncing sccache cache to ${S3_PREFIX} ..."
    aws s3 sync "${LOCAL_DIR}/" "${S3_PREFIX}" --quiet 2>/dev/null \
      && echo "sccache cache uploaded ($(du -sh "${LOCAL_DIR}" | cut -f1))" \
      || echo "sccache cache upload failed (non-fatal)"
    ;;
  *)
    echo "ERROR: --action must be 'pull' or 'push', got '${ACTION}'" >&2
    exit 1
    ;;
esac
