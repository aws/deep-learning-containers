#!/usr/bin/env bash
# Sync the Lambda vLLM sccache compilation cache to/from S3.
#
# Usage:
#   bash sync_sccache.sh --action pull [--bucket <bucket>]
#   bash sync_sccache.sh --action push [--bucket <bucket>]
#
# Local: docker/lambda/vllm/sccache-cache/  (COPY'd into the wheel-build stage)
# S3:    s3://<bucket>/sccache/lambda-vllm/
set -euo pipefail

ACTION=""
BUCKET="dlc-cicd-wheels"
LOCAL_DIR="docker/lambda/vllm/sccache-cache"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --action) ACTION="$2"; shift 2 ;;
    --bucket) BUCKET="$2"; shift 2 ;;
    --dir)    LOCAL_DIR="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

[[ -n "$ACTION" ]] || { echo "ERROR: --action is required (pull|push)" >&2; exit 1; }

S3_PREFIX="s3://${BUCKET}/sccache/lambda-vllm/"
mkdir -p "${LOCAL_DIR}"

case "${ACTION}" in
  pull)
    echo "Syncing sccache from ${S3_PREFIX} ..."
    aws s3 sync "${S3_PREFIX}" "${LOCAL_DIR}/" --quiet 2>/dev/null \
      && echo "sccache synced ($(du -sh "${LOCAL_DIR}" | cut -f1))" \
      || echo "sccache pull failed (non-fatal, cold cache)"
    ;;
  push)
    echo "Syncing sccache to ${S3_PREFIX} ..."
    aws s3 sync "${LOCAL_DIR}/" "${S3_PREFIX}" --quiet 2>/dev/null \
      && echo "sccache uploaded ($(du -sh "${LOCAL_DIR}" | cut -f1))" \
      || echo "sccache push failed (non-fatal)"
    ;;
  *)
    echo "ERROR: --action must be 'pull' or 'push', got '${ACTION}'" >&2
    exit 1
    ;;
esac
