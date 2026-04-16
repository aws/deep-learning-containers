#!/usr/bin/env bash
# sync_sccache.sh — Sync sccache local cache to/from S3.
#
# Usage:
#   sync_sccache.sh pull <framework> [bucket]   # S3 → local (before build)
#   sync_sccache.sh push <framework> [bucket]   # local → S3 (after build)
#
# Local cache: /tmp/sccache-cache/<framework>/
# S3 layout:   s3://<bucket>/sccache/<framework>/
set -euo pipefail

ACTION="$1"; FRAMEWORK="$2"; BUCKET="${3:-dlc-cicd-wheels}"
LOCAL_DIR="/tmp/sccache-cache/${FRAMEWORK}"
S3_PREFIX="s3://${BUCKET}/sccache/${FRAMEWORK}/"

mkdir -p "${LOCAL_DIR}"

case "${ACTION}" in
  pull)
    echo "⬇️  Syncing sccache cache from ${S3_PREFIX} ..."
    aws s3 sync "${S3_PREFIX}" "${LOCAL_DIR}/" --quiet 2>/dev/null \
      && echo "✅ sccache cache synced ($(du -sh "${LOCAL_DIR}" | cut -f1))" \
      || echo "⚠️  sccache cache sync failed (non-fatal, cold cache)"
    ;;
  push)
    echo "⬆️  Syncing sccache cache to ${S3_PREFIX} ..."
    aws s3 sync "${LOCAL_DIR}/" "${S3_PREFIX}" --quiet 2>/dev/null \
      && echo "✅ sccache cache uploaded ($(du -sh "${LOCAL_DIR}" | cut -f1))" \
      || echo "⚠️  sccache cache upload failed (non-fatal)"
    ;;
  *)
    echo "Usage: $0 {pull|push} <framework> [bucket]" >&2
    exit 1
    ;;
esac
