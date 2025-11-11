#!/usr/bin/env bash
set -euo pipefail

# Configurable cutoff age (default 1 day)
CUTOFF_HOURS=${CUTOFF_HOURS:-24}
CUTOFF_TS=$(date -d "${CUTOFF_HOURS} hours ago" +%s)

echo "=== Docker disk usage before cleanup ==="
docker system df -v || true
echo

echo "=== Checking images older than ${CUTOFF_HOURS}h ==="
deleted=0
kept=0

docker images --format '{{.ID}} {{.Repository}}:{{.Tag}} {{.CreatedAt}}' \
  | while read -r id name created_at _; do
      # skip dangling images (no repo:tag)
      [ "$name" = "<none>:<none>" ] && continue
      created_ts=$(date -d "$created_at" +%s 2>/dev/null || echo 0)
      if (( created_ts < CUTOFF_TS )); then
        echo "🗑️  Removing old image: $name (created $created_at)"
        docker rmi -f "$id" >/dev/null 2>&1 && ((deleted++)) || true
      else
        ((kept++))
      fi
    done

echo
echo "=== Cleanup summary ==="
echo "Images kept:   $kept"
echo "Images deleted: $deleted"
echo

echo "=== Docker disk usage after cleanup ==="
docker system df -v || true
