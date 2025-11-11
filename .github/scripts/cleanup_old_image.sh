#!/usr/bin/env bash
set -euo pipefail

# configurable cutoff
CUTOFF_HOURS=${CUTOFF_HOURS:-24}
CUTOFF_TS=$(date -u -d "${CUTOFF_HOURS} hours ago" +%s 2>/dev/null || date -u -v-"${CUTOFF_HOURS}"H +%s)

echo "=== Docker disk usage before cleanup ==="
docker system df -v || true
echo

deleted=0
kept=0

echo "=== Checking images older than ${CUTOFF_HOURS}h (UTC) ==="

docker images --format '{{json .}}' | while read -r json; do
  id=$(jq -r '.ID' <<<"$json")
  repo=$(jq -r '.Repository' <<<"$json")
  tag=$(jq -r '.Tag' <<<"$json")
  created_at=$(jq -r '.CreatedAt' <<<"$json")

  # Skip empty or invalid
  [ -z "$id" ] && continue

  # Normalize name
  name="${repo}:${tag}"

  # Convert CreatedAt → epoch (cross-platform)
  if date --version >/dev/null 2>&1; then
    created_ts=$(date -u -d "$created_at" +%s)
  else
    created_ts=$(date -u -j -f "%Y-%m-%d %H:%M:%S %z %Z" "$created_at" +%s)
  fi

  # Compare
  if [ "$created_ts" -lt "$CUTOFF_TS" ]; then
    echo "🗑️  Removing old image: $name (created $created_at)"
    if docker rmi -f "$id" >/dev/null 2>&1; then
      deleted=$((deleted+1))
    else
      echo "(warning: failed to remove $name)"
    fi
  else
    kept=$((kept+1))
  fi
done

echo
echo "=== Cleanup summary ==="
echo "Images kept:    $kept"
echo "Images deleted: $deleted"
echo

echo "=== Docker disk usage after cleanup ==="
docker system df -v || true

echo
echo "=== Disk space for /var/lib/docker ==="
df -h /var/lib/docker 2>/dev/null || df -h /
