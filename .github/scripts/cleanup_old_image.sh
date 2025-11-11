#!/usr/bin/env bash
set -u  # only fail on unset vars

CUTOFF_HOURS=${CUTOFF_HOURS:-24}
# always compute cutoff in UTC for comparison
CUTOFF_TS=$(date -u -d "${CUTOFF_HOURS} hours ago" +%s 2>/dev/null || \
            date -u -v-"${CUTOFF_HOURS}"H +%s)

echo "=== Docker disk usage before cleanup ==="
docker system df -v || echo "(warning: docker system df failed)"
echo

echo "=== Checking images older than ${CUTOFF_HOURS}h (UTC-based) ==="

deleted=0
kept=0

while IFS= read -r line; do
  id=$(awk '{print $1}' <<<"$line")
  name=$(awk '{print $2}' <<<"$line")
  created_at=$(awk '{$1=$2=""; print substr($0,3)}' <<<"$line")

  [ -z "$id" ] && continue
  [ "$name" = "<none>:<none>" ] && continue

  # parse docker UTC timestamp safely
  created_ts=$(date -u -d "$created_at" +%s 2>/dev/null || echo 0)

  if [ "$created_ts" -lt "$CUTOFF_TS" ]; then
    echo "🗑️  Removing old image: $name (created $created_at UTC)"
    if docker rmi -f "$id" >/dev/null 2>&1; then
      deleted=$((deleted+1))
    else
      echo "(warning: failed to remove $name)"
    fi
  else
    kept=$((kept+1))
  fi
done < <(docker images --format '{{.ID}} {{.Repository}}:{{.Tag}} {{.CreatedAt}}' 2>/dev/null || true)

echo
echo "=== Cleanup summary ==="
echo "Images kept:    $kept"
echo "Images deleted: $deleted"
echo

echo "=== Docker disk usage after cleanup ==="
docker system df -v || echo "(warning: docker system df failed)"

echo
echo "=== Host disk space (for /var/lib/docker) ==="
df -h /var/lib/docker 2>/dev/null || df -h /
