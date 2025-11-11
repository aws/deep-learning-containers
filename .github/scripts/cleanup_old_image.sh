#!/usr/bin/env bash
set -euo pipefail

# Delete images older than 1 day (24h)
cutoff=$(date -d '1 day ago' +%s)

docker images --format '{{.ID}} {{.Repository}}:{{.Tag}} {{.CreatedAt}}' \
  | while read -r id name created_at _; do
      created_ts=$(date -d "$created_at" +%s 2>/dev/null || echo 0)
      if (( created_ts < cutoff )); then
        echo "Deleting old image: $name ($id, created $created_at)"
        docker rmi -f "$id" || true
      fi
    done
