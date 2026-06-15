#!/usr/bin/env bash
# Discover image configs matching a glob pattern.
#
# Usage:
#   bash scripts/ci/discover_configs.sh --pattern ".github/config/image/base/*.yml"
#
# Output: JSON array to stdout:
#   [{"config_file": "path/to/file.yml", "name": "image-name"}]
#
# Requires: yq, jq

set -euo pipefail

PATTERN=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pattern) PATTERN="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

[[ -n "$PATTERN" ]] || { echo "ERROR: --pattern is required" >&2; exit 1; }

CONFIGS="[]"

for f in $PATTERN; do
  [[ -f "$f" ]] || continue

  NAME=$(yq '.image.name' "$f")

  CONFIGS=$(echo "$CONFIGS" | jq -c \
    --arg f "$f" \
    --arg n "$NAME" \
    '. + [{"config_file": $f, "name": $n}]')
done

echo "$CONFIGS"
