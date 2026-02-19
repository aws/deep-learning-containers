#!/usr/bin/env bash
set -euo pipefail
CACHE_DIR="$HOME/.cache"
[ ! -d "$CACHE_DIR" ] && exit 0
FAILED=0
for F in $(ls -A "$CACHE_DIR" 2>/dev/null); do
  case "$F" in
    pip*) continue ;;
  esac
  echo "FAIL: Unexpected file in cache dir: $F"
  FAILED=1
done
exit $FAILED