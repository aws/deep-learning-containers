#!/usr/bin/env bash
set -euo pipefail
FAILED=0
for DIR in /tmp /var/tmp "$HOME" /; do
  for F in $(ls -A "$DIR" 2>/dev/null); do
    if echo "$F" | grep -qE '\.py$'; then
      echo "FAIL: Stray .py artifact $F found in $DIR"
      FAILED=1
    fi
  done
done
exit $FAILED