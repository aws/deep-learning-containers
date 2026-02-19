#!/usr/bin/env bash
set -euo pipefail
FAILED=0

# /var/tmp must be empty
CONTENTS=$(ls -A /var/tmp 2>/dev/null)
if [ -n "$CONTENTS" ]; then
  echo "FAIL: /var/tmp is not empty: $CONTENTS"
  FAILED=1
fi

# /tmp should only contain expected files
for F in $(ls -A /tmp/ 2>/dev/null); do
  case "$F" in
    .*|*system*|*System*|*dkms*|*hsperfdata*) continue ;;
  esac
  echo "FAIL: Unexpected file in /tmp: $F"
  FAILED=1
done
exit $FAILED