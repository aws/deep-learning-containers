#!/usr/bin/env bash
set -euo pipefail
FAILED=0
[ -f "$HOME/.viminfo" ] && echo "FAIL: .viminfo exists" && FAILED=1
if [ -f "$HOME/.bash_history" ] && [ -s "$HOME/.bash_history" ]; then
  echo "FAIL: .bash_history contains history"
  FAILED=1
fi
exit $FAILED