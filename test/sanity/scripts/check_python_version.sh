#!/usr/bin/env bash
set -euo pipefail
PY_RAW="${1:?Usage: check_python_version.sh <python-version>}"
PY_MAJOR="${PY_RAW:2:1}"
PY_MINOR="${PY_RAW:3}"
EXPECTED="${PY_MAJOR}.${PY_MINOR}"
INSTALLED=$(python3 --version)
echo "Installed: $INSTALLED, Expected: Python $EXPECTED"
if ! echo "$INSTALLED" | grep -q "$EXPECTED"; then
  echo "FAIL: Expected Python $EXPECTED, got: $INSTALLED"
  exit 1
fi