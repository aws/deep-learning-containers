#!/usr/bin/env bash
set -euo pipefail
FRAMEWORK="${1:?Usage: check_framework_version.sh <framework> <version>}"
EXPECTED="${2:?}"
case "$FRAMEWORK" in
  pytorch) MODULE="torch" ;;
  *) MODULE="$FRAMEWORK" ;;
esac
INSTALLED=$(python3 -c "import ${MODULE}; print(${MODULE}.__version__)")
INSTALLED="${INSTALLED%%+*}"
echo "Installed ${MODULE}: $INSTALLED, Expected: $EXPECTED"
if ! echo "$INSTALLED" | grep -q "^${EXPECTED}"; then
  echo "FAIL: Expected ${MODULE} ${EXPECTED}, got: $INSTALLED"
  exit 1
fi