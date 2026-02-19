#!/usr/bin/env bash
set -euo pipefail
if ! command -v conda &>/dev/null; then
  echo "SKIP: conda not installed"
  exit 0
fi
OFFENDING=$(conda list --explicit 2>/dev/null | grep "repo.anaconda.com" || true)
if [ -n "$OFFENDING" ]; then
  echo "FAIL: Packages from repo.anaconda.com found:"
  echo "$OFFENDING"
  exit 1
fi
echo "No anaconda packages found"