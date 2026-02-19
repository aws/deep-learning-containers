#!/usr/bin/env bash
set -euo pipefail
EXPECTED="${1:?Usage: check_os_version.sh <os-version>}"
OS_NAME=$(echo "$EXPECTED" | grep -oP '^[a-zA-Z]+')
OS_VER=$(echo "$EXPECTED" | grep -oP '[0-9]+\.[0-9]+')
OS_RELEASE=$(cat /etc/os-release)
if ! echo "$OS_RELEASE" | grep -qi "$OS_NAME"; then
  echo "FAIL: Expected OS $OS_NAME not found in /etc/os-release"
  echo "$OS_RELEASE"
  exit 1
fi
if ! echo "$OS_RELEASE" | grep -q "$OS_VER"; then
  echo "FAIL: Expected version $OS_VER not found in /etc/os-release"
  echo "$OS_RELEASE"
  exit 1
fi