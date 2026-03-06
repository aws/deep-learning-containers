#!/bin/bash
# Verify installed package versions match pinned requirements.txt.
# Runs inside the container (repo mounted at /workdir).
#
# Usage (CI): docker exec ${CONTAINER_ID} bash /workdir/test/ray/test_pinned_versions.sh
# Usage (manual): bash test/ray/test_pinned_versions.sh
set -eo pipefail

# Resolve requirements.txt
# CI: repo mounted at /workdir inside container
# Manual: script lives at test/ray/, so repo root is ../../ relative to script
if [ -f /workdir/scripts/ray/requirements.txt ]; then
    REQ_FILE="/workdir/scripts/ray/requirements.txt"
else
    SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
    REPO_ROOT="${SCRIPT_DIR}/../.."
    REQ_FILE="${REPO_ROOT}/scripts/ray/requirements.txt"
fi

if [ ! -f "$REQ_FILE" ]; then
    echo "ERROR: requirements.txt not found"
    exit 1
fi

PASS=0; FAIL=0
pass() { echo "  PASS: $1"; PASS=$((PASS + 1)); }
fail() { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }

echo "=========================================="
echo "Pinned Version Check"
echo "Requirements: ${REQ_FILE}"
echo "=========================================="
echo

# Get installed versions
INSTALLED=$(uv pip list --python /opt/venv/bin/python --format=freeze 2>/dev/null || \
            pip list --format=freeze 2>/dev/null || true)

if [ -z "$INSTALLED" ]; then
    echo "ERROR: could not get installed packages"
    exit 1
fi

# Parse requirements.txt: extract "package==version" lines
while IFS= read -r line; do
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ -z "${line// }" ]] && continue

    line="${line%%#*}"
    line="$(echo "$line" | xargs)"

    # Handle extras like ray[serve]==2.54.0
    if [[ "$line" =~ ^([a-zA-Z0-9._-]+)(\[.*\])?==([^ ]+)$ ]]; then
        PKG="${BASH_REMATCH[1]}"
        EXPECTED="${BASH_REMATCH[3]}"
    else
        continue
    fi

    PKG_NORMALIZED=$(echo "$PKG" | tr '_' '-' | tr '[:upper:]' '[:lower:]')

    ACTUAL=$(echo "$INSTALLED" | tr '[:upper:]' '[:lower:]' | tr '_' '-' | \
        grep -i "^${PKG_NORMALIZED}==" | head -1 | cut -d'=' -f3)

    if [ -z "$ACTUAL" ]; then
        fail "${PKG}: expected ${EXPECTED}, NOT INSTALLED"
    elif [ "$ACTUAL" = "$(echo "$EXPECTED" | tr '[:upper:]' '[:lower:]')" ]; then
        pass "${PKG}==${EXPECTED}"
    else
        fail "${PKG}: expected ${EXPECTED}, got ${ACTUAL}"
    fi
done < "$REQ_FILE"

echo
echo "=========================================="
echo "Results: ${PASS} passed, ${FAIL} failed"
if [ "${FAIL}" -gt 0 ]; then
    echo "VERSION CHECK FAILED"
    exit 1
else
    echo "ALL VERSIONS MATCH"
fi
echo "=========================================="
