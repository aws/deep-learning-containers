#!/usr/bin/env bash
set -euo pipefail

# Framework license check: /license.txt must match the published object.
# Usage: check_license_file.sh <container_id> <framework> <framework_version>
# Example: check_license_file.sh abc123 pytorch_runtime 2.13.0

CONTAINER_ID="${1:?Usage: check_license_file.sh <container_id> <framework> <framework_version>}"
FRAMEWORK="${2:?Usage: check_license_file.sh <container_id> <framework> <framework_version>}"
FRAMEWORK_VERSION="${3:?Usage: check_license_file.sh <container_id> <framework> <framework_version>}"

BUCKET_URL="https://aws-dlc-licenses.s3.amazonaws.com"

# pytorch_runtime -> pytorch bucket prefix.
case "$FRAMEWORK" in
  pytorch_runtime) PREFIX="pytorch" ;;
  *) PREFIX="$FRAMEWORK" ;;
esac

MINOR=$(echo "$FRAMEWORK_VERSION" | cut -d. -f1,2)
OBJECT_URL="${BUCKET_URL}/${PREFIX}-${MINOR}/license.txt"

WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT

if ! docker cp "${CONTAINER_ID}:/license.txt" "${WORKDIR}/image_license.txt" 2>/dev/null; then
  echo "FAIL: /license.txt is missing from the image"
  echo "      The Dockerfile must fetch it from ${OBJECT_URL}"
  exit 1
fi

if ! curl -fsLo "${WORKDIR}/published_license.txt" "$OBJECT_URL" 2>/dev/null; then
  echo "FAIL: no license published at ${OBJECT_URL}"
  echo "      Upload it before releasing this version."
  exit 1
fi

if cmp -s "${WORKDIR}/image_license.txt" "${WORKDIR}/published_license.txt"; then
  echo "PASS: /license.txt matches ${PREFIX}-${MINOR}/license.txt ($(wc -c <"${WORKDIR}/image_license.txt") bytes)"
  exit 0
fi

echo "FAIL: /license.txt does not match ${OBJECT_URL}"
echo "      A stale version path in the Dockerfile is the usual cause."
echo "--- diff (image vs published, first 20 lines) ---"
diff "${WORKDIR}/image_license.txt" "${WORKDIR}/published_license.txt" | head -20 || true
exit 1
