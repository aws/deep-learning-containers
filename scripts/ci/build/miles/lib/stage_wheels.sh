#!/usr/bin/env bash
# Stage the Miles release wheels into the build context.
#
# Usage:
#   bash stage_wheels.sh --version-dir <docker/miles/vX.Y.Z> --wheels-url <release download URL> [--bucket <bucket>]
#
# Stages every wheel in wheel-artifacts-official.sha256 that
# wheel-artifact-deviations.json does not rebuild or omit into
# <version-dir>/wheels/, checked against its official SHA-256.
#
# With --bucket, wheels are read from the content-addressed mirror
# s3://<bucket>/wheels/miles/<sha256>/<file>. On a miss the wheel is downloaded
# from the upstream release, verified, and uploaded to that key, so the mirror
# only ever holds the official bytes. Without --bucket (local builds) wheels come
# straight from the upstream release, with the same check.

set -euo pipefail

VERSION_DIR=""
WHEELS_URL=""
BUCKET=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version-dir) VERSION_DIR="$2"; shift 2 ;;
    --wheels-url)  WHEELS_URL="$2"; shift 2 ;;
    --bucket)      BUCKET="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

[[ -n "${VERSION_DIR}" ]] || { echo "ERROR: --version-dir is required" >&2; exit 1; }
[[ -n "${WHEELS_URL}" ]]  || { echo "ERROR: --wheels-url is required" >&2; exit 1; }

OFFICIAL="${VERSION_DIR}/wheel-artifacts-official.sha256"
DEVIATIONS="${VERSION_DIR}/wheel-artifact-deviations.json"
DEST="${VERSION_DIR}/wheels"
mkdir -p "${DEST}"

# Official artifacts the image rebuilds from source or omits are not staged.
SKIPPED=$(python3 -c 'import json, sys; print("\n".join(json.load(open(sys.argv[1]))))' "${DEVIATIONS}")

while read -r sha file; do
  [[ -z "${sha}" || "${sha}" == \#* ]] && continue
  grep -Fxq "${file}" <<<"${SKIPPED}" && continue

  target="${DEST}/${file}"
  key="wheels/miles/${sha}/${file}"
  if [[ -f "${target}" ]] && sha256sum -c --quiet - <<<"${sha}  ${target}" 2>/dev/null; then
    echo "staged: ${file} (already present)"
    continue
  fi

  rm -f "${target}"
  if [[ -n "${BUCKET}" ]] && aws s3 cp --only-show-errors "s3://${BUCKET}/${key}" "${target}" 2>/dev/null; then
    origin="s3://${BUCKET}/${key}"
  else
    curl -fsSL --retry 3 --retry-delay 2 -o "${target}" "${WHEELS_URL}/${file//+/%2B}"
    origin="${WHEELS_URL}"
  fi

  if ! sha256sum -c --quiet - <<<"${sha}  ${target}"; then
    rm -f "${target}"
    echo "ERROR: ${file} from ${origin} does not match its official sha256 ${sha}." >&2
    echo "  If upstream replaced it, review the new artifact before changing ${OFFICIAL} or ${DEVIATIONS}." >&2
    exit 1
  fi

  if [[ -n "${BUCKET}" && "${origin}" == "${WHEELS_URL}" ]]; then
    aws s3 cp --only-show-errors "${target}" "s3://${BUCKET}/${key}" \
      || echo "WARNING: mirror upload failed for ${file} (non-fatal)"
  fi
  echo "staged: ${file} (${origin})"
done <"${OFFICIAL}"
