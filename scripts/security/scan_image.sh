#!/usr/bin/env sh

# shellcheck shell=sh

# Attempt to enable errexit, nounset, and pipefail (where supported)
if set -euo pipefail 2>/dev/null; then
    :
else
    set -eu
    # Enable pipefail when the shell supports it
    (set -o pipefail) 2>/dev/null && set -o pipefail || true
fi

usage() {
    cat <<EOT
usage: $0 IMAGE_TAG

Environment variables:
  VULN_SEVERITY   Severity levels to scan for (default: ${VULN_SEVERITY:-CRITICAL})
  VULN_FAIL_ON    When "true", exit non-zero on findings (default: ${VULN_FAIL_ON:-true})
  GENERATE_SBOM   When "true", emit SBOM artifacts (default: ${GENERATE_SBOM:-true})
  SBOM_FORMAT     SBOM output format (default: ${SBOM_FORMAT:-spdx-json})
  SBOM_DIR        Directory to store SBOM files (default: ${SBOM_DIR:-sbom})
EOT
}

if [ "$#" -ne 1 ]; then
    echo "error: missing IMAGE_TAG argument" >&2
    usage >&2
    exit 1
fi

IMAGE_TAG=$1
VULN_SEVERITY=${VULN_SEVERITY:-CRITICAL}
VULN_FAIL_ON=${VULN_FAIL_ON:-true}
GENERATE_SBOM=${GENERATE_SBOM:-true}
SBOM_FORMAT=${SBOM_FORMAT:-spdx-json}
SBOM_DIR=${SBOM_DIR:-sbom}

TRIVY_BIN=${TRIVY_BIN:-trivy}

if ! command -v "$TRIVY_BIN" >/dev/null 2>&1; then
    echo "error: required scanner '$TRIVY_BIN' not found in PATH" >&2
    exit 1
fi

echo "Using scanner: $($TRIVY_BIN --version 2>/dev/null | head -n 1)"

SAFE_TAG=$(printf '%s' "$IMAGE_TAG" | tr '/:' '__')

mkdir -p "$SBOM_DIR"

is_true() {
    case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
        1|t|true|y|yes) return 0 ;;
        *) return 1 ;;
    esac
}

if is_true "$GENERATE_SBOM"; then
    SBOM_PATH="$SBOM_DIR/${SAFE_TAG}.sbom.json"
    echo "Generating SBOM (${SBOM_FORMAT}) at $SBOM_PATH"
    if ! "$TRIVY_BIN" sbom -f "$SBOM_FORMAT" -o "$SBOM_PATH" "$IMAGE_TAG"; then
        echo "error: SBOM generation failed for $IMAGE_TAG" >&2
        exit 1
    fi
else
    echo "Skipping SBOM generation (GENERATE_SBOM=$GENERATE_SBOM)"
fi

SCAN_ARGS="image --severity $VULN_SEVERITY --ignore-unfixed --no-progress"
if is_true "$VULN_FAIL_ON"; then
    SCAN_ARGS="$SCAN_ARGS --exit-code 1"
else
    SCAN_ARGS="$SCAN_ARGS --exit-code 0"
fi

echo "Scanning image $IMAGE_TAG (severity >= $VULN_SEVERITY, fail_on=$VULN_FAIL_ON)"
if ! $TRIVY_BIN $SCAN_ARGS "$IMAGE_TAG"; then
    echo "error: vulnerability scan reported issues for $IMAGE_TAG" >&2
    exit 1
fi

echo "Scan completed successfully for $IMAGE_TAG"
if is_true "$GENERATE_SBOM"; then
    echo "SBOM saved to $SBOM_PATH"
fi
