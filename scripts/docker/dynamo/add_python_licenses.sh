#!/bin/bash
# Adds the license text of Python packages whose wheel ships no license file to /root/PYTHON_PACKAGES_LICENSES: the OSS
# compliance tool (setup_oss_compliance.sh) leaves such packages out. Packages not installed in this image are skipped.
#   add_python_licenses.sh <python> <apache-2.0-text-file>
# The Apache-2.0 text comes from the repo's LICENSE (from its "Apache License" heading on).
set -euo pipefail
PYTHON=$1 APACHE_FILE=$2
OUT=/root/PYTHON_PACKAGES_LICENSES

license_text() {
    case "$1" in
    Apache-2.0)
        # The whole license, through its last section; fail if LICENSE's layout changes.
        local text
        text=$(sed -n '/^ *Apache License$/,$p' "$APACHE_FILE")
        grep -q 'Version 2.0, January 2004' <<<"$text" && grep -q '9. Accepting Warranty or Additional Liability' <<<"$text" \
            || { echo "ERROR: no Apache-2.0 text in $APACHE_FILE" >&2; return 1; }
        echo "$text"
        ;;
    *) echo "ERROR: no license text for $1" >&2; return 1 ;;
    esac
}

# distribution  SPDX license  package URL
while read -r DIST LICENSE URL; do
    # Empty when not installed; any other error fails the build.
    VERSION=$("$PYTHON" -c '
import importlib.metadata as m, sys
try:
    print(m.version(sys.argv[1]))
except m.PackageNotFoundError:
    pass' "$DIST")
    [ -n "$VERSION" ] || continue
    # The tool may write the name with _ or . instead of - and in another case.
    if grep -q -i -E "^Package Name: $(sed 's/[-_.]/[-_.]/g' <<<"$DIST")$" "$OUT"; then
        echo "${DIST}: already in ${OUT}"
        continue
    fi
    TEXT=$(license_text "$LICENSE")
    [ -n "$TEXT" ] || { echo "ERROR: empty ${LICENSE} text from ${APACHE_FILE}" >&2; exit 1; }
    printf '\nPackage Name: %s\nPackage Version: %s\nPackage URL: %s\nPackage License: %s\n\n%s\n' \
        "$DIST" "$VERSION" "$URL" "$LICENSE" "$TEXT" >>"$OUT"
    echo "${DIST} ${VERSION}: ${LICENSE} text added to ${OUT}"
done <<'EOF'
nvshmem4py-cu13 Apache-2.0 https://pypi.org/project/nvshmem4py-cu13/
EOF
