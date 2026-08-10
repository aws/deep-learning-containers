#!/usr/bin/env bash
# utils_test.sh — Unit tests for the version helpers in utils.sh.
# No dependencies beyond bash. Run directly:
#
#   ./scripts/ci/autocurrency/utils_test.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils.sh"

PASSED=0
FAILED=0

###############################################################################
# assert_newer(upstream, current, expected_rc, description)
#   expected_rc: 0 = upstream is newer, 1 = it is not, 2 = uncomparable
###############################################################################
assert_newer() {
  local upstream="$1" current="$2" expected="$3" description="$4"
  local actual=0

  is_newer_version "${upstream}" "${current}" 2>/dev/null || actual=$?

  if [[ "${actual}" == "${expected}" ]]; then
    PASSED=$((PASSED + 1))
    echo "  ok   ${description}"
  else
    FAILED=$((FAILED + 1))
    echo "  FAIL ${description}"
    echo "         is_newer_version('${upstream}', '${current}') returned ${actual}, expected ${expected}"
  fi
}

assert_segments() {
  local version="$1" expected="$2"
  local actual
  actual="$(release_segments "${version}")"

  if [[ "${actual}" == "${expected}" ]]; then
    PASSED=$((PASSED + 1))
    echo "  ok   release_segments('${version}') == '${expected}'"
  else
    FAILED=$((FAILED + 1))
    echo "  FAIL release_segments('${version}')"
    echo "         returned '${actual}', expected '${expected}'"
  fi
}

echo "release_segments: strips local and suffix segments"
assert_segments "0.17.0" "0 17 0"
assert_segments "0.5.13+dlc1" "0 5 13"
assert_segments "0.17.0.post1" "0 17 0"
assert_segments "0.20.0.dev361" "0 20 0"
assert_segments "1.0.0.1" "1 0 0 1"
assert_segments "0.17" "0 17"
assert_segments "notaversion" ""

echo ""
echo "is_newer_version: plain numeric releases"
assert_newer "0.17.0" "0.16.0" 0 "newer minor is detected"
assert_newer "0.16.0" "0.16.0" 1 "identical versions are not newer"
assert_newer "0.9.0" "0.16.0" 1 "older minor is not newer"
assert_newer "0.10.0" "0.9.10" 0 "segments compare numerically, not lexically"
assert_newer "0.17.1" "0.17.0" 0 "newer patch is detected"

echo ""
echo "is_newer_version: local version segments (regression — issue #1)"
assert_newer "0.5.14" "0.5.13+dlc1" 0 "+dlc suffix on current does not abort"
assert_newer "0.5.14+dlc1" "0.5.13" 0 "+dlc suffix on upstream does not abort"
assert_newer "0.5.13" "0.5.13+dlc1" 1 "+dlc rebuild is not behind its own release"
assert_newer "0.5.14" "0.20.0.dev361" 1 ".dev suffix compares on the release portion"
assert_newer "0.17.0" "0.17.0.post1" 1 ".post suffix compares on the release portion"

echo ""
echo "is_newer_version: segments beyond the third (regression — issue #1)"
assert_newer "1.0.0.1" "1.0.0" 0 "fourth segment is not truncated away"
assert_newer "1.0.0" "1.0.0.1" 1 "fourth segment on current is respected"
assert_newer "1.0.0.2" "1.0.0.1" 0 "fourth segments compare against each other"

echo ""
echo "is_newer_version: unpadded and zero-padded input"
assert_newer "0.17" "0.16.0" 0 "missing patch segment pads to zero"
assert_newer "0.17.0" "0.17" 1 "explicit zero patch equals a missing one"
assert_newer "0.09.0" "0.8.0" 0 "leading zeros do not become octal"

echo ""
echo "is_newer_version: malformed input reports rather than aborting"
assert_newer "notaversion" "0.17.0" 2 "unparseable upstream returns 2"
assert_newer "0.17.0" "notaversion" 2 "unparseable current returns 2"

echo ""
echo "============================================================"
echo "Passed: ${PASSED}  Failed: ${FAILED}"
if [[ ${FAILED} -gt 0 ]]; then
  exit 1
fi
