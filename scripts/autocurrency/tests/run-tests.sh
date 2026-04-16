#!/usr/bin/env bash
# run-tests.sh — Test harness for the autocurrency version detection system.
#
# Approach: "option b" — mock older current versions in our config files
# so the detection logic sees a version gap, without needing real upstream
# releases. Also mocks `docker` to avoid network/container dependencies.
#
# Usage:
#   bash scripts/autocurrency/tests/run-tests.sh           # run all tests
#   bash scripts/autocurrency/tests/run-tests.sh <pattern>  # run matching tests
#
# Requirements: yq, jq (same as the real scripts)
#
# Platform notes:
#   The production scripts target Linux (GitHub Actions on ubuntu-latest).
#   This test harness patches sandbox copies of the scripts so they also
#   work on macOS (grep -oP → grep+sed, sed -i → sed -i '').

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
AUTOCURRENCY_DIR="${REPO_ROOT}/scripts/autocurrency"

IS_MACOS=false
[[ "$(uname)" == "Darwin" ]] && IS_MACOS=true

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0
FILTER="${1:-}"

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

###############################################################################
# Version arithmetic helpers — derive test versions from real config values
###############################################################################

# Bump the patch segment: "0.17.1" → "0.17.2", "0.5.9" → "0.5.10"
_bump_patch_version() {
  local ver="$1"
  IFS='.' read -ra parts <<< "${ver}"
  while [[ ${#parts[@]} -lt 3 ]]; do parts+=("0"); done
  parts[2]=$(( 10#${parts[2]} + 1 ))
  echo "${parts[0]}.${parts[1]}.${parts[2]}"
}

# Decrement the minor segment: "0.17.1" → "0.16.0", "0.5.9" → "0.4.0"
_decrement_minor_version() {
  local ver="$1"
  IFS='.' read -ra parts <<< "${ver}"
  while [[ ${#parts[@]} -lt 3 ]]; do parts+=("0"); done
  parts[1]=$(( 10#${parts[1]} - 1 ))
  parts[2]=0
  echo "${parts[0]}.${parts[1]}.${parts[2]}"
}

###############################################################################
# Portable helpers (work on both Linux and macOS)
###############################################################################

# Read the ARG BASE_IMAGE= value from a Dockerfile (no grep -P needed)
get_base_image_from_dockerfile() {
  local file="$1"
  grep "^ARG BASE_IMAGE=" "${file}" | head -1 | sed 's/^ARG BASE_IMAGE=//'
}

# Portable in-place sed (handles macOS requiring backup extension)
portable_sed_i() {
  if [[ "${IS_MACOS}" == "true" ]]; then
    sed -i '' "$@"
  else
    sed -i "$@"
  fi
}

###############################################################################
# Sandbox: isolated copy of configs + patched scripts
###############################################################################

setup_sandbox() {
  SANDBOX=$(mktemp -d)
  export SANDBOX

  mkdir -p "${SANDBOX}/.github/config/image"
  mkdir -p "${SANDBOX}/docker/vllm"
  mkdir -p "${SANDBOX}/docker/sglang"
  mkdir -p "${SANDBOX}/scripts/autocurrency"
  mkdir -p "${SANDBOX}/mock-bin"

  # Copy real files
  cp "${REPO_ROOT}/.github/config/autocurrency-tracker.yml" "${SANDBOX}/.github/config/"
  cp "${REPO_ROOT}/.github/config/image/vllm-ec2.yml" "${SANDBOX}/.github/config/image/"
  cp "${REPO_ROOT}/.github/config/image/vllm-sagemaker.yml" "${SANDBOX}/.github/config/image/"
  cp "${REPO_ROOT}/.github/config/image/sglang-ec2.yml" "${SANDBOX}/.github/config/image/"
  cp "${REPO_ROOT}/.github/config/image/sglang-sagemaker.yml" "${SANDBOX}/.github/config/image/"
  cp "${REPO_ROOT}/docker/vllm/Dockerfile" "${SANDBOX}/docker/vllm/"
  cp "${REPO_ROOT}/docker/sglang/Dockerfile" "${SANDBOX}/docker/sglang/"

  # Copy scripts into sandbox
  cp "${AUTOCURRENCY_DIR}/utils.sh" "${SANDBOX}/scripts/autocurrency/"
  cp "${AUTOCURRENCY_DIR}/update-configs.sh" "${SANDBOX}/scripts/autocurrency/"
  cp "${AUTOCURRENCY_DIR}/detect-versions.sh" "${SANDBOX}/scripts/autocurrency/"

  # Patch sandbox copies for macOS compatibility (production scripts are
  # correct for Linux CI — these patches are test-only).
  if [[ "${IS_MACOS}" == "true" ]]; then
    _patch_for_macos
  fi

  # Rewrite relative paths in tracker.yml to absolute sandbox paths so
  # detect-versions functions work regardless of CWD.
  _absolutize_tracker_paths
}

_patch_for_macos() {
  local f

  # 1. update-configs.sh uses GNU sed -i (no backup ext). macOS needs sed -i ''.
  #    Replace: sed -i "s|  →  sed -i '' "s|
  f="${SANDBOX}/scripts/autocurrency/update-configs.sh"
  python3 -c "
import pathlib, re
p = pathlib.Path('${f}')
t = p.read_text()
t = t.replace('sed -i \"s|', 'sed -i \"\" \"s|')
p.write_text(t)
"

  # 2. detect-versions.sh uses grep -oP (Perl regex, Linux-only).
  #    Replace the whole grep -oP line with a portable grep+sed pipeline.
  f="${SANDBOX}/scripts/autocurrency/detect-versions.sh"
  python3 -c "
import pathlib
p = pathlib.Path('${f}')
t = p.read_text()
old = '''base_image=\$(grep -oP '^ARG BASE_IMAGE=\\\K.*' \"\${dockerfile_path}\" | head -1)'''
new = '''base_image=\$(grep '^ARG BASE_IMAGE=' \"\${dockerfile_path}\" | head -1 | sed 's/^ARG BASE_IMAGE=//')'''
t = t.replace(old, new)
p.write_text(t)
"
}

_absolutize_tracker_paths() {
  # The tracker YAML has relative paths like ".github/config/image/vllm-ec2.yml"
  # and "docker/vllm/Dockerfile". Prefix them with the sandbox root so
  # detect-versions functions work from any CWD.
  local tracker="${SANDBOX}/.github/config/autocurrency-tracker.yml"
  python3 -c "
import pathlib, yaml, sys

tracker = pathlib.Path('${tracker}')
data = yaml.safe_load(tracker.read_text())
sandbox = '${SANDBOX}'

for fw_key, fw in data.get('frameworks', {}).items():
    for entry in fw.get('config_files', []):
        if 'path' in entry and not entry['path'].startswith('/'):
            entry['path'] = f'{sandbox}/{entry[\"path\"]}'
    for entry in fw.get('dockerfiles', []):
        if 'path' in entry and not entry['path'].startswith('/'):
            entry['path'] = f'{sandbox}/{entry[\"path\"]}'

# Write back preserving structure
with open(tracker, 'w') as f:
    yaml.dump(data, f, default_flow_style=False, sort_keys=False)
"
}

teardown_sandbox() {
  [[ -n "${SANDBOX:-}" && -d "${SANDBOX}" ]] && rm -rf "${SANDBOX}"
}

###############################################################################
# Test runner
###############################################################################

run_test() {
  local test_name="$1"
  local test_func="$2"

  if [[ -n "${FILTER}" && "${test_name}" != *"${FILTER}"* ]]; then
    return 0
  fi

  TESTS_RUN=$((TESTS_RUN + 1))
  echo -n "  ${test_name} ... "

  setup_sandbox

  local output exit_code=0
  output=$("${test_func}" 2>&1) || exit_code=$?

  teardown_sandbox

  if [[ ${exit_code} -eq 0 ]]; then
    TESTS_PASSED=$((TESTS_PASSED + 1))
    echo -e "${GREEN}PASS${NC}"
  else
    TESTS_FAILED=$((TESTS_FAILED + 1))
    echo -e "${RED}FAIL${NC}"
    echo "${output}" | sed 's/^/    /'
  fi
}

assert_eq() {
  local expected="$1" actual="$2" msg="${3:-}"
  if [[ "${expected}" != "${actual}" ]]; then
    echo "ASSERTION FAILED${msg:+: ${msg}}"
    echo "  expected: '${expected}'"
    echo "  actual:   '${actual}'"
    return 1
  fi
}

assert_contains() {
  local haystack="$1" needle="$2" msg="${3:-}"
  if [[ "${haystack}" != *"${needle}"* ]]; then
    echo "ASSERTION FAILED${msg:+: ${msg}}"
    echo "  expected to contain: '${needle}'"
    echo "  actual: '${haystack}'"
    return 1
  fi
}

assert_file_field() {
  local file="$1"
  local field_path="$2"
  local expected="$3"
  local msg="${4:-${field_path} in $(basename "${file}")}"
  local actual
  actual=$(yq eval "${field_path}" "${file}")
  assert_eq "${expected}" "${actual}" "${msg}"
}

###############################################################################
# Docker mock — returns configurable version strings without real containers
###############################################################################

create_docker_mock() {
  local cuda_version="${1:-12.9}"
  local python_version="${2:-3.12}"
  local os_id="${3:-ubuntu}"
  local os_version_id="${4:-22.04}"

  cat > "${SANDBOX}/mock-bin/docker" <<'OUTER'
#!/usr/bin/env bash
# Mock docker — replaces the entire container execution.
# The real scripts run commands INSIDE the container (nvcc, python3, etc.)
# and pipe through grep/sed. Since we intercept `docker run` entirely,
# we return the final extracted value that the in-container pipeline
# would produce.

CUDA_VER="__CUDA__"
PYTHON_VER="__PYTHON__"
OS_STR="__OS_ID____OS_VER__"

if [[ "$1" == "pull" ]]; then
  echo "Mock: pulling $2"
  exit 0
fi

if [[ "$1" == "run" ]]; then
  all_args="$*"

  # nvcc --version | grep -oP 'V\K\d+\.\d+' → just "12.9"
  if echo "${all_args}" | grep -q "nvcc"; then
    [[ -n "${CUDA_VER}" ]] && echo "${CUDA_VER}"
    exit 0
  fi

  # python3 -c "import sys; print(f'{major}.{minor}')" → "3.12"
  if echo "${all_args}" | grep -q "version_info"; then
    echo "${PYTHON_VER}"
    exit 0
  fi

  # source /etc/os-release && echo "${ID}${VERSION_ID}" → "ubuntu22.04"
  if echo "${all_args}" | grep -q "os-release"; then
    echo "${OS_STR}"
    exit 0
  fi

  exit 0
fi

exit 0
OUTER

  # Substitute placeholders with actual values
  portable_sed_i "s|__CUDA__|${cuda_version}|g" "${SANDBOX}/mock-bin/docker"
  portable_sed_i "s|__PYTHON__|${python_version}|g" "${SANDBOX}/mock-bin/docker"
  portable_sed_i "s|__OS_ID__|${os_id}|g" "${SANDBOX}/mock-bin/docker"
  portable_sed_i "s|__OS_VER__|${os_version_id}|g" "${SANDBOX}/mock-bin/docker"

  chmod +x "${SANDBOX}/mock-bin/docker"
}

###############################################################################
# utils.sh tests
###############################################################################

test_strip_tag_prefix_default() {
  source "${SANDBOX}/scripts/autocurrency/utils.sh"
  assert_eq "0.17.0" "$(strip_tag_prefix "v0.17.0")" "strip default v prefix"
}

test_strip_tag_prefix_custom() {
  source "${SANDBOX}/scripts/autocurrency/utils.sh"
  assert_eq "1.2.3" "$(strip_tag_prefix "release-1.2.3" "release-")" "strip custom prefix"
}

test_strip_tag_prefix_no_match() {
  source "${SANDBOX}/scripts/autocurrency/utils.sh"
  assert_eq "0.17.0" "$(strip_tag_prefix "0.17.0" "v")" "no prefix to strip"
}

test_is_rc_version_true() {
  source "${SANDBOX}/scripts/autocurrency/utils.sh"
  is_rc_version "0.17.0rc1"   || { echo "0.17.0rc1 should be RC"; return 1; }
  is_rc_version "0.17.0-rc.2" || { echo "0.17.0-rc.2 should be RC"; return 1; }
  is_rc_version "0.17.0RC1"   || { echo "0.17.0RC1 should be RC"; return 1; }
}

test_is_rc_version_false() {
  source "${SANDBOX}/scripts/autocurrency/utils.sh"
  ! is_rc_version "0.17.0"       || { echo "0.17.0 should NOT be RC"; return 1; }
  ! is_rc_version "0.17.0.post1" || { echo "0.17.0.post1 should NOT be RC"; return 1; }
}

test_is_newer_version_true() {
  source "${SANDBOX}/scripts/autocurrency/utils.sh"
  is_newer_version "0.17.0" "0.16.0"  || { echo "0.17.0 > 0.16.0"; return 1; }
  is_newer_version "1.0.0"  "0.99.99" || { echo "1.0.0 > 0.99.99"; return 1; }
  is_newer_version "0.5.10" "0.5.9"   || { echo "0.5.10 > 0.5.9"; return 1; }
}

test_is_newer_version_false() {
  source "${SANDBOX}/scripts/autocurrency/utils.sh"
  ! is_newer_version "0.16.0" "0.16.0" || { echo "equal should be false"; return 1; }
  ! is_newer_version "0.9.0"  "0.16.0" || { echo "0.9.0 < 0.16.0"; return 1; }
  ! is_newer_version "0.5.8"  "0.5.9"  || { echo "0.5.8 < 0.5.9"; return 1; }
}

test_is_newer_version_two_segment() {
  source "${SANDBOX}/scripts/autocurrency/utils.sh"
  is_newer_version "0.18" "0.17.1"   || { echo "0.18 > 0.17.1 (padded)"; return 1; }
  ! is_newer_version "0.17" "0.17.1" || { echo "0.17.0 < 0.17.1"; return 1; }
}

test_get_current_version() {
  source "${SANDBOX}/scripts/autocurrency/utils.sh"
  assert_eq "0.17.1" "$(get_current_version "${SANDBOX}/.github/config/image/vllm-ec2.yml")" "vllm current version"
}

test_render_prod_image() {
  source "${SANDBOX}/scripts/autocurrency/utils.sh"
  assert_eq "vllm:0.18-gpu-py312-ec2" "$(render_prod_image "vllm:{major}.{minor}-gpu-py312-ec2" "0.18.0")"
}

test_render_prod_image_patch() {
  source "${SANDBOX}/scripts/autocurrency/utils.sh"
  assert_eq "img:2.9.1-tag" "$(render_prod_image "img:{major}.{minor}.{patch}-tag" "2.9.1")"
}

###############################################################################
# update-configs.sh tests
###############################################################################

test_update_config_files_version_bump() {
  source "${SANDBOX}/scripts/autocurrency/utils.sh"
  source "${SANDBOX}/scripts/autocurrency/update-configs.sh"

  local current_ver
  current_ver=$(get_current_version "${SANDBOX}/.github/config/image/vllm-ec2.yml")
  local new_ver
  new_ver=$(_bump_patch_version "${current_ver}")

  local config_json='[
    {"path": "'"${SANDBOX}"'/.github/config/image/vllm-ec2.yml", "prod_image_template": "vllm:{major}.{minor}-gpu-py312-ec2"},
    {"path": "'"${SANDBOX}"'/.github/config/image/vllm-sagemaker.yml", "prod_image_template": "vllm:{major}.{minor}-gpu-py312"}
  ]'

  local expected_prod_image
  expected_prod_image=$(render_prod_image "vllm:{major}.{minor}-gpu-py312-ec2" "${new_ver}")
  local expected_prod_image_sm
  expected_prod_image_sm=$(render_prod_image "vllm:{major}.{minor}-gpu-py312" "${new_ver}")

  local updated
  updated=$(update_config_files "vllm" "${new_ver}" "${config_json}")

  assert_contains "${updated}" "vllm-ec2.yml"
  assert_contains "${updated}" "vllm-sagemaker.yml"
  assert_file_field "${SANDBOX}/.github/config/image/vllm-ec2.yml" ".common.framework_version" "${new_ver}"
  assert_file_field "${SANDBOX}/.github/config/image/vllm-ec2.yml" ".common.prod_image" "${expected_prod_image}"
  assert_file_field "${SANDBOX}/.github/config/image/vllm-sagemaker.yml" ".common.framework_version" "${new_ver}"
  assert_file_field "${SANDBOX}/.github/config/image/vllm-sagemaker.yml" ".common.prod_image" "${expected_prod_image_sm}"
}

test_update_config_files_sglang() {
  source "${SANDBOX}/scripts/autocurrency/utils.sh"
  source "${SANDBOX}/scripts/autocurrency/update-configs.sh"

  local current_ver
  current_ver=$(get_current_version "${SANDBOX}/.github/config/image/sglang-ec2.yml")
  local new_ver
  new_ver=$(_bump_patch_version "${current_ver}")

  local config_json='[
    {"path": "'"${SANDBOX}"'/.github/config/image/sglang-ec2.yml", "prod_image_template": "sglang:{major}.{minor}-gpu-py312-ec2"},
    {"path": "'"${SANDBOX}"'/.github/config/image/sglang-sagemaker.yml", "prod_image_template": "sglang:{major}.{minor}-gpu-py312"}
  ]'

  local expected_prod_image
  expected_prod_image=$(render_prod_image "sglang:{major}.{minor}-gpu-py312-ec2" "${new_ver}")

  update_config_files "sglang" "${new_ver}" "${config_json}" > /dev/null

  assert_file_field "${SANDBOX}/.github/config/image/sglang-ec2.yml" ".common.framework_version" "${new_ver}"
  assert_file_field "${SANDBOX}/.github/config/image/sglang-ec2.yml" ".common.prod_image" "${expected_prod_image}"
}

test_update_dockerfiles() {
  source "${SANDBOX}/scripts/autocurrency/utils.sh"
  source "${SANDBOX}/scripts/autocurrency/update-configs.sh"

  local current_ver
  current_ver=$(get_current_version "${SANDBOX}/.github/config/image/vllm-ec2.yml")
  local new_ver
  new_ver=$(_bump_patch_version "${current_ver}")

  local dockerfile_json='[{"path": "'"${SANDBOX}"'/docker/vllm/Dockerfile", "base_image_template": "vllm/vllm-openai:v{version}"}]'

  local updated
  updated=$(update_dockerfiles "${new_ver}" "${dockerfile_json}")

  assert_contains "${updated}" "Dockerfile"
  assert_eq "vllm/vllm-openai:v${new_ver}" \
    "$(get_base_image_from_dockerfile "${SANDBOX}/docker/vllm/Dockerfile")" \
    "Dockerfile BASE_IMAGE"
}

###############################################################################
# detect-versions.sh tests (with docker mock)
###############################################################################

test_detect_versions_updates_cuda() {
  yq eval -i '.common.cuda_version = "cu124"' "${SANDBOX}/.github/config/image/vllm-ec2.yml"
  yq eval -i '.common.cuda_version = "cu124"' "${SANDBOX}/.github/config/image/vllm-sagemaker.yml"

  create_docker_mock "12.9" "3.12" "ubuntu" "22.04"
  export PATH="${SANDBOX}/mock-bin:${PATH}"
  export TRACKER_FILE="${SANDBOX}/.github/config/autocurrency-tracker.yml"

  source "${SANDBOX}/scripts/autocurrency/utils.sh"
  source "${SANDBOX}/scripts/autocurrency/detect-versions.sh"

  # Run from sandbox so relative paths in tracker.yml resolve correctly
  detect_and_update_versions "vllm"

  assert_file_field "${SANDBOX}/.github/config/image/vllm-ec2.yml" ".common.cuda_version" "cu129"
  assert_file_field "${SANDBOX}/.github/config/image/vllm-sagemaker.yml" ".common.cuda_version" "cu129"
}

test_detect_versions_updates_python() {
  yq eval -i '.common.python_version = "py311"' "${SANDBOX}/.github/config/image/sglang-ec2.yml"
  yq eval -i '.common.python_version = "py311"' "${SANDBOX}/.github/config/image/sglang-sagemaker.yml"

  create_docker_mock "12.9" "3.12" "ubuntu" "24.04"
  export PATH="${SANDBOX}/mock-bin:${PATH}"
  export TRACKER_FILE="${SANDBOX}/.github/config/autocurrency-tracker.yml"

  source "${SANDBOX}/scripts/autocurrency/utils.sh"
  source "${SANDBOX}/scripts/autocurrency/detect-versions.sh"

  detect_and_update_versions "sglang"

  assert_file_field "${SANDBOX}/.github/config/image/sglang-ec2.yml" ".common.python_version" "py312"
  assert_file_field "${SANDBOX}/.github/config/image/sglang-sagemaker.yml" ".common.python_version" "py312"
}

test_detect_versions_updates_os() {
  yq eval -i '.common.os_version = "ubuntu20.04"' "${SANDBOX}/.github/config/image/vllm-ec2.yml"
  yq eval -i '.common.os_version = "ubuntu20.04"' "${SANDBOX}/.github/config/image/vllm-sagemaker.yml"

  create_docker_mock "12.9" "3.12" "ubuntu" "22.04"
  export PATH="${SANDBOX}/mock-bin:${PATH}"
  export TRACKER_FILE="${SANDBOX}/.github/config/autocurrency-tracker.yml"

  source "${SANDBOX}/scripts/autocurrency/utils.sh"
  source "${SANDBOX}/scripts/autocurrency/detect-versions.sh"

  detect_and_update_versions "vllm"

  assert_file_field "${SANDBOX}/.github/config/image/vllm-ec2.yml" ".common.os_version" "ubuntu22.04"
  assert_file_field "${SANDBOX}/.github/config/image/vllm-sagemaker.yml" ".common.os_version" "ubuntu22.04"
}

test_detect_versions_no_changes_when_current() {
  create_docker_mock "12.9" "3.12" "ubuntu" "22.04"
  export PATH="${SANDBOX}/mock-bin:${PATH}"
  export TRACKER_FILE="${SANDBOX}/.github/config/autocurrency-tracker.yml"

  source "${SANDBOX}/scripts/autocurrency/utils.sh"
  source "${SANDBOX}/scripts/autocurrency/detect-versions.sh"

  local exit_code=0
  detect_and_update_versions "vllm" || exit_code=$?
  assert_eq "1" "${exit_code}" "should return 1 when no changes needed"
}

test_detect_versions_multiple_field_update() {
  yq eval -i '.common.cuda_version = "cu118"'     "${SANDBOX}/.github/config/image/vllm-ec2.yml"
  yq eval -i '.common.python_version = "py310"'   "${SANDBOX}/.github/config/image/vllm-ec2.yml"
  yq eval -i '.common.os_version = "ubuntu20.04"' "${SANDBOX}/.github/config/image/vllm-ec2.yml"
  yq eval -i '.common.cuda_version = "cu118"'     "${SANDBOX}/.github/config/image/vllm-sagemaker.yml"
  yq eval -i '.common.python_version = "py310"'   "${SANDBOX}/.github/config/image/vllm-sagemaker.yml"
  yq eval -i '.common.os_version = "ubuntu20.04"' "${SANDBOX}/.github/config/image/vllm-sagemaker.yml"

  create_docker_mock "12.9" "3.12" "ubuntu" "22.04"
  export PATH="${SANDBOX}/mock-bin:${PATH}"
  export TRACKER_FILE="${SANDBOX}/.github/config/autocurrency-tracker.yml"

  source "${SANDBOX}/scripts/autocurrency/utils.sh"
  source "${SANDBOX}/scripts/autocurrency/detect-versions.sh"

  detect_and_update_versions "vllm"

  assert_file_field "${SANDBOX}/.github/config/image/vllm-ec2.yml" ".common.cuda_version" "cu129"
  assert_file_field "${SANDBOX}/.github/config/image/vllm-ec2.yml" ".common.python_version" "py312"
  assert_file_field "${SANDBOX}/.github/config/image/vllm-ec2.yml" ".common.os_version" "ubuntu22.04"
}

test_detect_versions_skips_disabled_framework() {
  yq eval -i '.frameworks.vllm.version_detection = false' "${SANDBOX}/.github/config/autocurrency-tracker.yml"

  create_docker_mock "12.9" "3.12" "ubuntu" "22.04"
  export PATH="${SANDBOX}/mock-bin:${PATH}"
  export TRACKER_FILE="${SANDBOX}/.github/config/autocurrency-tracker.yml"

  source "${SANDBOX}/scripts/autocurrency/utils.sh"
  source "${SANDBOX}/scripts/autocurrency/detect-versions.sh"

  local exit_code=0
  detect_and_update_versions "vllm" || exit_code=$?
  assert_eq "1" "${exit_code}" "disabled framework should return 1"
}

###############################################################################
# extract_framework_from_branch tests
###############################################################################

test_extract_framework_vllm() {
  source "${SANDBOX}/scripts/autocurrency/utils.sh"
  source "${SANDBOX}/scripts/autocurrency/detect-versions.sh"
  assert_eq "vllm" "$(extract_framework_from_branch "auto-update/vllm-0.18.0")"
}

test_extract_framework_sglang() {
  source "${SANDBOX}/scripts/autocurrency/utils.sh"
  source "${SANDBOX}/scripts/autocurrency/detect-versions.sh"
  assert_eq "sglang" "$(extract_framework_from_branch "auto-update/sglang-0.6.0")"
}

test_extract_framework_invalid_branch() {
  source "${SANDBOX}/scripts/autocurrency/utils.sh"
  source "${SANDBOX}/scripts/autocurrency/detect-versions.sh"
  local exit_code=0
  extract_framework_from_branch "feature/something" 2>/dev/null || exit_code=$?
  assert_eq "1" "${exit_code}" "non auto-update branch should fail"
}

###############################################################################
# End-to-end: simulate "older current version" scenario
###############################################################################

test_e2e_mock_older_current_version() {
  source "${SANDBOX}/scripts/autocurrency/utils.sh"

  local real_ver
  real_ver=$(get_current_version "${SANDBOX}/.github/config/image/vllm-ec2.yml")
  local older_ver
  older_ver=$(_decrement_minor_version "${real_ver}")

  yq eval -i ".common.framework_version = \"${older_ver}\"" "${SANDBOX}/.github/config/image/vllm-ec2.yml"

  local current
  current=$(get_current_version "${SANDBOX}/.github/config/image/vllm-ec2.yml")
  assert_eq "${older_ver}" "${current}" "mocked current version"

  is_newer_version "${real_ver}" "${current}" || {
    echo "${real_ver} should be newer than ${older_ver}"
    return 1
  }
}

test_e2e_full_update_pipeline() {
  source "${SANDBOX}/scripts/autocurrency/utils.sh"
  source "${SANDBOX}/scripts/autocurrency/update-configs.sh"

  local real_ver
  real_ver=$(get_current_version "${SANDBOX}/.github/config/image/vllm-ec2.yml")
  local older_ver
  older_ver=$(_decrement_minor_version "${real_ver}")
  local new_ver
  new_ver=$(_bump_patch_version "${real_ver}")

  yq eval -i ".common.framework_version = \"${older_ver}\"" "${SANDBOX}/.github/config/image/vllm-ec2.yml"
  yq eval -i ".common.framework_version = \"${older_ver}\"" "${SANDBOX}/.github/config/image/vllm-sagemaker.yml"

  local expected_prod_image
  expected_prod_image=$(render_prod_image "vllm:{major}.{minor}-gpu-py312-ec2" "${new_ver}")

  local config_json='[
    {"path": "'"${SANDBOX}"'/.github/config/image/vllm-ec2.yml", "prod_image_template": "vllm:{major}.{minor}-gpu-py312-ec2"},
    {"path": "'"${SANDBOX}"'/.github/config/image/vllm-sagemaker.yml", "prod_image_template": "vllm:{major}.{minor}-gpu-py312"}
  ]'
  update_config_files "vllm" "${new_ver}" "${config_json}" > /dev/null

  local dockerfile_json='[{"path": "'"${SANDBOX}"'/docker/vllm/Dockerfile", "base_image_template": "vllm/vllm-openai:v{version}"}]'
  update_dockerfiles "${new_ver}" "${dockerfile_json}" > /dev/null

  assert_file_field "${SANDBOX}/.github/config/image/vllm-ec2.yml" ".common.framework_version" "${new_ver}"
  assert_file_field "${SANDBOX}/.github/config/image/vllm-ec2.yml" ".common.prod_image" "${expected_prod_image}"
  assert_file_field "${SANDBOX}/.github/config/image/vllm-sagemaker.yml" ".common.framework_version" "${new_ver}"

  assert_eq "vllm/vllm-openai:v${new_ver}" \
    "$(get_base_image_from_dockerfile "${SANDBOX}/docker/vllm/Dockerfile")" \
    "Dockerfile BASE_IMAGE after full pipeline"
}

###############################################################################
# docs-pr.sh tests
###############################################################################

setup_docs_pr_sandbox() {
  # Source docs-pr.sh helpers (guarded by BASH_SOURCE check, so main won't run)
  source "${SANDBOX}/scripts/autocurrency/docs-pr.sh"
}

# --- get_display_name ---

test_docs_pr_display_name_vllm() {
  cp "${AUTOCURRENCY_DIR}/docs-pr.sh" "${SANDBOX}/scripts/autocurrency/"
  setup_docs_pr_sandbox
  assert_eq "vLLM" "$(get_display_name "vllm")"
}

test_docs_pr_display_name_sglang() {
  cp "${AUTOCURRENCY_DIR}/docs-pr.sh" "${SANDBOX}/scripts/autocurrency/"
  setup_docs_pr_sandbox
  assert_eq "SGLang" "$(get_display_name "sglang")"
}

test_docs_pr_display_name_unknown() {
  cp "${AUTOCURRENCY_DIR}/docs-pr.sh" "${SANDBOX}/scripts/autocurrency/"
  setup_docs_pr_sandbox
  assert_eq "myframework" "$(get_display_name "myframework")"
}

# --- parse_major_minor ---

test_docs_pr_parse_major_minor_three_part() {
  cp "${AUTOCURRENCY_DIR}/docs-pr.sh" "${SANDBOX}/scripts/autocurrency/"
  setup_docs_pr_sandbox
  assert_eq "0.17" "$(parse_major_minor "0.17.1")"
}

test_docs_pr_parse_major_minor_two_part() {
  cp "${AUTOCURRENCY_DIR}/docs-pr.sh" "${SANDBOX}/scripts/autocurrency/"
  setup_docs_pr_sandbox
  assert_eq "0.5" "$(parse_major_minor "0.5")"
}

test_docs_pr_parse_major_minor_four_part() {
  cp "${AUTOCURRENCY_DIR}/docs-pr.sh" "${SANDBOX}/scripts/autocurrency/"
  setup_docs_pr_sandbox
  assert_eq "2.9" "$(parse_major_minor "2.9.1.post1")"
}

# --- generate_tags ---

test_docs_pr_tags_ec2() {
  cp "${AUTOCURRENCY_DIR}/docs-pr.sh" "${SANDBOX}/scripts/autocurrency/"
  setup_docs_pr_sandbox
  local tags
  tags=$(generate_tags "0.17.1" "gpu" "py312" "cu129" "ubuntu22.04" "ec2")
  local tag1 tag2 tag3 tag4
  tag1=$(echo "$tags" | sed -n '1p')
  tag2=$(echo "$tags" | sed -n '2p')
  tag3=$(echo "$tags" | sed -n '3p')
  tag4=$(echo "$tags" | sed -n '4p')
  assert_eq "0.17.1-gpu-py312-cu129-ubuntu22.04-ec2" "$tag1" "ec2 tag1"
  assert_eq "0.17-gpu-py312-cu129-ubuntu22.04-ec2-v1" "$tag2" "ec2 tag2"
  assert_eq "0.17.1-gpu-py312-ec2" "$tag3" "ec2 tag3"
  assert_eq "0.17-gpu-py312-ec2" "$tag4" "ec2 tag4"
}

test_docs_pr_tags_sagemaker() {
  cp "${AUTOCURRENCY_DIR}/docs-pr.sh" "${SANDBOX}/scripts/autocurrency/"
  setup_docs_pr_sandbox
  local tags
  tags=$(generate_tags "0.17.1" "gpu" "py312" "cu129" "ubuntu22.04" "sagemaker")
  local tag1 tag2 tag3 tag4
  tag1=$(echo "$tags" | sed -n '1p')
  tag2=$(echo "$tags" | sed -n '2p')
  tag3=$(echo "$tags" | sed -n '3p')
  tag4=$(echo "$tags" | sed -n '4p')
  assert_eq "0.17.1-gpu-py312-cu129-ubuntu22.04-sagemaker" "$tag1" "sm tag1"
  assert_eq "0.17-gpu-py312-cu129-ubuntu22.04-sagemaker-v1" "$tag2" "sm tag2"
  assert_eq "0.17.1-gpu-py312" "$tag3" "sm tag3"
  assert_eq "0.17-gpu-py312" "$tag4" "sm tag4"
}

test_docs_pr_tags_sglang_sagemaker() {
  cp "${AUTOCURRENCY_DIR}/docs-pr.sh" "${SANDBOX}/scripts/autocurrency/"
  setup_docs_pr_sandbox
  local tags
  tags=$(generate_tags "0.5.9" "gpu" "py312" "cu129" "ubuntu24.04" "sagemaker")
  local tag1 tag2 tag3 tag4
  tag1=$(echo "$tags" | sed -n '1p')
  tag2=$(echo "$tags" | sed -n '2p')
  tag3=$(echo "$tags" | sed -n '3p')
  tag4=$(echo "$tags" | sed -n '4p')
  assert_eq "0.5.9-gpu-py312-cu129-ubuntu24.04-sagemaker" "$tag1" "sglang sm tag1"
  assert_eq "0.5-gpu-py312-cu129-ubuntu24.04-sagemaker-v1" "$tag2" "sglang sm tag2"
  assert_eq "0.5.9-gpu-py312" "$tag3" "sglang sm tag3"
  assert_eq "0.5-gpu-py312" "$tag4" "sglang sm tag4"
}

test_docs_pr_tags_count_always_four() {
  cp "${AUTOCURRENCY_DIR}/docs-pr.sh" "${SANDBOX}/scripts/autocurrency/"
  setup_docs_pr_sandbox
  for plat in ec2 sagemaker; do
    local count
    count=$(generate_tags "1.2.3" "gpu" "py312" "cu129" "ubuntu22.04" "$plat" | wc -l | tr -d ' ')
    assert_eq "4" "$count" "tag count for $plat"
  done
}

test_docs_pr_tags_unknown_platform_empty() {
  cp "${AUTOCURRENCY_DIR}/docs-pr.sh" "${SANDBOX}/scripts/autocurrency/"
  setup_docs_pr_sandbox
  local count
  count=$(generate_tags "1.0.0" "gpu" "py312" "cu129" "ubuntu22.04" "unknown" | wc -l | tr -d ' ')
  # wc -l on empty string still returns 0 on some systems, or 1 with trailing newline
  # Use -c to check for empty output instead
  local output
  output=$(generate_tags "1.0.0" "gpu" "py312" "cu129" "ubuntu22.04" "unknown")
  assert_eq "" "$output" "unknown platform should produce empty output"
}

# --- generate_announcement ---

test_docs_pr_announcement_ec2() {
  cp "${AUTOCURRENCY_DIR}/docs-pr.sh" "${SANDBOX}/scripts/autocurrency/"
  setup_docs_pr_sandbox
  assert_eq "Introduced vLLM 0.17.1 containers for EC2, ECS, EKS" \
    "$(generate_announcement "vllm" "0.17.1" "ec2")"
}

test_docs_pr_announcement_sagemaker() {
  cp "${AUTOCURRENCY_DIR}/docs-pr.sh" "${SANDBOX}/scripts/autocurrency/"
  setup_docs_pr_sandbox
  assert_eq "Introduced vLLM 0.17.1 containers for SageMaker" \
    "$(generate_announcement "vllm" "0.17.1" "sagemaker")"
}

test_docs_pr_announcement_sglang() {
  cp "${AUTOCURRENCY_DIR}/docs-pr.sh" "${SANDBOX}/scripts/autocurrency/"
  setup_docs_pr_sandbox
  assert_eq "Introduced SGLang 0.5.9 containers for SageMaker" \
    "$(generate_announcement "sglang" "0.5.9" "sagemaker")"
}

# --- generate_branch_name ---

test_docs_pr_branch_name() {
  cp "${AUTOCURRENCY_DIR}/docs-pr.sh" "${SANDBOX}/scripts/autocurrency/"
  setup_docs_pr_sandbox
  assert_eq "docs/auto-update-vllm-0.17.1-ec2" \
    "$(generate_branch_name "vllm" "0.17.1" "ec2")"
}

test_docs_pr_branch_name_sagemaker() {
  cp "${AUTOCURRENCY_DIR}/docs-pr.sh" "${SANDBOX}/scripts/autocurrency/"
  setup_docs_pr_sandbox
  assert_eq "docs/auto-update-sglang-0.5.9-sagemaker" \
    "$(generate_branch_name "sglang" "0.5.9" "sagemaker")"
}

# --- generate_pr_title ---

test_docs_pr_title_ec2() {
  cp "${AUTOCURRENCY_DIR}/docs-pr.sh" "${SANDBOX}/scripts/autocurrency/"
  setup_docs_pr_sandbox
  assert_eq "docs: Add vLLM 0.17.1 EC2 image data" \
    "$(generate_pr_title "vllm" "0.17.1" "ec2")"
}

test_docs_pr_title_sagemaker() {
  cp "${AUTOCURRENCY_DIR}/docs-pr.sh" "${SANDBOX}/scripts/autocurrency/"
  setup_docs_pr_sandbox
  assert_eq "docs: Add vLLM 0.17.1 SAGEMAKER image data" \
    "$(generate_pr_title "vllm" "0.17.1" "sagemaker")"
}

test_docs_pr_title_sglang() {
  cp "${AUTOCURRENCY_DIR}/docs-pr.sh" "${SANDBOX}/scripts/autocurrency/"
  setup_docs_pr_sandbox
  assert_eq "docs: Add SGLang 0.5.9 SAGEMAKER image data" \
    "$(generate_pr_title "sglang" "0.5.9" "sagemaker")"
}

###############################################################################
# Run all tests
###############################################################################

echo ""
echo "============================================================"
echo "  Autocurrency Version Detection — Test Suite"
echo "============================================================"
if [[ "${IS_MACOS}" == "true" ]]; then
  echo "  (running with macOS compatibility patches)"
fi
echo ""

echo "--- utils.sh ---"
run_test "strip_tag_prefix: default v prefix"       test_strip_tag_prefix_default
run_test "strip_tag_prefix: custom prefix"           test_strip_tag_prefix_custom
run_test "strip_tag_prefix: no matching prefix"      test_strip_tag_prefix_no_match
run_test "is_rc_version: detects RC versions"        test_is_rc_version_true
run_test "is_rc_version: rejects non-RC versions"    test_is_rc_version_false
run_test "is_newer_version: newer returns true"      test_is_newer_version_true
run_test "is_newer_version: equal/older returns false" test_is_newer_version_false
run_test "is_newer_version: two-segment versions"    test_is_newer_version_two_segment
run_test "get_current_version: reads from config"    test_get_current_version
run_test "render_prod_image: major.minor template"   test_render_prod_image
run_test "render_prod_image: with patch segment"     test_render_prod_image_patch

echo ""
echo "--- update-configs.sh ---"
run_test "update_config_files: bumps version and prod_image" test_update_config_files_version_bump
run_test "update_config_files: sglang version bump"          test_update_config_files_sglang
run_test "update_dockerfiles: updates BASE_IMAGE arg"        test_update_dockerfiles

echo ""
echo "--- detect-versions.sh ---"
run_test "detect_versions: updates cuda_version"     test_detect_versions_updates_cuda
run_test "detect_versions: updates python_version"   test_detect_versions_updates_python
run_test "detect_versions: updates os_version"       test_detect_versions_updates_os
run_test "detect_versions: no changes when current"  test_detect_versions_no_changes_when_current
run_test "detect_versions: multiple fields at once"  test_detect_versions_multiple_field_update
run_test "detect_versions: skips disabled framework" test_detect_versions_skips_disabled_framework
run_test "extract_framework_from_branch: vllm"       test_extract_framework_vllm
run_test "extract_framework_from_branch: sglang"     test_extract_framework_sglang
run_test "extract_framework_from_branch: invalid"    test_extract_framework_invalid_branch

echo ""
echo "--- end-to-end ---"
run_test "e2e: mock older current version triggers update" test_e2e_mock_older_current_version
run_test "e2e: full update pipeline (config + Dockerfile)" test_e2e_full_update_pipeline

echo ""
echo "--- docs-pr.sh ---"
run_test "get_display_name: vllm → vLLM"                    test_docs_pr_display_name_vllm
run_test "get_display_name: sglang → SGLang"                 test_docs_pr_display_name_sglang
run_test "get_display_name: unknown falls back to raw"       test_docs_pr_display_name_unknown
run_test "parse_major_minor: three-part version"             test_docs_pr_parse_major_minor_three_part
run_test "parse_major_minor: two-part version"               test_docs_pr_parse_major_minor_two_part
run_test "parse_major_minor: four-part version"              test_docs_pr_parse_major_minor_four_part
run_test "generate_tags: ec2 known config"                   test_docs_pr_tags_ec2
run_test "generate_tags: sagemaker known config"             test_docs_pr_tags_sagemaker
run_test "generate_tags: sglang sagemaker known config"      test_docs_pr_tags_sglang_sagemaker
run_test "generate_tags: always produces 4 tags"             test_docs_pr_tags_count_always_four
run_test "generate_tags: unknown platform → empty"           test_docs_pr_tags_unknown_platform_empty
run_test "generate_announcement: ec2"                        test_docs_pr_announcement_ec2
run_test "generate_announcement: sagemaker"                  test_docs_pr_announcement_sagemaker
run_test "generate_announcement: sglang sagemaker"           test_docs_pr_announcement_sglang
run_test "generate_branch_name: vllm ec2"                    test_docs_pr_branch_name
run_test "generate_branch_name: sglang sagemaker"            test_docs_pr_branch_name_sagemaker
run_test "generate_pr_title: vllm ec2"                       test_docs_pr_title_ec2
run_test "generate_pr_title: vllm sagemaker"                 test_docs_pr_title_sagemaker
run_test "generate_pr_title: sglang sagemaker"               test_docs_pr_title_sglang

echo ""
echo "============================================================"
echo -e "  Results: ${TESTS_PASSED}/${TESTS_RUN} passed, ${TESTS_FAILED} failed"
echo "============================================================"

[[ ${TESTS_FAILED} -gt 0 ]] && exit 1
exit 0
