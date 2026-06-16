#!/usr/bin/env bash
# detect-versions.sh — Detects CUDA, Python, and OS versions from upstream
# base Docker images and updates config YAML files if they differ.
#
# Called by the prcheck-detect-versions.yml workflow after an auto-update PR is created.
#
# Environment variables (set by the workflow):
#   GITHUB_TOKEN    — GitHub token for pushing commits
#   PR_BRANCH       — The auto-update PR branch name
#   PR_NUMBER       — The PR number (for comments)
#   TRACKER_FILE    — Path to autocurrency-tracker.yml (optional, defaults below)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils.sh"

TRACKER_FILE="${TRACKER_FILE:-.github/config/autocurrency-tracker.yml}"

###############################################################################
# resolve_base_image(framework)
#   Reads the first Dockerfile for a framework from tracker.yml and extracts
#   the current ARG BASE_IMAGE= value from the actual Dockerfile on disk.
#
#   Returns the full image reference (e.g., "vllm/vllm-openai:v0.18.0")
###############################################################################
resolve_base_image() {
  local framework="${1:?Usage: resolve_base_image FRAMEWORK}"

  local dockerfile_path
  dockerfile_path=$(yq eval ".frameworks.${framework}.dockerfiles[0].path" "${TRACKER_FILE}")

  if [[ -z "${dockerfile_path}" || "${dockerfile_path}" == "null" ]]; then
    echo "Error: no Dockerfile configured for ${framework}" >&2
    return 1
  fi

  if [[ ! -f "${dockerfile_path}" ]]; then
    echo "Error: Dockerfile not found: ${dockerfile_path}" >&2
    return 1
  fi

  local base_image
  base_image=$(grep -oP '^ARG BASE_IMAGE=\K.*' "${dockerfile_path}" | head -1)

  if [[ -z "${base_image}" ]]; then
    echo "Error: ARG BASE_IMAGE= not found in ${dockerfile_path}" >&2
    return 1
  fi

  echo "${base_image}"
}

###############################################################################
# extract_cuda_version(image)
#   Runs nvcc --version inside the container and extracts the CUDA version.
#   Returns the version in config format: "cu129" for CUDA 12.9
#   Returns empty string if nvcc is not available.
###############################################################################
extract_cuda_version() {
  local image="${1:?Usage: extract_cuda_version IMAGE}"

  local cuda_raw
  cuda_raw=$(docker run --rm --entrypoint "" "${image}" \
    bash -c "nvcc --version 2>/dev/null | grep -oP 'V\K\d+\.\d+'" 2>/dev/null) || true

  if [[ -z "${cuda_raw}" ]]; then
    echo ""
    return 0
  fi

  # "12.9" → "cu129"
  echo "cu$(echo "${cuda_raw}" | tr -d '.')"
}

###############################################################################
# extract_python_version(image)
#   Runs python3 inside the container and extracts the Python version.
#   Returns the version in config format: "py312" for Python 3.12
###############################################################################
extract_python_version() {
  local image="${1:?Usage: extract_python_version IMAGE}"

  local python_raw
  python_raw=$(docker run --rm --entrypoint "" "${image}" \
    python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null) || true

  if [[ -z "${python_raw}" ]]; then
    echo ""
    return 0
  fi

  # "3.12" → "py312"
  echo "py$(echo "${python_raw}" | tr -d '.')"
}

###############################################################################
# extract_os_version(image)
#   Reads /etc/os-release inside the container and extracts the OS version.
#   Returns the version in config format: "ubuntu22.04"
###############################################################################
extract_os_version() {
  local image="${1:?Usage: extract_os_version IMAGE}"

  local os_raw
  os_raw=$(docker run --rm --entrypoint "" "${image}" \
    bash -c 'source /etc/os-release && echo "${ID}${VERSION_ID}"' 2>/dev/null) || true

  if [[ -z "${os_raw}" ]]; then
    echo ""
    return 0
  fi

  echo "${os_raw}"
}

###############################################################################
# update_config_versions(framework, detected_cuda, detected_python, detected_os)
#   Compares detected versions against config YAML values and updates if
#   they differ. Returns newline-separated list of updated config file paths.
###############################################################################
update_config_versions() {
  local framework="${1:?Usage: update_config_versions FRAMEWORK CUDA PYTHON OS}"
  local detected_cuda="${2:-}"
  local detected_python="${3:-}"
  local detected_os="${4:-}"

  local updated_files=""
  local config_count
  config_count=$(yq eval ".frameworks.${framework}.config_files | length" "${TRACKER_FILE}")

  for i in $(seq 0 $(( config_count - 1 ))); do
    local config_path
    config_path=$(yq eval ".frameworks.${framework}.config_files[$i].path" "${TRACKER_FILE}")
    local changed=false

    if [[ ! -f "${config_path}" ]]; then
      echo "Warning: config file not found: ${config_path}" >&2
      continue
    fi

    # Update cuda_version if detected and different
    if [[ -n "${detected_cuda}" ]]; then
      local current_cuda
      current_cuda=$(yq eval '.common.cuda_version' "${config_path}")
      if [[ "${current_cuda}" != "${detected_cuda}" ]]; then
        echo "  ${config_path}: cuda_version ${current_cuda} → ${detected_cuda}"
        yq eval -i ".common.cuda_version = \"${detected_cuda}\"" "${config_path}"
        changed=true
      fi
    fi

    # Update python_version if detected and different
    if [[ -n "${detected_python}" ]]; then
      local current_python
      current_python=$(yq eval '.common.python_version' "${config_path}")
      if [[ "${current_python}" != "${detected_python}" ]]; then
        echo "  ${config_path}: python_version ${current_python} → ${detected_python}"
        yq eval -i ".common.python_version = \"${detected_python}\"" "${config_path}"
        changed=true
      fi
    fi

    # Update os_version if detected and different
    if [[ -n "${detected_os}" ]]; then
      local current_os
      current_os=$(yq eval '.common.os_version' "${config_path}")
      if [[ "${current_os}" != "${detected_os}" ]]; then
        echo "  ${config_path}: os_version ${current_os} → ${detected_os}"
        yq eval -i ".common.os_version = \"${detected_os}\"" "${config_path}"
        changed=true
      fi
    fi

    if [[ "${changed}" == "true" ]]; then
      if [[ -n "${updated_files}" ]]; then
        updated_files="${updated_files}"$'\n'"${config_path}"
      else
        updated_files="${config_path}"
      fi
    fi
  done

  echo "${updated_files}"
}

###############################################################################
# detect_and_update_versions(framework)
#   Main entry point for a single framework. Pulls the base image, extracts
#   versions, and updates config files if needed.
#   Returns 0 if changes were made, 1 if no changes needed.
###############################################################################
detect_and_update_versions() {
  local framework="${1:?Usage: detect_and_update_versions FRAMEWORK}"

  # Check if version detection is enabled for this framework
  local agent_enabled
  agent_enabled=$(yq eval ".frameworks.${framework}.version_detection // false" "${TRACKER_FILE}")
  if [[ "${agent_enabled}" != "true" ]]; then
    echo "${framework}: Version detection not enabled. Skipping."
    return 1
  fi

  # Resolve the base image from the Dockerfile
  local base_image
  base_image=$(resolve_base_image "${framework}")
  echo "${framework}: Pulling base image: ${base_image}"
  docker pull "${base_image}"

  # Extract versions
  echo "${framework}: Extracting versions from ${base_image}..."
  local detected_cuda detected_python detected_os
  detected_cuda=$(extract_cuda_version "${base_image}")
  detected_python=$(extract_python_version "${base_image}")
  detected_os=$(extract_os_version "${base_image}")

  echo "${framework}: Detected — CUDA: ${detected_cuda:-n/a}, Python: ${detected_python:-n/a}, OS: ${detected_os:-n/a}"

  # Fail if critical versions could not be extracted
  if [[ -z "${detected_python}" ]]; then
    echo "::error::${framework}: Failed to extract Python version from ${base_image}"
    return 1
  fi

  if [[ -z "${detected_os}" ]]; then
    echo "::error::${framework}: Failed to extract OS version from ${base_image}"
    return 1
  fi

  if [[ -z "${detected_cuda}" ]]; then
    echo "::warning::${framework}: CUDA version not detected (expected for CPU images)"
  fi

  # Update config files if versions differ
  local updated_files
  updated_files=$(update_config_versions "${framework}" "${detected_cuda}" "${detected_python}" "${detected_os}")

  if [[ -z "${updated_files}" ]]; then
    echo "${framework}: All versions are up to date. No changes needed."
    return 1
  fi

  echo "${framework}: Updated config files:"
  echo "${updated_files}"
  return 0
}

###############################################################################
# extract_framework_from_branch(branch_name)
#   Extracts the framework name from an auto-update branch name.
#   Branch format: auto-update/<framework>-<version>
#   Returns the framework key (e.g., "vllm", "sglang")
###############################################################################
extract_framework_from_branch() {
  local branch="${1:?Usage: extract_framework_from_branch BRANCH_NAME}"

  # Strip "auto-update/" prefix
  local remainder="${branch#auto-update/}"

  if [[ "${remainder}" == "${branch}" ]]; then
    echo "Error: branch '${branch}' does not match auto-update/<framework>-<version>" >&2
    return 1
  fi

  # Framework is everything before the last -<version> segment
  # Version starts with a digit, so find the last "-<digit>" boundary
  local framework
  framework=$(echo "${remainder}" | sed -E 's/-[0-9][0-9.]*$//')

  if [[ -z "${framework}" ]]; then
    echo "Error: could not extract framework from branch '${branch}'" >&2
    return 1
  fi

  echo "${framework}"
}

###############################################################################
# Main — detect versions for the framework from the PR branch
###############################################################################
main() {
  if [[ ! -f "${TRACKER_FILE}" ]]; then
    echo "::error::Tracker registry not found: ${TRACKER_FILE}"
    exit 1
  fi

  # Extract framework from PR branch name
  local pr_branch="${PR_BRANCH:-}"
  if [[ -z "${pr_branch}" ]]; then
    echo "::error::PR_BRANCH environment variable not set"
    exit 1
  fi

  local framework
  framework=$(extract_framework_from_branch "${pr_branch}")
  echo "Framework from branch: ${framework}"

  # Validate framework exists in tracker
  local exists
  exists=$(yq eval ".frameworks.${framework} // null" "${TRACKER_FILE}")
  if [[ "${exists}" == "null" ]]; then
    echo "::error::Framework '${framework}' not found in ${TRACKER_FILE}"
    exit 1
  fi

  echo ""
  echo "============================================================"
  echo "Version Detection: ${framework}"
  echo "============================================================"

  local any_changes=false

  set +e
  detect_and_update_versions "${framework}" 2>&1
  local exit_code=$?
  set -e

  if [[ ${exit_code} -eq 0 ]]; then
    any_changes=true
  fi

  if [[ "${any_changes}" != "true" ]]; then
    echo ""
    echo "No version changes detected for ${framework}."
    exit 0
  fi

  # Stage and commit all changes
  echo ""
  echo "Committing version updates..."
  git config user.name "github-actions[bot]"
  git config user.email "github-actions[bot]@users.noreply.github.com"
  git add -A
  git commit -m "[Detect-Versions] Update CUDA/Python/OS versions for ${framework}" || {
    echo "No changes to commit."
    exit 0
  }
  git push

  echo "Version detection complete. Changes pushed to PR branch."
}

# Run main only when executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
