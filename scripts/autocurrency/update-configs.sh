#!/usr/bin/env bash
# update-configs.sh — Config file update functions for the upstream release tracker.
# Source this file; it defines functions only and has no top-level side effects.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/tracker-utils.sh"

###############################################################################
# update_config_files(framework_key, new_version, config_entries_json)
#   Iterates over config file entries, updates common.framework_version and
#   common.prod_image using yq. Echoes the list of updated file paths
#   (one per line).
#
#   Arguments:
#     framework_key       — Framework identifier (e.g., "vllm")
#     new_version         — New upstream version string (e.g., "0.17.0")
#     config_entries_json — JSON array of objects with "path" and
#                           "prod_image_template" fields
#
#   Usage:
#     updated=$(update_config_files "vllm" "0.17.0" '[
#       {"path": ".github/config/vllm-ec2.yml", "prod_image_template": "vllm:{major}.{minor}-gpu-py312-ec2"},
#       {"path": ".github/config/vllm-sagemaker.yml", "prod_image_template": "vllm:{major}.{minor}-gpu-py312"}
#     ]')
###############################################################################
update_config_files() {
  local framework_key="${1:?Usage: update_config_files FRAMEWORK_KEY NEW_VERSION CONFIG_ENTRIES_JSON}"
  local new_version="${2:?Usage: update_config_files FRAMEWORK_KEY NEW_VERSION CONFIG_ENTRIES_JSON}"
  local config_entries_json="${3:?Usage: update_config_files FRAMEWORK_KEY NEW_VERSION CONFIG_ENTRIES_JSON}"

  local entry_count
  entry_count="$(echo "${config_entries_json}" | jq -r 'length')"

  if [[ "${entry_count}" -eq 0 ]]; then
    echo "Warning: no config entries provided for ${framework_key}" >&2
    return 0
  fi

  for i in $(seq 0 $(( entry_count - 1 ))); do
    local config_path
    local template

    config_path="$(echo "${config_entries_json}" | jq -r ".[$i].path")"
    template="$(echo "${config_entries_json}" | jq -r ".[$i].prod_image_template")"

    # Validate the config file exists
    if [[ ! -f "${config_path}" ]]; then
      echo "Error: config file not found: ${config_path}" >&2
      return 1
    fi

    # Update common.framework_version
    yq eval -i ".common.framework_version = \"${new_version}\"" "${config_path}"

    # Render prod_image from template and update
    local new_prod_image
    new_prod_image="$(render_prod_image "${template}" "${new_version}")"
    yq eval -i ".common.prod_image = \"${new_prod_image}\"" "${config_path}"

    echo "${config_path}"
  done
}

###############################################################################
# update_dockerfiles(new_version, dockerfile_entries_json)
#   Iterates over Dockerfile entries, renders base_image_template with
#   {version} replaced by new_version, and uses sed to update the
#   ARG BASE_IMAGE= line. Echoes the list of updated Dockerfile paths
#   (one per line).
#
#   Arguments:
#     new_version             — New upstream version string (e.g., "0.17.0")
#     dockerfile_entries_json — JSON array of objects with "path" and
#                               "base_image_template" fields
#
#   Usage:
#     updated=$(update_dockerfiles "0.17.0" '[
#       {"path": "docker/vllm/Dockerfile", "base_image_template": "vllm/vllm-openai:v{version}"},
#       {"path": "docker/sglang/Dockerfile", "base_image_template": "lmsysorg/sglang:v{version}-cu129-amd64"}
#     ]')
###############################################################################
update_dockerfiles() {
  local new_version="${1:?Usage: update_dockerfiles NEW_VERSION DOCKERFILE_ENTRIES_JSON}"
  local dockerfile_entries_json="${2:?Usage: update_dockerfiles NEW_VERSION DOCKERFILE_ENTRIES_JSON}"

  local entry_count
  entry_count="$(echo "${dockerfile_entries_json}" | jq -r 'length')"

  if [[ "${entry_count}" -eq 0 ]]; then
    echo "Warning: no Dockerfile entries provided" >&2
    return 0
  fi

  for i in $(seq 0 $(( entry_count - 1 ))); do
    local dockerfile_path
    local template

    dockerfile_path="$(echo "${dockerfile_entries_json}" | jq -r ".[$i].path")"
    template="$(echo "${dockerfile_entries_json}" | jq -r ".[$i].base_image_template")"

    # Validate the Dockerfile exists
    if [[ ! -f "${dockerfile_path}" ]]; then
      echo "Error: Dockerfile not found: ${dockerfile_path}" >&2
      return 1
    fi

    # Render the new BASE_IMAGE value by replacing {version} with new_version
    local new_base_image
    new_base_image="${template//\{version\}/${new_version}}"

    # Update the ARG BASE_IMAGE= line in the Dockerfile
    sed -i "s|^ARG BASE_IMAGE=.*|ARG BASE_IMAGE=${new_base_image}|" "${dockerfile_path}"

    echo "${dockerfile_path}"
  done
}

###############################################################################
# detect_cuda_python_changes(docker_hub_image, new_version, config_entries_json)
#   Queries Docker Hub API for image tag metadata, extracts CUDA/Python
#   versions from tag names, and updates config files if versions differ.
#
#   This function is best-effort: if the Docker Hub API fails or metadata
#   is missing, it echoes a warning string to stdout and returns 0.
#   It NEVER fails the workflow.
#
#   Arguments:
#     docker_hub_image    — Docker Hub image name (e.g., "vllm/vllm-openai")
#     new_version         — New upstream version string (e.g., "0.17.0")
#     config_entries_json — JSON array of config file entries with "path" fields
#
#   Output (stdout):
#     If detection fails: a warning string starting with "⚠️"
#     If CUDA changed: "CUDA_CHANGED:<old>:<new>"
#     If Python changed: "PYTHON_CHANGED:<old>:<new>"
#     (multiple lines possible)
#
#   Usage:
#     warnings=$(detect_cuda_python_changes "vllm/vllm-openai" "0.17.0" '[
#       {"path": ".github/config/vllm-ec2.yml"},
#       {"path": ".github/config/vllm-sagemaker.yml"}
#     ]')
###############################################################################
detect_cuda_python_changes() {
  local docker_hub_image="${1:?Usage: detect_cuda_python_changes DOCKER_HUB_IMAGE NEW_VERSION CONFIG_ENTRIES_JSON}"
  local new_version="${2:?Usage: detect_cuda_python_changes DOCKER_HUB_IMAGE NEW_VERSION CONFIG_ENTRIES_JSON}"
  local config_entries_json="${3:?Usage: detect_cuda_python_changes DOCKER_HUB_IMAGE NEW_VERSION CONFIG_ENTRIES_JSON}"

  # Split image into namespace/repo for Docker Hub API
  local namespace repo
  namespace="$(echo "${docker_hub_image}" | cut -d'/' -f1)"
  repo="$(echo "${docker_hub_image}" | cut -d'/' -f2)"

  if [[ -z "${namespace}" || -z "${repo}" ]]; then
    echo "⚠️ Could not parse Docker Hub image name '${docker_hub_image}'. Please verify CUDA/Python versions manually."
    return 0
  fi

  # Try the primary tag format: v{version}
  local primary_tag="v${new_version}"
  local api_url="https://hub.docker.com/v2/repositories/${namespace}/${repo}/tags/${primary_tag}"

  local api_response
  api_response="$(curl -s --max-time 10 "${api_url}" 2>/dev/null)" || {
    echo "⚠️ Could not reach Docker Hub API for ${docker_hub_image}:${primary_tag}. Please verify CUDA/Python versions manually."
    return 0
  }

  # Check if the API returned an error (e.g., tag not found)
  if echo "${api_response}" | jq -e '.errinfo // .message // empty' &>/dev/null; then
    echo "⚠️ Docker Hub tag '${primary_tag}' not found for ${docker_hub_image}. Please verify CUDA/Python versions manually."
    return 0
  fi

  # Try to find related tags that contain CUDA/Python version info.
  # Docker Hub tags API can list tags matching a pattern. We search for tags
  # that start with the version and may contain cu/py suffixes.
  local upstream_cuda=""
  local upstream_python=""

  # Strategy 1: Check if the primary tag response has useful metadata
  # (Docker Hub v2 tags endpoint returns digest info but not labels directly)

  # Strategy 2: Search for related tags that contain version + cu/py patterns
  # e.g., "v0.17.0-cu129-py312" or "v0.17.0-cu129"
  local tags_url="https://hub.docker.com/v2/repositories/${namespace}/${repo}/tags?page_size=100&name=v${new_version}"
  local tags_response
  tags_response="$(curl -s --max-time 10 "${tags_url}" 2>/dev/null)" || {
    echo "⚠️ Could not fetch tags from Docker Hub for ${docker_hub_image}. Please verify CUDA/Python versions manually."
    return 0
  }

  # Extract all tag names matching this version
  local tag_names
  tag_names="$(echo "${tags_response}" | jq -r '.results[]?.name // empty' 2>/dev/null)" || {
    echo "⚠️ Could not parse Docker Hub tags response for ${docker_hub_image}. Please verify CUDA/Python versions manually."
    return 0
  }

  if [[ -z "${tag_names}" ]]; then
    echo "⚠️ No tags found on Docker Hub for ${docker_hub_image} matching version ${new_version}. Please verify CUDA/Python versions manually."
    return 0
  fi

  # Search tag names for CUDA version pattern (cu + digits, e.g., cu129, cu130)
  local cuda_match
  cuda_match="$(echo "${tag_names}" | grep -oP 'cu\d+' | head -1)" || true
  if [[ -n "${cuda_match}" ]]; then
    upstream_cuda="${cuda_match}"
  fi

  # Search tag names for Python version pattern (py + digits, e.g., py312, py311)
  local python_match
  python_match="$(echo "${tag_names}" | grep -oP 'py\d+' | head -1)" || true
  if [[ -n "${python_match}" ]]; then
    upstream_python="${python_match}"
  fi

  # If we couldn't extract either version, warn and return
  if [[ -z "${upstream_cuda}" && -z "${upstream_python}" ]]; then
    echo "⚠️ Could not detect CUDA/Python versions from Docker Hub tags for ${docker_hub_image}:${primary_tag}. Please verify manually."
    return 0
  fi

  # Read current versions from the first config file
  local first_config_path
  first_config_path="$(echo "${config_entries_json}" | jq -r '.[0].path')"

  if [[ ! -f "${first_config_path}" ]]; then
    echo "⚠️ Config file not found: ${first_config_path}. Cannot compare CUDA/Python versions."
    return 0
  fi

  local current_cuda current_python
  current_cuda="$(yq eval '.common.cuda_version // ""' "${first_config_path}")" || true
  current_python="$(yq eval '.common.python_version // ""' "${first_config_path}")" || true

  local entry_count
  entry_count="$(echo "${config_entries_json}" | jq -r 'length')"

  # Update CUDA version if changed
  if [[ -n "${upstream_cuda}" && "${upstream_cuda}" != "${current_cuda}" ]]; then
    for i in $(seq 0 $(( entry_count - 1 ))); do
      local config_path
      config_path="$(echo "${config_entries_json}" | jq -r ".[$i].path")"
      if [[ -f "${config_path}" ]]; then
        yq eval -i ".common.cuda_version = \"${upstream_cuda}\"" "${config_path}"
      fi
    done
    echo "CUDA_CHANGED:${current_cuda}:${upstream_cuda}"
  fi

  # Update Python version if changed
  if [[ -n "${upstream_python}" && "${upstream_python}" != "${current_python}" ]]; then
    for i in $(seq 0 $(( entry_count - 1 ))); do
      local config_path
      config_path="$(echo "${config_entries_json}" | jq -r ".[$i].path")"
      if [[ -f "${config_path}" ]]; then
        yq eval -i ".common.python_version = \"${upstream_python}\"" "${config_path}"
      fi
    done
    echo "PYTHON_CHANGED:${current_python}:${upstream_python}"
  fi

  return 0
}

###############################################################################
# rename_test_setup_script(old_version, new_version, test_setup_config_json)
#   Renames a version-named test setup script from the old version pattern to
#   the new version pattern using `git mv`, then updates all workflow file
#   references using `sed`. Echoes the list of updated file paths (one per
#   line): the new script path followed by each updated workflow file.
#
#   Arguments:
#     old_version            — Current version string (e.g., "0.16.0")
#     new_version            — New version string (e.g., "0.17.0")
#     test_setup_config_json — JSON object with "pattern" and "workflow_files":
#       {
#         "pattern": "scripts/vllm/vllm_{version_underscored}_test_setup.sh",
#         "workflow_files": [
#           ".github/workflows/auto-release-vllm-sagemaker.yml",
#           ".github/workflows/auto-release-vllm-ec2.yml"
#         ]
#       }
#
#   Usage:
#     updated=$(rename_test_setup_script "0.16.0" "0.17.0" '{
#       "pattern": "scripts/vllm/vllm_{version_underscored}_test_setup.sh",
#       "workflow_files": [
#         ".github/workflows/auto-release-vllm-sagemaker.yml",
#         ".github/workflows/auto-release-vllm-ec2.yml"
#       ]
#     }')
###############################################################################
rename_test_setup_script() {
  local old_version="${1:?Usage: rename_test_setup_script OLD_VERSION NEW_VERSION TEST_SETUP_CONFIG_JSON}"
  local new_version="${2:?Usage: rename_test_setup_script OLD_VERSION NEW_VERSION TEST_SETUP_CONFIG_JSON}"
  local test_setup_config_json="${3:?Usage: rename_test_setup_script OLD_VERSION NEW_VERSION TEST_SETUP_CONFIG_JSON}"

  # Convert versions to underscored format: "0.16.0" → "0_16_0"
  local old_underscored="${old_version//./_}"
  local new_underscored="${new_version//./_}"

  # Extract pattern and workflow_files from JSON
  local pattern
  pattern="$(echo "${test_setup_config_json}" | jq -r '.pattern')"

  # Resolve old and new script paths from pattern
  local old_script_path="${pattern//\{version_underscored\}/${old_underscored}}"
  local new_script_path="${pattern//\{version_underscored\}/${new_underscored}}"

  # Validate that the old script file exists
  if [[ ! -f "${old_script_path}" ]]; then
    echo "Error: old test setup script not found: ${old_script_path}" >&2
    return 1
  fi

  # Rename the script using git mv
  git mv "${old_script_path}" "${new_script_path}"
  echo "${new_script_path}"

  # Update references in each workflow file
  local workflow_count
  workflow_count="$(echo "${test_setup_config_json}" | jq -r '.workflow_files | length')"

  for i in $(seq 0 $(( workflow_count - 1 ))); do
    local workflow_file
    workflow_file="$(echo "${test_setup_config_json}" | jq -r ".workflow_files[$i]")"

    # Validate that the workflow file exists
    if [[ ! -f "${workflow_file}" ]]; then
      echo "Error: workflow file not found: ${workflow_file}" >&2
      return 1
    fi

    # Replace old script path with new script path in the workflow file
    sed -i "s|${old_script_path}|${new_script_path}|g" "${workflow_file}"
    echo "${workflow_file}"
  done
}
