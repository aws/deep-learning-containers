#!/usr/bin/env bash
# update-configs.sh — Config file update functions for the upstream release tracker.
# Source this file; it defines functions only and has no top-level side effects.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils.sh"

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
#       {"path": ".github/config/image/vllm-ec2.yml", "prod_image_template": "vllm:{major}.{minor}-gpu-py312-ec2"},
#       {"path": ".github/config/image/vllm-sagemaker.yml", "prod_image_template": "vllm:{major}.{minor}-gpu-py312"}
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
#       {"path": "docker/sglang/Dockerfile", "base_image_template": "lmsysorg/sglang:v{version}"}
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
#         "pattern": "test/vllm/scripts/vllm_{version_underscored}_test_setup.sh",
#         "workflow_files": [
#           ".github/workflows/autorelease-vllm-sagemaker.yml",
#           ".github/workflows/autorelease-vllm-ec2.yml"
#         ]
#       }
#
#   Usage:
#     updated=$(rename_test_setup_script "0.16.0" "0.17.0" '{
#       "pattern": "test/vllm/scripts/vllm_{version_underscored}_test_setup.sh",
#       "workflow_files": [
#         ".github/workflows/autorelease-vllm-sagemaker.yml",
#         ".github/workflows/autorelease-vllm-ec2.yml"
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
