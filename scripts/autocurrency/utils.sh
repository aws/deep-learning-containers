#!/usr/bin/env bash
# utils.sh — Shared utility functions for the upstream release tracker.
# Source this file; it defines functions only and has no top-level side effects.

set -euo pipefail

###############################################################################
# strip_tag_prefix(tag, [prefix])
#   Strips the configured prefix from a release tag.
#   Defaults to "v" if no prefix is provided.
#   If the tag doesn't start with the prefix, returns the tag as-is.
#
#   Usage: version=$(strip_tag_prefix "v0.17.0" "v")
###############################################################################
strip_tag_prefix() {
  local tag="${1:?Usage: strip_tag_prefix TAG [PREFIX]}"
  local prefix="${2:-v}"

  if [[ "${tag}" == "${prefix}"* ]]; then
    echo "${tag#"${prefix}"}"
  else
    echo "${tag}"
  fi
}

###############################################################################
# is_rc_version(version)
#   Returns 0 (true) if the version string contains an RC suffix.
#   Case-insensitive match for patterns: rc\d+, -rc.\d+
#
#   Examples:
#     is_rc_version "0.17.0rc1"   → 0 (true)
#     is_rc_version "0.17.0-rc.2" → 0 (true)
#     is_rc_version "0.17.0RC1"   → 0 (true)
#     is_rc_version "0.17.0"      → 1 (false)
#     is_rc_version "0.17.0.post1"→ 1 (false)
###############################################################################
is_rc_version() {
  local version="${1:?Usage: is_rc_version VERSION}"

  # Convert to lowercase for case-insensitive matching
  local lower
  lower="$(echo "${version}" | tr '[:upper:]' '[:lower:]')"

  # Match patterns: "rc" followed by digits, or "-rc." followed by digits
  if [[ "${lower}" =~ rc[0-9]+ ]]; then
    return 0
  fi
  if [[ "${lower}" =~ -rc\.[0-9]+ ]]; then
    return 0
  fi

  return 1
}


###############################################################################
# is_newer_version(upstream, current)
#   Returns 0 (true) if upstream is strictly greater than current.
#   Numeric segment-by-segment comparison, padded to 3 segments.
#   Caller must ensure RC versions are filtered before calling.
#
#   Examples:
#     is_newer_version "0.17.0" "0.16.0" → 0 (true)
#     is_newer_version "0.16.0" "0.16.0" → 1 (false)
#     is_newer_version "0.9.0"  "0.16.0" → 1 (false)
###############################################################################
is_newer_version() {
  local upstream="${1:?Usage: is_newer_version UPSTREAM CURRENT}"
  local current="${2:?Usage: is_newer_version UPSTREAM CURRENT}"

  # Split into arrays on "."
  IFS='.' read -ra u_parts <<< "${upstream}"
  IFS='.' read -ra c_parts <<< "${current}"

  # Pad to 3 segments with zeros
  while [[ ${#u_parts[@]} -lt 3 ]]; do u_parts+=("0"); done
  while [[ ${#c_parts[@]} -lt 3 ]]; do c_parts+=("0"); done

  for i in 0 1 2; do
    local u_seg=$((10#${u_parts[$i]}))
    local c_seg=$((10#${c_parts[$i]}))
    if (( u_seg > c_seg )); then
      return 0
    fi
    if (( u_seg < c_seg )); then
      return 1
    fi
  done

  # Versions are equal
  return 1
}

###############################################################################
# get_current_version(config_file)
#   Reads common.framework_version from a config YAML using yq.
#   Echoes the version string.
#
#   Usage: version=$(get_current_version ".github/config/vllm-ec2.yml")
###############################################################################
get_current_version() {
  local config_file="${1:?Usage: get_current_version CONFIG_FILE}"

  if [[ ! -f "${config_file}" ]]; then
    echo "Error: config file not found: ${config_file}" >&2
    return 1
  fi

  local version
  version="$(yq eval '.common.framework_version' "${config_file}")"

  if [[ -z "${version}" || "${version}" == "null" ]]; then
    echo "Error: common.framework_version not found in ${config_file}" >&2
    return 1
  fi

  echo "${version}"
}

###############################################################################
# render_prod_image(template, version)
#   Substitutes {major}, {minor}, {patch} placeholders in the template
#   using segments from the version string. Pads to 3 segments.
#
#   Example:
#     render_prod_image "vllm:{major}.{minor}-gpu-py312-ec2" "0.17.0"
#     → "vllm:0.17-gpu-py312-ec2"
###############################################################################
render_prod_image() {
  local template="${1:?Usage: render_prod_image TEMPLATE VERSION}"
  local version="${2:?Usage: render_prod_image TEMPLATE VERSION}"

  IFS='.' read -ra parts <<< "${version}"

  # Pad to 3 segments with zeros
  while [[ ${#parts[@]} -lt 3 ]]; do parts+=("0"); done

  local major="${parts[0]}"
  local minor="${parts[1]}"
  local patch="${parts[2]}"

  local result="${template}"
  result="${result//\{major\}/${major}}"
  result="${result//\{minor\}/${minor}}"
  result="${result//\{patch\}/${patch}}"

  echo "${result}"
}

###############################################################################
# send_slack_notification(webhook_url, framework_key, new_version, pr_url)
#   Sends key-value data to a Slack Workflow webhook after a PR is created.
#   The Slack Workflow on the receiving end handles message formatting and
#   channel routing — this function only provides raw data.
#
#   Returns 0 (true) if HTTP 200, 1 (false) otherwise.
#   NEVER fails the workflow — logs warnings on error.
#
#   Arguments:
#     webhook_url    — Slack Workflow webhook URL (from SLACK_WEBHOOK_URL secret)
#     framework_key  — Framework identifier (e.g., "vllm")
#     new_version    — New upstream version string (e.g., "0.17.0")
#     pr_url         — URL of the created pull request
#
#   Usage:
#     send_slack_notification "${SLACK_WEBHOOK_URL}" "vllm" "0.17.0" \
#       "https://github.com/aws/deep-learning-containers/pull/123"
###############################################################################
send_slack_notification() {
  local webhook_url="${1:-}"
  local framework_key="${2:?Usage: send_slack_notification WEBHOOK_URL FRAMEWORK_KEY NEW_VERSION PR_URL}"
  local new_version="${3:?Usage: send_slack_notification WEBHOOK_URL FRAMEWORK_KEY NEW_VERSION PR_URL}"
  local pr_url="${4:?Usage: send_slack_notification WEBHOOK_URL FRAMEWORK_KEY NEW_VERSION PR_URL}"

  # If webhook URL is missing or empty, skip silently
  if [[ -z "${webhook_url}" ]]; then
    echo "Slack notifications: webhook URL not configured. Skipping."
    return 1
  fi

  # Construct simple key-value JSON payload
  local payload
  payload=$(jq -n \
    --arg is_auto_currency "true" \
    --arg framework_name "${framework_key}" \
    --arg framework_version "${new_version}" \
    --arg pr_url "${pr_url}" \
    '{
      is_auto_currency: $is_auto_currency,
      pr_url: $pr_url,
      framework_name: $framework_name,
      framework_version: $framework_version,
      is_docs_updates: ""
    }')

  # POST to Slack Workflow webhook
  local http_code
  http_code=$(curl -s -o /dev/null -w "%{http_code}" \
    --max-time 10 \
    -X POST \
    -H "Content-Type: application/json" \
    -d "${payload}" \
    "${webhook_url}" 2>/dev/null) || {
    echo "Warning: Slack notification failed (network error) for ${framework_key} ${new_version}. PR was created successfully."
    return 1
  }

  if [[ "${http_code}" == "200" ]]; then
    echo "Slack notification sent successfully for ${framework_key} ${new_version}."
    return 0
  else
    echo "Warning: Slack notification returned HTTP ${http_code} for ${framework_key} ${new_version}. PR was created successfully."
    return 1
  fi
}
