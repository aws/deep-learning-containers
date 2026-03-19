#!/usr/bin/env bash
# docs-pr.sh — Generate a docs data YAML file from a released Docker image
# and create a PR via the GitHub CLI.
#
# This script is called by the step4-docs-pr job in reusable-release-image.yml.
# It expects the following environment variables to be set by the workflow:
#
#   FRAMEWORK        — framework identifier (e.g. "vllm", "sglang")
#   VERSION          — framework version (e.g. "0.17.1")
#   PYTHON           — python version tag (e.g. "py312")
#   CUDA             — cuda version tag (e.g. "cu129")
#   OS               — os version (e.g. "ubuntu22.04")
#   PLATFORM         — customer type (e.g. "ec2", "sagemaker")
#   DEVICE           — device type (e.g. "gpu")
#   PUBLIC_REGISTRY  — whether image is in public ECR (true/false)
#   PROD_IMAGE       — production image name (unused, kept for reference)
#   GH_TOKEN         — GitHub App token for push/PR operations
#
# Usage:
#   bash scripts/autocurrency/docs-pr.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/utils.sh"

###############################################################################
# Helpers (pure functions — sourced by test harness)
###############################################################################

get_display_name() {
  local framework="$1"
  case "$framework" in
    vllm)   echo "vLLM" ;;
    sglang) echo "SGLang" ;;
    *)      echo "$framework" ;;
  esac
}

parse_major_minor() {
  local version="$1"
  echo "$version" | grep -oE '^[0-9]+\.[0-9]+'
}

# Generate 4 Docker image tags for a given platform.
# Usage: generate_tags <version> <device> <python> <cuda> <os> <platform>
# Outputs one tag per line.
generate_tags() {
  local version="$1" device="$2" python="$3" cuda="$4" os_ver="$5" platform="$6"
  local mm
  mm=$(parse_major_minor "$version")

  if [ "$platform" = "ec2" ]; then
    echo "${version}-${device}-${python}-${cuda}-${os_ver}-ec2"
    echo "${mm}-${device}-${python}-${cuda}-${os_ver}-ec2-v1"
    echo "${version}-${device}-${python}-ec2"
    echo "${mm}-${device}-${python}-ec2"
  elif [ "$platform" = "sagemaker" ]; then
    echo "${version}-${device}-${python}-${cuda}-${os_ver}-sagemaker"
    echo "${mm}-${device}-${python}-${cuda}-${os_ver}-sagemaker-v1"
    echo "${version}-${device}-${python}"
    echo "${mm}-${device}-${python}"
  fi
}

# Generate a release announcement string.
# Usage: generate_announcement <framework> <version> <platform>
generate_announcement() {
  local framework="$1" version="$2" platform="$3"
  local display
  display=$(get_display_name "$framework")
  if [ "$platform" = "ec2" ]; then
    echo "Introduced ${display} ${version} containers for EC2, ECS, EKS"
  elif [ "$platform" = "sagemaker" ]; then
    echo "Introduced ${display} ${version} containers for SageMaker"
  fi
}

# Generate the git branch name for a docs-update PR.
# Usage: generate_branch_name <framework> <version> <platform>
generate_branch_name() {
  local framework="$1" version="$2" platform="$3"
  echo "docs/auto-update-${framework}-${version}-${platform}"
}

# Generate the PR title.
# Usage: generate_pr_title <framework> <version> <platform>
generate_pr_title() {
  local framework="$1" version="$2" platform="$3"
  local display
  display=$(get_display_name "$framework")
  echo "docs: Add ${display} ${version} ${platform^^} image data"
}

###############################################################################
# Step 1: Pull image and extract package versions
###############################################################################

pull_and_extract_packages() {
  local image_uri="$1"

  echo "Pulling image: ${image_uri}"
  docker pull "${image_uri}"

  # Read package lists from tracker config
  local tracker="${REPO_ROOT}/${TRACKER_FILE:-".github/config/autocurrency-tracker.yml"}"
  local pip_count system_count
  pip_count=$(yq eval ".frameworks.${FRAMEWORK}.docs_packages.pip | length" "$tracker")
  system_count=$(yq eval ".frameworks.${FRAMEWORK}.docs_packages.system | length" "$tracker")

  if [[ "${pip_count}" == "0" && "${system_count}" == "0" ]]; then
    echo "::warning::No docs_packages defined for '${FRAMEWORK}' in tracker config"
  fi

  FAILED_PACKAGES=()

  # Extract pip package versions (name/key from config)
  for i in $(seq 0 $(( pip_count - 1 ))); do
    local pkg_name output_key version=""
    pkg_name=$(yq eval ".frameworks.${FRAMEWORK}.docs_packages.pip[$i].name" "$tracker")
    output_key=$(yq eval ".frameworks.${FRAMEWORK}.docs_packages.pip[$i].key // \"${pkg_name}\"" "$tracker")

    version=$(docker run --rm "$image_uri" pip show "$pkg_name" 2>/dev/null | grep "^Version:" | awk '{print $2}') || true
    if [ -n "$version" ]; then
      echo "  ✅ ${output_key}: ${version}"
      export "PKG_${output_key}=${version}"
      echo "PKG_${output_key}=${version}" >> "$GITHUB_ENV"
    else
      echo "::warning::Failed to extract version for pip package '${pkg_name}' (key: ${output_key})"
      FAILED_PACKAGES+=("$output_key")
      export "PKG_${output_key}="
      echo "PKG_${output_key}=" >> "$GITHUB_ENV"
    fi
  done

  # Extract system package versions (extraction commands are framework-agnostic)
  for i in $(seq 0 $(( system_count - 1 ))); do
    local sys_pkg version=""
    sys_pkg=$(yq eval ".frameworks.${FRAMEWORK}.docs_packages.system[$i]" "$tracker")

    case "$sys_pkg" in
      cuda)
        version=$(docker run --rm "$image_uri" nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//') || true
        ;;
      nccl)
        version=$(docker run --rm "$image_uri" python3 -c "import torch; print(torch.cuda.nccl.version())" 2>/dev/null) || true
        ;;
      efa)
        version=$(docker run --rm "$image_uri" fi_info --version 2>/dev/null | head -1 | awk '{print $2}') || true
        ;;
      cudnn)
        version=$(docker run --rm "$image_uri" python3 -c "import torch; print(torch.backends.cudnn.version())" 2>/dev/null) || true
        ;;
      gdrcopy)
        version=$(docker run --rm "$image_uri" dpkg -l 2>/dev/null | grep gdrcopy | awk '{print $3}' | head -1) || true
        ;;
      *)
        echo "::warning::Unknown system package '${sys_pkg}', skipping"
        continue
        ;;
    esac
    if [ -n "$version" ]; then
      echo "  ✅ ${sys_pkg}: ${version}"
      export "PKG_${sys_pkg}=${version}"
      echo "PKG_${sys_pkg}=${version}" >> "$GITHUB_ENV"
    else
      echo "::warning::Failed to extract version for system package '${sys_pkg}'"
      FAILED_PACKAGES+=("$sys_pkg")
      export "PKG_${sys_pkg}="
      echo "PKG_${sys_pkg}=" >> "$GITHUB_ENV"
    fi
  done

  # Store failed packages as comma-separated string
  local failed_str
  failed_str=$(IFS=,; echo "${FAILED_PACKAGES[*]}")
  export FAILED_PACKAGES="${failed_str}"
  echo "FAILED_PACKAGES=${failed_str}" >> "$GITHUB_ENV"
  if [ ${#FAILED_PACKAGES[@]} -gt 0 ]; then
    echo "::warning::Failed to extract versions for: ${failed_str}"
  else
    echo "✅ All package versions extracted successfully"
  fi
}

###############################################################################
# Step 2: Generate docs data YAML file
###############################################################################

generate_docs_yaml() {
  local display_name
  display_name=$(get_display_name "$FRAMEWORK")

  local major_minor
  major_minor=$(parse_major_minor "$VERSION")

  # Generate tags using the helper function
  local tags
  tags=$(generate_tags "$VERSION" "$DEVICE" "$PYTHON" "$CUDA" "$OS" "$PLATFORM")
  local tag1 tag2 tag3 tag4
  tag1=$(echo "$tags" | sed -n '1p')
  tag2=$(echo "$tags" | sed -n '2p')
  tag3=$(echo "$tags" | sed -n '3p')
  tag4=$(echo "$tags" | sed -n '4p')

  # Generate announcement using the helper function
  local announcement
  announcement=$(generate_announcement "$FRAMEWORK" "$VERSION" "$PLATFORM")

  # Write the YAML file
  local output_dir="${REPO_ROOT}/docs/src/data/${FRAMEWORK}"
  OUTPUT_FILE="${output_dir}/${VERSION}-${DEVICE}-${PLATFORM}.yml"
  mkdir -p "$output_dir"

  # Build packages section dynamically from tracker config
  local tracker="${REPO_ROOT}/${TRACKER_FILE:-".github/config/autocurrency-tracker.yml"}"
  local packages_yaml=""

  # Pip packages
  local pip_count
  pip_count=$(yq eval ".frameworks.${FRAMEWORK}.docs_packages.pip | length" "$tracker")
  for i in $(seq 0 $(( pip_count - 1 ))); do
    local pkg_name output_key
    pkg_name=$(yq eval ".frameworks.${FRAMEWORK}.docs_packages.pip[$i].name" "$tracker")
    output_key=$(yq eval ".frameworks.${FRAMEWORK}.docs_packages.pip[$i].key // \"${pkg_name}\"" "$tracker")
    local env_var="PKG_${output_key}"
    packages_yaml="${packages_yaml}  ${output_key}: \"${!env_var:-}\"\n"
  done

  # System packages
  local system_count
  system_count=$(yq eval ".frameworks.${FRAMEWORK}.docs_packages.system | length" "$tracker")
  for i in $(seq 0 $(( system_count - 1 ))); do
    local sys_pkg
    sys_pkg=$(yq eval ".frameworks.${FRAMEWORK}.docs_packages.system[$i]" "$tracker")
    local env_var="PKG_${sys_pkg}"
    packages_yaml="${packages_yaml}  ${sys_pkg}: \"${!env_var:-}\"\n"
  done

  # Write the YAML using a single template
  cat > "$OUTPUT_FILE" <<EOF
framework: ${display_name}
version: "${VERSION}"
accelerator: ${DEVICE}
python: ${PYTHON}
cuda: ${CUDA}
os: ${OS}
platform: ${PLATFORM}
public_registry: ${PUBLIC_REGISTRY}

tags:
  - "${tag1}"
  - "${tag2}"
  - "${tag3}"
  - "${tag4}"

announcements:
  - "${announcement}"

packages:
$(echo -e "${packages_yaml}" | sed '/^$/d')
EOF

  echo "✅ Generated docs data file: ${OUTPUT_FILE}"
  cat "$OUTPUT_FILE"
  export OUTPUT_FILE
  echo "OUTPUT_FILE=${OUTPUT_FILE}" >> "$GITHUB_ENV"
}

###############################################################################
# Step 3: Create branch, commit, and open PR
###############################################################################

create_pr() {
  local display_name
  display_name=$(get_display_name "$FRAMEWORK")
  local branch_name
  branch_name=$(generate_branch_name "$FRAMEWORK" "$VERSION" "$PLATFORM")
  local pr_title
  pr_title=$(generate_pr_title "$FRAMEWORK" "$VERSION" "$PLATFORM")

  # Configure git
  git config user.name "asimov-bot[bot]"
  git config user.email "asimov-bot[bot]@users.noreply.github.com"

  # Create and push branch (force-push for idempotency)
  git checkout -B "$branch_name"
  git add "$OUTPUT_FILE"
  git commit -m "$pr_title"
  git push --force origin "$branch_name"

  # Build PR body
  local pr_body="Auto-generated by the release workflow.

This PR adds the docs data file for ${display_name} ${VERSION} (${PLATFORM}).

**File:** \`${OUTPUT_FILE}\`"

  # Add failed packages warning if any
  if [ -n "${FAILED_PACKAGES:-}" ]; then
    pr_body="${pr_body}

---

⚠️ **Warning: Some package versions could not be extracted:**

$(echo "$FAILED_PACKAGES" | tr ',' '\n' | while read -r pkg; do echo "- \`${pkg}\`"; done)"
  fi

  # Create or update PR
  local existing_pr
  existing_pr=$(gh pr list --head "$branch_name" --json number --jq '.[0].number' 2>/dev/null) || true
  if [ -n "$existing_pr" ] && [ "$existing_pr" != "null" ]; then
    echo "Updating existing PR #${existing_pr}"
    gh pr edit "$existing_pr" --title "$pr_title" --body "$pr_body"
    PR_URL=$(gh pr view "$existing_pr" --json url --jq '.url')
  else
    echo "Creating new PR"
    PR_URL=$(gh pr create --title "$pr_title" --body "$pr_body" --head "$branch_name" --base main)
  fi

  echo "✅ PR URL: ${PR_URL}"
  export PR_URL
  echo "PR_URL=${PR_URL}" >> "$GITHUB_ENV"
}

###############################################################################
# Step 4: Send Slack notification
###############################################################################

send_notification() {
  local pr_url_or_run_url="${PR_URL:-}"

  # Fall back to the Actions run URL if no PR was created
  if [ -z "${pr_url_or_run_url}" ]; then
    pr_url_or_run_url="${GITHUB_SERVER_URL}/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}"
  fi

  send_slack_notification \
    "${SLACK_WEBHOOK_URL:-}" \
    "docs_update" \
    "${FRAMEWORK:-unknown}" \
    "${VERSION:-unknown}" \
    "${pr_url_or_run_url}" || true
}

###############################################################################
# Run all steps: parse spec → extract packages → generate YAML → PR → notify
###############################################################################

run_all() {
  local release_spec="${1:?Usage: docs-pr.sh run-all <release-spec-yaml> <image-uri>}"
  local image_uri="${2:?Usage: docs-pr.sh run-all <release-spec-yaml> <image-uri>}"

  # Parse release spec into environment variables
  echo "Parsing release spec: ${release_spec}"
  FRAMEWORK=$(yq '.framework' "$release_spec")
  VERSION=$(yq '.version' "$release_spec")
  PYTHON=$(yq '.python_version' "$release_spec")
  CUDA=$(yq '.cuda_version' "$release_spec")
  OS=$(yq '.os_version' "$release_spec")
  PLATFORM=$(yq '.customer_type' "$release_spec")
  DEVICE=$(yq '.device_type' "$release_spec")
  PUBLIC_REGISTRY=$(yq '.public_registry' "$release_spec")
  export FRAMEWORK VERSION PYTHON CUDA OS PLATFORM DEVICE PUBLIC_REGISTRY

  pull_and_extract_packages "$image_uri"
  generate_docs_yaml
  create_pr
  send_notification
}

###############################################################################
# Main — dispatch based on subcommand
###############################################################################

main() {
  local cmd="${1:?Usage: docs-pr.sh <run-all|extract-packages|generate-yaml|create-pr|notify>}"

  case "$cmd" in
    run-all)
      local release_spec="${2:?Usage: docs-pr.sh run-all <release-spec-yaml> <image-uri>}"
      local image_uri="${3:?Usage: docs-pr.sh run-all <release-spec-yaml> <image-uri>}"
      run_all "$release_spec" "$image_uri"
      ;;
    extract-packages)
      local image_uri="${2:?Usage: docs-pr.sh extract-packages <image-uri>}"
      pull_and_extract_packages "$image_uri"
      ;;
    generate-yaml)
      generate_docs_yaml
      ;;
    create-pr)
      create_pr
      ;;
    notify)
      send_notification
      ;;
    *)
      echo "Unknown command: $cmd"
      echo "Usage: docs-pr.sh <run-all|extract-packages|generate-yaml|create-pr|notify>"
      exit 1
      ;;
  esac
}

# Only run main when executed directly (not when sourced for testing)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
