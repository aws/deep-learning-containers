#!/usr/bin/env bash
# docs-pr.sh — Generate a docs data YAML file from a released Docker image
# and create a PR via the GitHub CLI.
#
# Called by the step4-docs-pr job in reusable-release-image.yml.
# Main logic lives at the bottom (guarded by BASH_SOURCE check so tests
# can source the helper functions without triggering execution).
#
# Usage:
#   bash scripts/autocurrency/docs-pr.sh <release-spec-yaml>
#
# Arguments:
#   release-spec-yaml — Path to the release specification YAML file
#
# Required environment variables (set by the workflow):
#   GH_TOKEN          — GitHub App token for push/PR operations
#   SLACK_WEBHOOK_URL — Slack webhook URL (optional)

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
  echo "[Docs Update] ${display} ${version} ${platform^^}"
}

###############################################################################
# Main logic — only runs when executed directly (not when sourced for testing)
###############################################################################

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  return 0
fi

RELEASE_SPEC="${1:?Usage: docs-pr.sh <release-spec-yaml>}"

# -----------------------------------------------------------------
# Parse release spec
# -----------------------------------------------------------------
echo "Parsing release spec: ${RELEASE_SPEC}"
FRAMEWORK=$(yq '.framework' "$RELEASE_SPEC")
VERSION=$(yq '.version' "$RELEASE_SPEC")
PYTHON=$(yq '.python_version' "$RELEASE_SPEC")
CUDA=$(yq '.cuda_version' "$RELEASE_SPEC")
OS=$(yq '.os_version' "$RELEASE_SPEC")
PLATFORM=$(yq '.customer_type' "$RELEASE_SPEC")
DEVICE=$(yq '.device_type' "$RELEASE_SPEC")
PUBLIC_REGISTRY=$(yq '.public_registry' "$RELEASE_SPEC")

TRACKER="${REPO_ROOT}/${TRACKER_FILE:-".github/config/autocurrency-tracker.yml"}"

# Check if framework is defined in tracker config
if [[ "$(yq eval ".frameworks.${FRAMEWORK}" "$TRACKER")" == "null" ]]; then
  echo "${FRAMEWORK}: Not defined in tracker config. Skipping docs PR."
  exit 0
fi

# Build IMAGE_URI from parsed spec fields
IMAGE_URI="public.ecr.aws/deep-learning-containers/${FRAMEWORK}:${VERSION}-${DEVICE}-${PYTHON}-${CUDA}-${OS}-${PLATFORM}"

# Build upstream release URL from tracker config
GITHUB_REPO=$(yq eval ".frameworks.${FRAMEWORK}.github_repo" "$TRACKER")
TAG_PREFIX=$(yq eval ".frameworks.${FRAMEWORK}.tag_prefix // \"\"" "$TRACKER")
UPSTREAM_RELEASE_URL="https://github.com/${GITHUB_REPO}/releases/tag/${TAG_PREFIX}${VERSION}"

# -----------------------------------------------------------------
# Early exit: skip unsupported platforms
# -----------------------------------------------------------------
if [ "$PLATFORM" = "rayserve_ec2" ] || [ "$FRAMEWORK" = "xgboost" ]; then
  echo "${FRAMEWORK}: Platform '${PLATFORM}' is not supported for docs generation. Skipping."
  exit 0
fi

# -----------------------------------------------------------------
# Early exit: check if docs PR branch already exists
# -----------------------------------------------------------------
branch_name=$(generate_branch_name "$FRAMEWORK" "$VERSION" "$PLATFORM")
if git ls-remote --exit-code --heads origin "${branch_name}" &>/dev/null; then
  echo "${FRAMEWORK}: Branch '${branch_name}' already exists. PR likely in progress. Skipping."
  exit 0
fi

# -----------------------------------------------------------------
# Early exit: check if docs data file already exists
# -----------------------------------------------------------------
OUTPUT_FILE="${REPO_ROOT}/docs/src/data/${FRAMEWORK}/${VERSION}-${DEVICE}-${PLATFORM}.yml"
if [ -f "$OUTPUT_FILE" ]; then
  echo "${FRAMEWORK}: Docs file '${OUTPUT_FILE}' already exists. Skipping."
  exit 0
fi

# -----------------------------------------------------------------
# Step 1: Pull image and extract package versions
# -----------------------------------------------------------------
echo ""
echo "============================================================"
echo "Step 1: Pull image and extract package versions"
echo "============================================================"

echo "Pulling image: ${IMAGE_URI}"
docker pull "${IMAGE_URI}"

pip_count=$(yq eval ".frameworks.${FRAMEWORK}.docs_packages.pip | length" "$TRACKER")
system_count=$(yq eval ".frameworks.${FRAMEWORK}.docs_packages.system | length" "$TRACKER")

if [[ "${pip_count}" == "0" && "${system_count}" == "0" ]]; then
  echo "::warning::No docs_packages defined for '${FRAMEWORK}' in tracker config"
fi

FAILED_PACKAGES=()

# Extract pip package versions
for i in $(seq 0 $(( pip_count - 1 ))); do
  pkg_name=$(yq eval ".frameworks.${FRAMEWORK}.docs_packages.pip[$i].name" "$TRACKER")
  output_key=$(yq eval ".frameworks.${FRAMEWORK}.docs_packages.pip[$i].key // \"${pkg_name}\"" "$TRACKER")

  version=$(docker run --rm --entrypoint /bin/bash "$IMAGE_URI" -c "pip show ${pkg_name} 2>/dev/null" | grep "^Version:" | awk '{print $2}' | sed 's/+.*//') || true
  safe_key="${output_key//-/_}"
  if [ -n "$version" ]; then
    echo "  ✅ ${output_key}: ${version}"
    declare "PKG_${safe_key}=${version}"
  else
    echo "::warning::Failed to extract version for pip package '${pkg_name}' (key: ${output_key})"
    FAILED_PACKAGES+=("$output_key")
    declare "PKG_${safe_key}="
  fi
done

# Extract system package versions
for i in $(seq 0 $(( system_count - 1 ))); do
  sys_pkg=$(yq eval ".frameworks.${FRAMEWORK}.docs_packages.system[$i]" "$TRACKER")
  version=""

  case "$sys_pkg" in
    cuda)
      version=$(docker run --rm --entrypoint /bin/bash "$IMAGE_URI" -c "nvcc --version 2>/dev/null" | grep "release" | sed 's/.*release //' | sed 's/,.*//') || true
      ;;
    nccl)
      version=$(docker run --rm --entrypoint /bin/bash "$IMAGE_URI" -c "python3 -c \"import torch; v=torch.cuda.nccl.version(); print(f'{v[0]}.{v[1]}.{v[2]}')\" 2>/dev/null") || true
      ;;
    efa)
      version=$(docker run --rm --entrypoint /bin/bash "$IMAGE_URI" -c "cat /opt/amazon/efa_installed_packages 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+'") || true
      ;;
    cudnn)
      version=$(docker run --rm --entrypoint /bin/bash "$IMAGE_URI" -c "dpkg -l 2>/dev/null | grep 'libcudnn[0-9]*' | head -1 | awk '{print \$3}'" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+') || true
      ;;
    gdrcopy)
      version=$(docker run --rm --entrypoint /bin/bash "$IMAGE_URI" -c "dpkg -l 2>/dev/null" | grep gdrcopy | awk '{print $3}' | head -1) || true
      ;;
    *)
      echo "::warning::Unknown system package '${sys_pkg}', skipping"
      continue
      ;;
  esac
  if [ -n "$version" ]; then
    echo "  ✅ ${sys_pkg}: ${version}"
    declare "PKG_${sys_pkg}=${version}"
  else
    echo "::warning::Failed to extract version for system package '${sys_pkg}'"
    FAILED_PACKAGES+=("$sys_pkg")
    declare "PKG_${sys_pkg}="
  fi
done

FAILED_PACKAGES_STR=$(IFS=,; echo "${FAILED_PACKAGES[*]}")
if [ ${#FAILED_PACKAGES[@]} -gt 0 ]; then
  echo "::warning::Failed to extract versions for: ${FAILED_PACKAGES_STR}"
else
  echo "✅ All package versions extracted successfully"
fi

# -----------------------------------------------------------------
# Step 2: Generate docs data YAML file
# -----------------------------------------------------------------
echo ""
echo "============================================================"
echo "Step 2: Generate docs data YAML file"
echo "============================================================"

display_name=$(get_display_name "$FRAMEWORK")
major_minor=$(parse_major_minor "$VERSION")

tags=$(generate_tags "$VERSION" "$DEVICE" "$PYTHON" "$CUDA" "$OS" "$PLATFORM")
tag1=$(echo "$tags" | sed -n '1p')
tag2=$(echo "$tags" | sed -n '2p')
tag3=$(echo "$tags" | sed -n '3p')
tag4=$(echo "$tags" | sed -n '4p')

announcement=$(generate_announcement "$FRAMEWORK" "$VERSION" "$PLATFORM")

output_dir="$(dirname "$OUTPUT_FILE")"
mkdir -p "$output_dir"

# Build packages section dynamically from tracker config
packages_yaml=""

for i in $(seq 0 $(( pip_count - 1 ))); do
  pkg_name=$(yq eval ".frameworks.${FRAMEWORK}.docs_packages.pip[$i].name" "$TRACKER")
  output_key=$(yq eval ".frameworks.${FRAMEWORK}.docs_packages.pip[$i].key // \"${pkg_name}\"" "$TRACKER")
  env_var="PKG_${output_key//-/_}"
  packages_yaml="${packages_yaml}  ${output_key}: \"${!env_var:-}\"\n"
done

for i in $(seq 0 $(( system_count - 1 ))); do
  sys_pkg=$(yq eval ".frameworks.${FRAMEWORK}.docs_packages.system[$i]" "$TRACKER")
  env_var="PKG_${sys_pkg}"
  packages_yaml="${packages_yaml}  ${sys_pkg}: \"${!env_var:-}\"\n"
done

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

# -----------------------------------------------------------------
# Step 3: Create branch, commit, and open PR
# -----------------------------------------------------------------
echo ""
echo "============================================================"
echo "Step 3: Create branch, commit, and open PR"
echo "============================================================"

pr_title=$(generate_pr_title "$FRAMEWORK" "$VERSION" "$PLATFORM")

git config user.name "asimov-bot[bot]"
git config user.email "asimov-bot[bot]@users.noreply.github.com"

git checkout -B "$branch_name"
git add "$OUTPUT_FILE"
git commit -m "$pr_title"
git push --force origin "$branch_name"

pr_body=$(cat <<PRBODY
## Docs Update: ${display_name} ${VERSION} (${PLATFORM^^})

**Framework**: ${display_name}
**Version**: ${VERSION}
**Platform**: ${PLATFORM^^}
**Upstream Release**: [${GITHUB_REPO} ${TAG_PREFIX}${VERSION}](${UPSTREAM_RELEASE_URL})

### What to Review
- Verify the generated image tags
- Confirm package versions are present and correct
- Adjust the \`announcements\` section if needed

### Auto-generated by
- [reusable-release-image.yml](https://github.com/aws/deep-learning-containers/blob/main/.github/workflows/reusable-release-image.yml) — step4-docs-pr
PRBODY
)

if [ -n "${FAILED_PACKAGES_STR}" ]; then
  pr_body="${pr_body}

---

⚠️ **Warning: Some package versions could not be extracted:**

$(echo "$FAILED_PACKAGES_STR" | tr ',' '\n' | while read -r pkg; do echo "- \`${pkg}\`"; done)"
fi

echo "Creating PR..."
PR_URL=$(gh pr create --title "$pr_title" --body "$pr_body" --head "$branch_name" --base main)

echo "✅ PR URL: ${PR_URL}"

# -----------------------------------------------------------------
# Step 4: Send Slack notification
# -----------------------------------------------------------------
echo ""
echo "============================================================"
echo "Step 4: Send Slack notification"
echo "============================================================"

send_slack_notification \
  "${SLACK_WEBHOOK_URL:-}" \
  "docs_update" \
  "${FRAMEWORK}" \
  "${VERSION}" \
  "${PR_URL}" || true
