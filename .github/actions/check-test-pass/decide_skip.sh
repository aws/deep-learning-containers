#!/usr/bin/env bash
# Decide whether a test suite can be skipped for the image under test.
#
# Usage:
#   SUITE=<name> IMAGE_URI=<ref> PLATFORM=<os/arch> \
#     CI_IMAGES_TABLE_ACCOUNT_ID=<id> bash decide_skip.sh
#
# Output: writes skip=true|false, image-content-hash=<hash>, and
# suite-code-hash=<hash> to $GITHUB_OUTPUT
#
# Requires: python3, and scripts/ci/image_test_skip/{hash_suite_code,
# hash_image_content,ci_images_store}.py
set -uo pipefail

SCRIPTS="$(git rev-parse --show-toplevel)/scripts/ci/image_test_skip"

# Suites with skip_eligible=false are always run
ELIGIBLE=$(python3 "$SCRIPTS/hash_suite_code.py" --suite "$SUITE" --eligible-only)
if [[ "$ELIGIBLE" != "true" ]]; then
  echo "Suite '$SUITE' is not skip-eligible — running."
  {
    echo "skip=false"
    echo "image-content-hash="
    echo "suite-code-hash="
  } >>"$GITHUB_OUTPUT"
  exit 0
fi

# Compute suite_code_hash
SUITE_CODE_HASH=$(python3 "$SCRIPTS/hash_suite_code.py" --suite "$SUITE") || {
  echo "::warning::suite_code_hash failed for '$SUITE' — running."
  echo "skip=false" >>"$GITHUB_OUTPUT"
  exit 0
}
echo "suite-code-hash=$SUITE_CODE_HASH" >>"$GITHUB_OUTPUT"

# Compute image_content_hash from the registry config read (no layer pull)
IMAGE_CONTENT_HASH=$(python3 "$SCRIPTS/hash_image_content.py" \
  --image-uri "$IMAGE_URI" --platform "$PLATFORM") || {
  echo "::warning::image_content_hash failed for '$IMAGE_URI' — running."
  echo "skip=false" >>"$GITHUB_OUTPUT"
  exit 0
}
echo "image-content-hash=$IMAGE_CONTENT_HASH" >>"$GITHUB_OUTPUT"

if python3 "$SCRIPTS/ci_images_store.py" check \
  --image-content-hash "$IMAGE_CONTENT_HASH" \
  --suite "$SUITE" \
  --suite-code-hash "$SUITE_CODE_HASH"; then
  echo "skip=true" >>"$GITHUB_OUTPUT"
  {
    echo "### ⏭️ Test skipped: cached PASS"
    echo "Suite \`$SUITE\` already passed on this image content + test code."
    echo ""
    echo "- image_content_hash: \`$IMAGE_CONTENT_HASH\`"
    echo "- suite_code_hash: \`$SUITE_CODE_HASH\`"
  } >>"$GITHUB_STEP_SUMMARY"
else
  echo "skip=false" >>"$GITHUB_OUTPUT"
fi
