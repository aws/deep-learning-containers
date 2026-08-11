#!/usr/bin/env bash
# Record a PASS row in the dlc-ci-images table for a fully-passed test suite.
#
# Usage:
#   SUITE=<name> IMAGE_CONTENT_HASH=<hash> SUITE_CODE_HASH=<hash> \
#     CI_IMAGES_TABLE_ACCOUNT_ID=<id> bash record_pass.sh
#
# Output: writes the PASS row to DynamoDB and a summary to $GITHUB_STEP_SUMMARY
#
# Requires: python3, and scripts/ci/image_test_skip/ci_images_store.py
set -euo pipefail

SCRIPTS="$(git rev-parse --show-toplevel)/scripts/ci/image_test_skip"

if [[ -z "$IMAGE_CONTENT_HASH" || -z "$SUITE_CODE_HASH" ]]; then
  echo "::error::image has empty hash (image='$IMAGE_CONTENT_HASH' suite_code='$SUITE_CODE_HASH')."
  exit 1
fi

if python3 "$SCRIPTS/ci_images_store.py" record \
  --image-content-hash "$IMAGE_CONTENT_HASH" \
  --suite "$SUITE" \
  --suite-code-hash "$SUITE_CODE_HASH"; then
  {
    echo "### ✅ Recorded test PASS"
    echo "Suite \`$SUITE\` passed and was cached."
    echo ""
    echo "- image_content_hash: \`$IMAGE_CONTENT_HASH\`"
    echo "- suite_code_hash: \`$SUITE_CODE_HASH\`"
  } >>"$GITHUB_STEP_SUMMARY"
else
  echo "::warning::Failed to record PASS for '$SUITE' (cache write) — suite still passed."
fi
