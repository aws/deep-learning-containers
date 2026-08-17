#!/usr/bin/env bash
# Record a PASS row in the dlc-ci-images table for a fully-passed test suite.
#
# Usage:
#   SUITE=<name> IMAGE_CONTENT_HASH=<hash> SUITE_CODE_HASH=<hash> \
#     CI_IMAGES_TABLE_ACCOUNT_ID=<id> bash record_pass.sh
#
# Output: writes the PASS row to DynamoDB and a summary to $GITHUB_STEP_SUMMARY
#
# Requires: python3, and scripts/ci/test_skip/test_skip_db.py
set -euo pipefail

SCRIPTS="$(git rev-parse --show-toplevel)/scripts/ci/test_skip"

# Suites with skip_eligible=false are always run. Don't write them to the store.
ELIGIBLE=$(python3 "$SCRIPTS/hash_suite_code.py" --suite "$SUITE" --eligible-only) || ELIGIBLE=""
if [[ "$ELIGIBLE" != "true" ]]; then
  echo "Suite '$SUITE' is not skip-eligible — not recording a PASS row."
  exit 0
fi

if [[ -z "$IMAGE_CONTENT_HASH" || -z "$SUITE_CODE_HASH" ]]; then
  echo "::warning::Skipping cache write for '$SUITE': empty hash" \
       "(image='$IMAGE_CONTENT_HASH' suite_code='$SUITE_CODE_HASH')" \
       "— suite still passed."
  exit 0
fi

if python3 "$SCRIPTS/test_skip_db.py" record \
  --image-content-hash "$IMAGE_CONTENT_HASH" \
  --suite "$SUITE" \
  --suite-code-hash "$SUITE_CODE_HASH" \
  --ci-image-tag "${CI_IMAGE_TAG:-}"; then
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
