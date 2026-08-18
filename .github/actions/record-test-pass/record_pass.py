#!/usr/bin/env python3
"""Write a PASS row for a fully-passed suite.

Inputs come from the environment (set by the composite action).
"""

import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ci" / "test_skip"))

import hash_suite_code  # noqa: E402
import ci_images_db  # noqa: E402


def _write_summary(lines):
    """Append markdown lines to the GitHub step summary, if running in Actions."""
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if not path:
        return
    with open(path, "a") as f:
        f.write("\n".join(lines) + "\n")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    suite = os.environ["SUITE"]
    image_content_hash = os.environ.get("IMAGE_CONTENT_HASH", "")
    suite_code_hash = os.environ.get("SUITE_CODE_HASH", "")
    ci_image_tag = os.environ.get("CI_IMAGE_TAG", "")

    # Suites with skip_eligible=false are always run — never record them.
    try:
        eligible = hash_suite_code.is_skip_eligible(REPO_ROOT, suite)
    except Exception as e:
        print(f"::warning::could not read suite config for {suite!r} ({e}) — not recording.")
        return 0
    if not eligible:
        print(f"Suite {suite!r} is not skip-eligible — not recording a PASS row.")
        return 0

    if not image_content_hash or not suite_code_hash:
        print(
            f"::warning::Skipping cache write for {suite!r}: empty hash "
            f"(image={image_content_hash!r} suite_code={suite_code_hash!r}) — suite still passed."
        )
        return 0

    try:
        ci_images_db.record_test_pass(
            image_content_hash, suite, suite_code_hash, ci_image_tag=ci_image_tag
        )
    except Exception as e:
        print(
            f"::warning::Failed to record PASS for {suite!r} (cache write): {e} "
            "— suite still passed."
        )
        return 0

    _write_summary(
        [
            "### ✅ Recorded test PASS",
            f"Suite `{suite}` passed and was cached.",
            "",
            f"- image_content_hash: `{image_content_hash}`",
            f"- suite_code_hash: `{suite_code_hash}`",
        ]
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
