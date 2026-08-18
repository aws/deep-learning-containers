#!/usr/bin/env python3
"""Checks which skip-eligible suites already passed.

For every skip-eligible suite it computes the image content hash + suite code
hash and queries the dlc-ci-images table in a single batch. Inputs come from the
environment (set by the composite action).
"""

import json
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ci" / "test_skip"))

import hash_image_content  # noqa: E402
import hash_suite_code  # noqa: E402
import test_skip_db  # noqa: E402

DEFAULT_PLATFORM = "linux/amd64"


def compute_skips(image_uri, suites, platform=DEFAULT_PLATFORM):
    """Return (skips, image_content_hash, suite_code_hashes) for the eligible suites."""
    eligible = [s for s in dict.fromkeys(suites) if hash_suite_code.is_skip_eligible(REPO_ROOT, s)]
    # Skip the image hash + DB call and run everything.
    if not eligible:
        return {}, "", {}

    image_content_hash = hash_image_content.compute_image_content_hash(image_uri, platform=platform)

    suite_code_hashes = {}
    for s in eligible:
        try:
            suite_code_hashes[s] = hash_suite_code.hash_suite_code(REPO_ROOT, s)
        except (KeyError, ValueError) as e:
            print(f"::warning::skipping test-pass check for {s!r} ({e})", file=sys.stderr)

    skippable = (
        test_skip_db.check_test_pass(image_content_hash, suite_code_hashes)
        if suite_code_hashes
        else set()
    )
    skips = {s: (s in skippable) for s in suite_code_hashes}
    return skips, image_content_hash, suite_code_hashes


def _emit_outputs(skips, image_content_hash, suite_code_hashes):
    """Write the action outputs to $GITHUB_OUTPUT."""
    github_output = os.getenv("GITHUB_OUTPUT")
    if not github_output:
        return
    lines = [
        f"skips={json.dumps(skips, separators=(',', ':'))}",
        f"image-content-hash={image_content_hash}",
        f"suite-code-hashes={json.dumps(suite_code_hashes, separators=(',', ':'))}",
    ]
    with open(github_output, "a") as f:
        f.write("\n".join(lines) + "\n")


def _write_summary(lines):
    """Append markdown lines to the GitHub step summary, if running in Actions."""
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if not path:
        return
    with open(path, "a") as f:
        f.write("\n".join(lines) + "\n")


def _report_cache_hits(skips):
    """Summarize the cache hits in the step summary + a notice annotation."""
    hits = [suite for suite, skip in skips.items() if skip]
    if hits:
        _write_summary(
            [
                "### ✅ Test-pass cache hit",
                "The following suites will be skipped (no test runner provisioned):",
                *[f"- {h}" for h in hits],
            ]
        )
        print(f"::notice title=✅ Test-skip cache hits::{','.join(hits)}")
    else:
        _write_summary(
            [
                "### Test-skip cache check",
                "No cache hits — all checked suites will run.",
            ]
        )


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    suites_raw = os.environ["SUITES"]
    image_uri = os.environ.get("IMAGE_URI", "")

    try:
        suites = json.loads(suites_raw)
        if not isinstance(suites, list):
            raise ValueError(f"SUITES must be a JSON array, got {type(suites).__name__}")
        skips, image_content_hash, suite_code_hashes = compute_skips(image_uri, suites)
    except Exception as e:
        print(f"::warning::test-pass check failed ({e}); running all suites", file=sys.stderr)
        skips, image_content_hash, suite_code_hashes = {}, "", {}

    _emit_outputs(skips, image_content_hash, suite_code_hashes)
    _report_cache_hits(skips)
    return 0


if __name__ == "__main__":
    sys.exit(main())
