#!/usr/bin/env python3
"""Batch test-pass check.

Checks for skip-eligible suites and returns which ones already passed for
the current image content hash.

Usage:
    python3 check_test_pass.py --image-uri <ref> --repo-root <path> \
        --suites '["sanity", "sglang/upstream", "sglang/model"]'
"""

import argparse
import json
import sys
from pathlib import Path

DEFAULT_PLATFORM = "linux/amd64"


def _load_helpers(repo_root):
    """Import the shared test-skip modules from the repo checkout."""
    scripts_dir = Path(repo_root) / "scripts" / "ci" / "test_skip"
    sys.path.insert(0, str(scripts_dir))
    import test_skip_db
    import hash_image_content
    import hash_suite_code

    return hash_image_content, hash_suite_code, test_skip_db


def compute_skips(repo_root, image_uri, suites, platform=DEFAULT_PLATFORM):
    """Return (skips, image_content_hash, suite_code_hashes) for the eligible suites."""
    hash_image_content, hash_suite_code, store = _load_helpers(repo_root)
    eligible = [s for s in dict.fromkeys(suites) if hash_suite_code.is_skip_eligible(repo_root, s)]
    # Skip the image hash + DB call and run everything.
    if not eligible:
        return {}, "", {}

    image_content_hash = hash_image_content.compute_image_content_hash(image_uri, platform=platform)

    suite_code_hashes = {}
    for s in eligible:
        try:
            suite_code_hashes[s] = hash_suite_code.hash_suite_code(repo_root, s)
        except (KeyError, ValueError) as e:
            print(f"::warning::skipping test-pass check for {s!r} ({e})", file=sys.stderr)

    skippable = store.check_test_pass(image_content_hash, suite_code_hashes) if suite_code_hashes else set()
    skips = {s: (s in skippable) for s in suite_code_hashes}
    return skips, image_content_hash, suite_code_hashes


def main(argv=None):
    parser = argparse.ArgumentParser(description="Batch test-pass check.")
    parser.add_argument("--image-uri", required=True, help="Resolved image reference to hash.")
    parser.add_argument("--suites", required=True, help="JSON array of test-suites.yml keys to check.")
    parser.add_argument("--repo-root", default=".", help="Repo checkout root.")
    parser.add_argument("--platform", default=DEFAULT_PLATFORM, help=f"Platform (default {DEFAULT_PLATFORM}).")
    args = parser.parse_args(argv)

    try:
        suites = json.loads(args.suites)
        if not isinstance(suites, list):
            raise ValueError(f"--suites must be a JSON array, got {type(suites).__name__}")
        skips, image_content_hash, suite_code_hashes = compute_skips(
            args.repo_root, args.image_uri, suites, args.platform
        )
    except Exception as e:
        print(f"::warning::test-pass check failed ({e}); running all suites", file=sys.stderr)
        print(json.dumps({"skips": {}, "image_content_hash": "", "suite_code_hashes": {}}, separators=(",", ":")))
        return 0

    print(json.dumps(
        {"skips": skips, "image_content_hash": image_content_hash, "suite_code_hashes": suite_code_hashes},
        separators=(",", ":"),
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
