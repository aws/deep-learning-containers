#!/usr/bin/env python3
"""Batch test-skip gate.

Usage:
    python3 gate.py --image-uri <ref> --repo-root <path> \
        --suite-map '{"sanity":"sanity","upstream":"sglang/upstream"}'
"""

import argparse
import json
import sys
from pathlib import Path

DEFAULT_PLATFORM = "linux/amd64"


def _load_helpers(repo_root):
    """Import the shared test-skip modules from the repo checkout."""
    scripts_dir = Path(repo_root) / "scripts" / "ci" / "image_test_skip"
    sys.path.insert(0, str(scripts_dir))
    import ci_images_store
    import hash_image_content
    import hash_suite_code

    return hash_image_content, hash_suite_code, ci_images_store


def compute_skips(repo_root, image_uri, suite_map, platform=DEFAULT_PLATFORM):
    """Return {test: skip_bool} for the skip-eligible tests in suite_map."""
    hash_image_content, hash_suite_code, store = _load_helpers(repo_root)
    
    eligible = [
        (test, suite)
        for test, suite in suite_map.items()
        if hash_suite_code.is_skip_eligible(repo_root, suite)
    ]
    if not eligible:
        return {}

    image_content_hash = hash_image_content.compute_image_content_hash(image_uri, platform=platform)

    # Several tests may share one suite; hash each distinct suite once.
    suite_code_hashes = {}
    for _, suite in eligible:
        if suite not in suite_code_hashes:
            suite_code_hashes[suite] = hash_suite_code.hash_suite_code(repo_root, suite)

    skippable = store.check_test_skip(image_content_hash, suite_code_hashes)
    return {test: (suite in skippable) for test, suite in eligible}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Batch test-skip gate.")
    parser.add_argument("--image-uri", required=True, help="Resolved image reference to hash.")
    parser.add_argument(
        "--suite-map", required=True, help="JSON object: test -> test-suites.yml key."
    )
    parser.add_argument("--repo-root", default=".", help="Repo checkout root.")
    parser.add_argument(
        "--platform", default=DEFAULT_PLATFORM, help=f"Platform (default {DEFAULT_PLATFORM})."
    )
    args = parser.parse_args(argv)

    try:
        suite_map = json.loads(args.suite_map)
        skips = compute_skips(args.repo_root, args.image_uri, suite_map, args.platform)
    except Exception as e:  # fail open — never let the gate suppress a real test
        print(f"::warning::test-skip gate failed ({e}); running all suites", file=sys.stderr)
        print("{}")
        return 0

    print(json.dumps(skips, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
