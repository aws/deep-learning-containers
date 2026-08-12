#!/usr/bin/env python3
"""Compute suite_code_hash for a test suite.

suite_code_hash is sha256 over the file set from .github/config/test-suites.yml —
each file's repo-relative path plus its bytes, sorted by path so the result is
order-independent. Editing, adding, moving, or removing a captured file changes
the hash.

Usage:
    python3 hash_suite_code.py --suite pytorch/single_gpu
    python3 hash_suite_code.py --suite sanity --repo-root /path/to/repo
"""

import argparse
import hashlib
import sys
from pathlib import Path

import yaml

# scripts/ci/image_test_skip/hash_suite_code.py -> repo root is three parents up.
DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_REL = ".github/config/test-suites.yml"


def load_config(repo_root):
    """Load the test-skip suite config as a dict keyed by suite name."""
    config_path = Path(repo_root) / CONFIG_REL
    data = yaml.safe_load(config_path.read_text())
    return data.get("suites", {})


def resolve_files(repo_root, code_paths):
    """Resolve code_paths globs to a sorted, de-duplicated list of existing files.
    Paths are specified relative to repo_root.
    """
    repo_root = Path(repo_root)
    matched = set()
    for pattern in code_paths:
        # pathlib's trailing `**` matches directories only; `**/*` is what
        # captures files recursively.
        if pattern.endswith("/**"):
            pattern = pattern + "/*"
        for path in repo_root.glob(pattern):
            if path.is_file():
                matched.add(path)
    return sorted(matched, key=lambda p: p.relative_to(repo_root).as_posix())


def hash_suite_code(repo_root, suite):
    """Return the suite_code_hash (``sha256:<hex>``) for one suite.

    Raises KeyError if the suite is not in the config.
    """
    suites = load_config(repo_root)
    if suite not in suites:
        raise KeyError(f"unknown suite: {suite!r} (not in {CONFIG_REL})")
    code_paths = suites[suite].get("code_paths", [])
    files = resolve_files(repo_root, code_paths)

    digest = hashlib.sha256()
    repo_root = Path(repo_root)
    for path in files:
        rel = path.relative_to(repo_root).as_posix()
        # Bind path to content so a rename (same bytes, new path) is a real change.
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def is_skip_eligible(repo_root, suite):
    """Return True iff the suite exists and is marked skip_eligible in the config.
    An unknown suite is treated as not eligible.
    """
    suite_cfg = load_config(repo_root).get(suite)
    return bool(suite_cfg and suite_cfg.get("skip_eligible"))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compute suite_code_hash for a test suite.")
    parser.add_argument("--suite", required=True, help="Suite name (key in test-suites.yml).")
    parser.add_argument(
        "--repo-root",
        default=str(DEFAULT_REPO_ROOT),
        help="Repo root (defaults to the checkout containing this script).",
    )
    parser.add_argument(
        "--eligible-only",
        action="store_true",
        help="Print skip_eligible (true|false) for the suite instead of its hash.",
    )
    args = parser.parse_args(argv)
    if args.eligible_only:
        print("true" if is_skip_eligible(args.repo_root, args.suite) else "false")
        return 0
    try:
        print(hash_suite_code(args.repo_root, args.suite))
    except KeyError as e:
        print(str(e), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
