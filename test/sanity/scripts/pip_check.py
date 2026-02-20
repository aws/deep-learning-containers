#!/usr/bin/env python3
"""Pip check with 3-level allowlist support.

Usage: python3 pip_check.py --image-uri IMAGE_URI --allowlist-dir DIR

Allowlist resolution (all optional, merged in order):
  1. <allowlist-dir>/pip_check.json           (global)
  2. <allowlist-dir>/<framework>/pip_check.json (framework)
  3. <allowlist-dir>/<framework>/<tag>.json     (image-specific)

Default allowlist-dir: test/data/pipcheck_allowlist
Each file: [{"pattern": "regex", "reason": "why"}]
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys

import test  # noqa: F401 — triggers colored logging setup

LOGGER = logging.getLogger(__name__)


def load_allowlist(allowlist_dir, framework=None, framework_version=None):
    entries = []
    paths = [os.path.join(allowlist_dir, "pip_check.json")]
    if framework:
        paths.append(os.path.join(allowlist_dir, framework, "pip_check.json"))
        if framework_version:
            paths.append(
                os.path.join(allowlist_dir, framework, f"{framework}-{framework_version}.json")
            )
    for path in paths:
        if os.path.exists(path):
            with open(path) as f:
                entries.extend(json.load(f))
    return [e["pattern"] for e in entries]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", required=True)
    parser.add_argument("--framework-version", default="")
    parser.add_argument(
        "--allowlist-dir",
        default="test/data/pipcheck_allowlist",
        help="Path to pipcheck_allowlist directory",
    )
    args = parser.parse_args()

    patterns = load_allowlist(args.allowlist_dir, args.framework, args.framework_version)

    result = subprocess.run(["pip", "check"], capture_output=True, text=True)
    if result.returncode == 0:
        LOGGER.info("pip check passed")
        return 0

    failures = []
    for line in result.stdout.strip().splitlines():
        if not any(re.search(p, line) for p in patterns):
            failures.append(line)

    if failures:
        LOGGER.error("pip check found broken dependencies:")
        for f in failures:
            LOGGER.error(f"  {f}")
        return 1

    LOGGER.info("pip check passed (all issues in allowlist)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
