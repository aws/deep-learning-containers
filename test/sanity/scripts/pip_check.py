#!/usr/bin/env python3
"""Pip check with 3-level allowlist support.

Usage: python3 pip_check.py --image-uri IMAGE_URI --allowlist-dir DIR

Allowlist resolution (all optional, merged in order):
  1. <allowlist-dir>/pip_check.json           (global)
  2. <allowlist-dir>/<framework>/pip_check.json (framework)
  3. <allowlist-dir>/<framework>/<tag>.json     (image-specific)

Each file: [{"pattern": "regex", "reason": "why"}]
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys

from test_utils.logger import ColoredFormatter

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(ColoredFormatter())
LOGGER.addHandler(_handler)


def load_allowlist(allowlist_dir, framework=None, image_tag=None):
    entries = []
    paths = [os.path.join(allowlist_dir, "pip_check.json")]
    if framework:
        paths.append(os.path.join(allowlist_dir, framework, "pip_check.json"))
    if framework and image_tag:
        framework_dir = os.path.join(allowlist_dir, framework)
        if os.path.isdir(framework_dir):
            for fname in os.listdir(framework_dir):
                if fname == "pip_check.json" or not fname.endswith(".json"):
                    continue
                prefix = fname[:-5]
                if image_tag.startswith(prefix):
                    paths.append(os.path.join(framework_dir, fname))
    for path in paths:
        if os.path.exists(path):
            with open(path) as f:
                entries.extend(json.load(f))
    return [e["pattern"] for e in entries]


def get_framework(image_uri):
    for name in ["vllm", "sglang", "pytorch", "tensorflow"]:
        if name in (image_uri or ""):
            return name
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--allowlist-dir", required=True)
    args = parser.parse_args()

    framework = get_framework(args.image_uri)
    tag = args.image_uri.split(":")[-1] if ":" in args.image_uri else None
    patterns = load_allowlist(args.allowlist_dir, framework, tag)

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
