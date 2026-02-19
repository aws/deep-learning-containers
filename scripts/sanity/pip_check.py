#!/usr/bin/env python3
"""Pip check with 3-level allowlist support.

Usage: python3 pip_check.py [--image-uri IMAGE_URI] [--allowlist-dir DIR]

Allowlist resolution (all optional, merged in order):
  1. <allowlist-dir>/pip_check.json           (global)
  2. <allowlist-dir>/<framework>/pip_check.json (framework)
  3. <allowlist-dir>/<framework>/<tag>.json     (image-specific)

Each file: [{"pattern": "regex", "reason": "why"}]
"""

import argparse
import json
import os
import re
import subprocess
import sys


def load_allowlist(allowlist_dir, framework=None, image_tag=None):
    entries = []
    paths = [os.path.join(allowlist_dir, "pip_check.json")]
    if framework:
        paths.append(os.path.join(allowlist_dir, framework, "pip_check.json"))
    if framework and image_tag:
        paths.append(os.path.join(allowlist_dir, framework, f"{image_tag}.json"))
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
    parser.add_argument("--image-uri", default="")
    parser.add_argument("--allowlist-dir", default="test/data/allowlists")
    args = parser.parse_args()

    framework = get_framework(args.image_uri)
    tag = args.image_uri.split(":")[-1] if ":" in args.image_uri else None
    patterns = load_allowlist(args.allowlist_dir, framework, tag)

    result = subprocess.run(["pip", "check"], capture_output=True, text=True)
    if result.returncode == 0:
        print("pip check passed")
        return 0

    failures = []
    for line in result.stdout.strip().splitlines():
        if not any(re.search(p, line) for p in patterns):
            failures.append(line)

    if failures:
        print("pip check found broken dependencies:")
        for f in failures:
            print(f"  {f}")
        return 1

    print("pip check passed (all issues in allowlist)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
