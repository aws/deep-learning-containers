#!/usr/bin/env python3
# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Pip check with 3-level allowlist support.

Usage: python3 pip_check.py --framework FRAMEWORK [--framework-version VERSION] [--allowlist-dir DIR]

Allowlist resolution (merged in order):
  1. <allowlist-dir>/pip_check.json                          (global)
  2. <allowlist-dir>/<framework>/pip_check.json              (framework)
  3. <allowlist-dir>/<framework>/<framework>-<version>.json  (version-specific)

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

import test  # noqa: F401

LOGGER = logging.getLogger("test").getChild("pip_check")
LOGGER.setLevel(logging.INFO)

# Debug logging setup
print(f"[DEBUG] test module location: {test.__file__}", flush=True)
test_dir = os.path.dirname(test.__file__)
print(f"[DEBUG] ls {test_dir}:", flush=True)
for entry in sorted(os.listdir(test_dir)):
    print(f"  {entry}", flush=True)
print(f"[DEBUG] LOGGER name: {LOGGER.name}", flush=True)
print(f"[DEBUG] LOGGER level: {LOGGER.level}", flush=True)
print(f"[DEBUG] LOGGER handlers: {LOGGER.handlers}", flush=True)
print(f"[DEBUG] LOGGER parent: {LOGGER.parent}", flush=True)
print(
    f"[DEBUG] LOGGER parent handlers: {LOGGER.parent.handlers if LOGGER.parent else 'None'}",
    flush=True,
)
print(f"[DEBUG] LOGGER effective level: {LOGGER.getEffectiveLevel()}", flush=True)


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
    print(f"[DEBUG] pip check returncode: {result.returncode}", flush=True)
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
