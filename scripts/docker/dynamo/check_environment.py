#!/usr/bin/env python3
"""`uv pip check` for the Dynamo image's environment, allowing only the deliberate exclusions.

The backend's pyproject.toml drops some dependencies with `tool.uv.exclude-dependencies`; uv's
lock and sync honor that, but `uv pip check` reads raw package metadata and reports them as
missing. Every other problem (a conflict, or any other missing package) fails.

Usage: check_environment.py <path/to/pyproject.toml>
"""

import re
import subprocess
import sys
import tomllib

ISSUE = re.compile(r"The package `(?P<pkg>[^`]+)` requires `(?P<req>[A-Za-z0-9_.\-]+)")


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def main(pyproject):
    with open(pyproject, "rb") as f:
        uv = tomllib.load(f).get("tool", {}).get("uv", {})
    excluded = {canonical(n) for n in uv.get("exclude-dependencies", [])}

    result = subprocess.run(["uv", "pip", "check"], capture_output=True, text=True)
    output = (result.stdout + result.stderr).splitlines()
    if result.returncode == 0:
        print("uv pip check: no issues")
        return 0

    issues = [line for line in output if ISSUE.search(line)]
    if not issues:  # failed for a reason we don't understand: surface it
        print("\n".join(output), file=sys.stderr)
        return 1
    allowed = [
        line
        for line in issues
        if "but it's not installed" in line and canonical(ISSUE.search(line)["req"]) in excluded
    ]
    for line in allowed:
        print(f"allowed (exclude-dependencies): {line}")
    problems = [line for line in issues if line not in allowed]
    for line in problems:
        print(f"ERROR: {line}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
