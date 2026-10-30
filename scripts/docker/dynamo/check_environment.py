#!/usr/bin/env python3
"""`uv pip check` for the Dynamo image's environment, allowing only the deliberate deviations.

The backend's pyproject.toml drops some dependencies with `tool.uv.exclude-dependencies` and
replaces some requirements with `tool.uv.override-dependencies`; uv's lock and sync honor both,
but `uv pip check` reads raw package metadata and reports an excluded package as missing and an
overridden one as a version conflict. Every other problem fails.

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
    overridden = {
        canonical(re.match(r"[A-Za-z0-9_.\-]+", r)[0]) for r in uv.get("override-dependencies", [])
    }

    result = subprocess.run(["uv", "pip", "check"], capture_output=True, text=True)
    output = (result.stdout + result.stderr).splitlines()
    if result.returncode == 0:
        print("uv pip check: no issues")
        return 0

    issues = [line for line in output if ISSUE.search(line)]
    if not issues:  # failed for a reason we don't understand: surface it
        print("\n".join(output), file=sys.stderr)
        return 1
    allowed = {}
    for line in issues:
        req = canonical(ISSUE.search(line)["req"])
        if "but it's not installed" in line and req in excluded:
            allowed[line] = "exclude-dependencies"
        elif "is installed" in line and req in overridden:
            allowed[line] = "override-dependencies"
    for line, why in allowed.items():
        print(f"allowed ({why}): {line}")
    problems = [line for line in issues if line not in allowed]
    for line in problems:
        print(f"ERROR: {line}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
