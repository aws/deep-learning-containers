"""Verify installed package versions match declared specifiers in requirements.txt.

Runs inside the built sklearn container. Reads a requirements.txt (copied in by
the caller), parses each entry as a PEP 508 requirement, and asserts the
installed version satisfies the declared specifier (==, >=, <, ~=, etc.).

Bare entries with no specifier (e.g. `certifi`) can't be verified — logged and
skipped. Unparseable lines are logged and skipped.

Usage: python3 check_versions.py <requirements.txt path>

Exits non-zero on any drift, with a summary of what changed.
"""

import importlib.metadata
import sys

from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import Version


def parse_reqs(path):
    reqs = []
    unparseable = []
    with open(path) as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            try:
                reqs.append((Requirement(line), line))
            except InvalidRequirement:
                unparseable.append(line)
    return reqs, unparseable


def installed_version(name):
    for candidate in (name, name.replace("-", "_"), name.replace("_", "-")):
        try:
            return importlib.metadata.version(candidate)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def main(path):
    reqs, unparseable = parse_reqs(path)
    if not reqs:
        print(f"No requirements found in {path}", file=sys.stderr)
        sys.exit(1)

    drift = []
    missing = []
    unconstrained = []
    checked = 0

    for req, raw in reqs:
        actual = installed_version(req.name)
        if actual is None:
            missing.append(req.name)
            continue
        if not req.specifier:
            unconstrained.append(raw)
            continue
        checked += 1
        if Version(actual) not in req.specifier:
            drift.append((req.name, str(req.specifier), actual))

    print(f"Checked {checked} constrained packages against declared specifiers.")
    if unconstrained:
        print(f"Skipped {len(unconstrained)} unconstrained entries (no version specifier):")
        for line in unconstrained:
            print(f"  {line}")
    if unparseable:
        print(f"Skipped {len(unparseable)} unparseable lines:", file=sys.stderr)
        for line in unparseable:
            print(f"  {line}", file=sys.stderr)

    if not drift and not missing:
        print(f"All {checked} constrained packages satisfy declared specifiers.")
        return

    for name, spec, act in drift:
        print(f"DRIFT: {name} declared{spec} but installed=={act}", file=sys.stderr)
    for name in missing:
        print(f"MISSING: {name} declared but not installed", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: check_versions.py <requirements.txt path>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
