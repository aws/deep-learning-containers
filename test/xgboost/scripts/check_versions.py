"""Verify installed package versions match the versions declared for the image.

Runs inside the built xgboost container. Reads a declared-dependency file
(copied in by the caller) and asserts the installed version of each package
satisfies the declared specifier (==, >=, <, ~=, etc.).

Two declaration formats are supported, chosen by file extension:
  - requirements.txt  — one PEP 508 requirement per line (used by 3.0-5).
  - pyproject.toml     — reads [project].dependencies (used by 3.2.0+).

Bare entries with no specifier (e.g. `certifi`) can't be verified — logged and
skipped. Unparsable lines are logged and skipped.

Usage: python3 check_versions.py <requirements.txt | pyproject.toml path>

Exits non-zero on any drift, with a summary of what changed.
"""

import importlib.metadata
import sys

from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import Version

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - fallback for older interpreters
    tomllib = None


def parse_requirements_txt(path):
    reqs = []
    unparsable = []
    with open(path) as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            try:
                reqs.append((Requirement(line), line))
            except InvalidRequirement:
                unparsable.append(line)
    return reqs, unparsable


def parse_pyproject(path):
    if tomllib is None:
        print("tomllib unavailable; cannot parse pyproject.toml", file=sys.stderr)
        sys.exit(1)
    with open(path, "rb") as f:
        data = tomllib.load(f)
    entries = data.get("project", {}).get("dependencies", [])
    reqs = []
    unparsable = []
    for line in entries:
        try:
            reqs.append((Requirement(line), line))
        except InvalidRequirement:
            unparsable.append(line)
    return reqs, unparsable


def parse_declared(path):
    if path.endswith(".toml"):
        return parse_pyproject(path)
    return parse_requirements_txt(path)


def installed_version(name):
    for candidate in (name, name.replace("-", "_"), name.replace("_", "-")):
        try:
            return importlib.metadata.version(candidate)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def main(path):
    reqs, unparsable = parse_declared(path)
    if not reqs:
        print(f"No dependencies found in {path}", file=sys.stderr)
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
    if unparsable:
        print(f"Skipped {len(unparsable)} unparsable lines:", file=sys.stderr)
        for line in unparsable:
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
        print("usage: check_versions.py <requirements.txt | pyproject.toml path>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
