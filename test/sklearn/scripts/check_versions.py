"""Verify installed package versions match declared specifiers.

Runs inside the built sklearn container. Accepts either a requirements.txt or a
pyproject.toml (uv-managed images). For each PEP 508 requirement (`==`, `>=`,
`<`, `~=`, etc.), asserts the installed version satisfies the declared specifier.
For pyproject.toml, also enforces `[project].requires-python` against the
running interpreter.

Bare entries with no specifier (e.g. `certifi`) can't be verified — logged and
skipped. Unparsable lines are logged and skipped.

Usage: python3 check_versions.py <requirements.txt | pyproject.toml path>

Exits non-zero on any drift, with a summary of what changed.
"""

import importlib.metadata
import sys
import tomllib
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version


def parse_reqs(path):
    path = Path(path)
    if path.suffix == ".toml":
        return _parse_pyproject(path)
    return _parse_requirements_txt(path)


def _parse_pyproject(path):
    reqs, unparsable = [], []
    data = tomllib.loads(path.read_text())
    lines = list(data.get("project", {}).get("dependencies", []))
    lines += list(data.get("tool", {}).get("uv", {}).get("override-dependencies", []))
    for line in lines:
        try:
            reqs.append((Requirement(line), line))
        except InvalidRequirement:
            unparsable.append(line)
    return reqs, unparsable


def _parse_requirements_txt(path):
    reqs, unparsable = [], []
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


def check_python_version(path):
    """Enforce [project].requires-python from pyproject.toml against the
    running interpreter. Returns True if satisfied (or not declared), False
    if the running interpreter fails the declared spec.
    """
    path = Path(path)
    if path.suffix != ".toml":
        return True
    data = tomllib.loads(path.read_text())
    spec = data.get("project", {}).get("requires-python")
    if not spec:
        return True
    running = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if Version(running) not in SpecifierSet(spec):
        print(
            f"DRIFT: Python {running} does not satisfy requires-python={spec}",
            file=sys.stderr,
        )
        return False
    print(f"Python {running} satisfies requires-python={spec}.")
    return True


def installed_version(name):
    for candidate in (name, name.replace("-", "_"), name.replace("_", "-")):
        try:
            return importlib.metadata.version(candidate)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def main(path):
    py_ok = check_python_version(path)
    reqs, unparsable = parse_reqs(path)
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
    if unparsable:
        print(f"Skipped {len(unparsable)} unparsable lines:", file=sys.stderr)
        for line in unparsable:
            print(f"  {line}", file=sys.stderr)

    if not drift and not missing and py_ok:
        print(f"All {checked} constrained packages satisfy declared specifiers.")
        return

    for name, spec, act in drift:
        print(f"DRIFT: {name} declared{spec} but installed=={act}", file=sys.stderr)
    for name in missing:
        print(f"MISSING: {name} declared but not installed", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "usage: check_versions.py <requirements.txt | pyproject.toml path>",
            file=sys.stderr,
        )
        sys.exit(2)
    main(sys.argv[1])
