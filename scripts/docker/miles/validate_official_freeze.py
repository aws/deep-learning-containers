#!/usr/bin/env python3
"""Compare installed distributions with the Miles official version freeze."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import distributions
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "freeze",
        nargs="?",
        default="/opt/aws/dlc/PYTHON_OFFICIAL_FREEZE",
        type=Path,
    )
    parser.add_argument(
        "--allow-version",
        action="append",
        default=[],
        metavar="NAME=VERSION",
        help="Allow an intentional installed-version replacement.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Print differences without returning a failure.",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Also fail when a frozen optional package is not installed.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="List frozen packages that are not installed.",
    )
    return parser.parse_args()


def parse_allowed(values: list[str]) -> dict[str, str]:
    allowed: dict[str, str] = {}
    for value in values:
        name, separator, version = value.partition("=")
        if not separator or not name or not version:
            raise ValueError(f"invalid --allow-version value: {value!r}")
        allowed[canonicalize_name(name)] = version
    return allowed


def read_expected(path: Path) -> dict[str, str]:
    expected: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        specifiers = list(requirement.specifier)
        if len(specifiers) != 1 or specifiers[0].operator != "==":
            raise ValueError(
                f"{path}:{line_number}: expected one exact version: {line}"
            )
        name = canonicalize_name(requirement.name)
        version = specifiers[0].version
        previous = expected.setdefault(name, version)
        if previous != version:
            raise ValueError(
                f"{path}:{line_number}: conflicting versions for {name}: "
                f"{previous} and {version}"
            )
    return expected


def read_installed() -> dict[str, set[str]]:
    installed: dict[str, set[str]] = {}
    for distribution in distributions():
        name = distribution.metadata.get("Name")
        if name:
            installed.setdefault(canonicalize_name(name), set()).add(
                distribution.version
            )
    return installed


def main() -> None:
    args = parse_args()
    allowed = parse_allowed(args.allow_version)
    expected = read_expected(args.freeze)
    installed = read_installed()

    missing = {
        name: version for name, version in expected.items() if name not in installed
    }
    mismatched = {
        name: {
            "expected": expected[name],
            "installed": sorted(installed[name]),
        }
        for name in expected
        if name in installed
        and not any(
            Requirement(f"{name}=={expected[name]}").specifier.contains(
                version, prereleases=True
            )
            for version in installed[name]
        )
    }
    accepted = {
        name: details
        for name, details in list(mismatched.items())
        if allowed.get(name) in details["installed"]
    }
    for name in accepted:
        del mismatched[name]

    report: dict[str, object] = {
        "accepted_replacements": accepted,
        "checked": len(expected),
        "installed_and_matched": len(expected) - len(missing) - len(mismatched),
        "mismatched": mismatched,
        "not_installed": len(missing),
    }
    if args.verbose or args.require_all:
        report["not_installed_packages"] = missing
    print(json.dumps(report, indent=2, sort_keys=True))

    failed = bool(mismatched) or (args.require_all and bool(missing))
    if failed and not args.report_only:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
