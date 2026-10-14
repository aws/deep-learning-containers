#!/usr/bin/env python3
"""Check that installed distributions match the image uv lock.

Every locked package must be installed at its locked version. Anything else in
the environment must be a source build (local path, VCS, or file install), so
no registry package can bypass the lock. Locked git sources must name the same
commits the Dockerfile built, as recorded in SOURCE_MANIFEST.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from importlib.metadata import distributions
from pathlib import Path

from packaging.utils import canonicalize_name
from packaging.version import Version


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "lock",
        nargs="?",
        default="/opt/aws/dlc/python/uv.lock",
        type=Path,
    )
    parser.add_argument(
        "--source-manifest",
        default="/opt/aws/dlc/SOURCE_MANIFEST",
        type=Path,
    )
    return parser.parse_args()


def read_lock(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    locked: dict[str, str] = {}
    git_commits: dict[str, str] = {}
    for package in tomllib.loads(path.read_text())["package"]:
        source = package.get("source", {})
        if {"virtual", "editable", "directory"} & source.keys():
            continue
        name = canonicalize_name(package["name"])
        locked[name] = package["version"]
        if "git" in source:
            git_commits[name] = source["git"].rpartition("#")[2]
    return locked, git_commits


def read_manifest(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator and key and value:
            values[canonicalize_name(key)] = value
    return values


def main() -> None:
    args = parse_args()
    locked, git_commits = read_lock(args.lock)
    manifest = read_manifest(args.source_manifest)
    installed: dict[str, str] = {}
    source_builds: dict[str, str] = {}
    unlocked_registry: list[str] = []
    for distribution in distributions():
        name = distribution.metadata.get("Name")
        if not name:
            continue
        name = canonicalize_name(name)
        installed[name] = distribution.version
        if name in locked:
            continue
        direct_url = distribution.read_text("direct_url.json")
        if direct_url:
            source_builds[name] = json.loads(direct_url)["url"]
        else:
            unlocked_registry.append(name)

    missing = sorted(locked.keys() - installed.keys())
    mismatched = {
        name: {"installed": installed[name], "locked": version}
        for name, version in sorted(locked.items())
        if name in installed and Version(installed[name]) != Version(version)
    }
    commit_mismatches = {
        name: {"built": manifest.get(name), "locked": commit}
        for name, commit in sorted(git_commits.items())
        if manifest.get(name) != commit
    }
    report = {
        "commit_mismatches": commit_mismatches,
        "locked": len(locked),
        "missing": missing,
        "mismatched": mismatched,
        "source_builds": dict(sorted(source_builds.items())),
        "unlocked_registry_installs": sorted(unlocked_registry),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if missing or mismatched or unlocked_registry or commit_mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
