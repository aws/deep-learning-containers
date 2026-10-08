#!/usr/bin/env python3
"""Validate selected Miles wheel artifacts against the official release."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SHA256 = re.compile(r"^[0-9a-f]{64}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--official",
        type=Path,
        default=Path("/opt/aws/dlc/WHEEL_ARTIFACTS_OFFICIAL"),
    )
    parser.add_argument(
        "--selected",
        type=Path,
        default=Path("/opt/aws/dlc/WHEEL_ARTIFACTS_SELECTED"),
    )
    parser.add_argument(
        "--deviations",
        type=Path,
        default=Path("/opt/aws/dlc/WHEEL_ARTIFACT_DEVIATIONS.json"),
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="NAME=COMMIT",
    )
    parser.add_argument("--source-manifest", type=Path)
    return parser.parse_args()


def read_hashes(path: Path) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        digest, separator, filename = line.partition("  ")
        filename = filename.lstrip("*")
        if not separator or not SHA256.fullmatch(digest) or not filename:
            raise ValueError(f"{path}:{line_number}: invalid sha256 manifest line")
        if filename in artifacts:
            raise ValueError(f"{path}:{line_number}: duplicate artifact {filename}")
        artifacts[filename] = digest
    return artifacts


def read_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        key, separator, value = raw_line.strip().partition("=")
        if separator and key and value:
            values[key] = value
    return values


def parse_sources(values: list[str]) -> dict[str, str]:
    sources: dict[str, str] = {}
    for value in values:
        name, separator, commit = value.partition("=")
        if not separator or not name or not commit:
            raise ValueError(f"invalid --source value: {value!r}")
        sources[name] = commit
    return sources


def main() -> None:
    args = parse_args()
    official = read_hashes(args.official)
    selected = read_hashes(args.selected)
    deviations = json.loads(args.deviations.read_text())
    if not isinstance(deviations, dict):
        raise ValueError("artifact deviations must be a JSON object")

    sources = parse_sources(args.source)
    if args.source_manifest:
        sources.update(read_key_values(args.source_manifest))

    errors: list[str] = []
    accepted: dict[str, str] = {}

    for filename, selected_digest in selected.items():
        official_digest = official.get(filename)
        if official_digest is None:
            errors.append(f"selected artifact is not in official release: {filename}")
        elif selected_digest == official_digest:
            accepted[filename] = "official"
        else:
            deviation = deviations.get(filename, {})
            if deviation.get("kind") != "replacement-hash":
                errors.append(f"unapproved hash replacement: {filename}")
            elif deviation.get("replacement_sha256") != selected_digest:
                errors.append(f"replacement hash does not match allowlist: {filename}")
            else:
                accepted[filename] = "replacement-hash"

    for filename in official.keys() - selected.keys():
        deviation = deviations.get(filename, {})
        kind = deviation.get("kind")
        if kind not in {"source-build", "omitted"}:
            errors.append(f"official artifact is missing without approval: {filename}")
            continue
        if kind == "source-build":
            source_name = deviation.get("source_name")
            expected_commit = deviation.get("source_commit")
            if sources.get(source_name) != expected_commit:
                errors.append(
                    f"source replacement mismatch for {filename}: "
                    f"expected {source_name}={expected_commit}"
                )
                continue
        accepted[filename] = kind

    unused_deviations = deviations.keys() - accepted.keys()
    if unused_deviations:
        errors.append(
            "unused artifact deviations: " + ", ".join(sorted(unused_deviations))
        )

    report = {
        "accepted": accepted,
        "deviations": {
            name: deviations[name]["reason"]
            for name, kind in accepted.items()
            if kind != "official"
        },
        "official_artifacts": len(official),
        "selected_artifacts": len(selected),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
