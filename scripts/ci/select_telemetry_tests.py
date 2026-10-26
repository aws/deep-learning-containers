#!/usr/bin/env python3
"""Select the telemetry tests a PR needs.

Every image copies scripts/docker/telemetry/deep_learning_container.py, so a path
trigger on it reruns every framework's PR workflow. This script diffs the file
against the PR base so telemetry.pr.yml tests only what a change can affect. The
tests run on a mock image that telemetry.pipeline.yml builds from the checkout
(scripts/ci/telemetry_mock), so they never depend on a released image:

    only new REGION_MAPPING entries     -> no image test
    new --framework choices             -> the mock, as each new framework
    any other change to the file, or a
    change to the telemetry test suite  -> the mock, as MOCK_FRAMEWORK

Usage:
    python3 scripts/ci/select_telemetry_tests.py [--base-ref origin/main]

--base-ref defaults to HEAD^1, the base side of the PR merge commit that CI checks out.

Output (written to $GITHUB_OUTPUT, or stdout if not in CI):
    frameworks=<json-array>   the --framework values to test the mock image as
"""

import argparse
import ast
import dataclasses
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

DLC_SCRIPT = "scripts/docker/telemetry/deep_learning_container.py"
# A change under these paths can change the telemetry test of every image.
SUITE_PATHS = (
    ".github/workflows/_reusable.telemetry-tests.yml",
    ".github/workflows/telemetry.pipeline.yml",
    "scripts/ci/telemetry_mock/",
    "scripts/docker/telemetry/",
    "test/telemetry/",
)
# The framework the mock reports for a change that can affect every framework.
MOCK_FRAMEWORK = "base"
CONFIG_DIR = ".github/config/image"


class RegistryError(Exception):
    """deep_learning_container.py no longer has the registry literals this script reads."""


@dataclass(frozen=True)
class Registry:
    regions: dict[str, str]
    frameworks: tuple[str, ...]
    # ast.dump() of the module with both registry literals blanked out
    logic: str


@dataclass(frozen=True)
class Scope:
    reason: str
    # The --framework values to test the mock image as; empty means no image test
    frameworks: frozenset[str] = frozenset()
    removed: frozenset[str] = frozenset()


def _literal(node, what):
    try:
        return ast.literal_eval(node)
    except (TypeError, ValueError) as e:
        raise RegistryError(f"{what} is not a literal") from e


def parse_registry(source: str) -> Registry:
    """Read REGION_MAPPING and the --framework choices from the telemetry script."""
    tree = ast.parse(source)
    region_nodes, choice_nodes = [], []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and [getattr(t, "id", None) for t in node.targets] == ["REGION_MAPPING"]
            and isinstance(node.value, ast.Dict)
        ):
            region_nodes.append(node)
        elif (
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", None) == "add_argument"
            and node.args
            and getattr(node.args[0], "value", None) == "--framework"
        ):
            choice_nodes.extend(kw for kw in node.keywords if kw.arg == "choices")
    if len(region_nodes) != 1:
        raise RegistryError("expected one REGION_MAPPING = {...} assignment")
    if len(choice_nodes) != 1:
        raise RegistryError('expected one add_argument("--framework", choices=[...]) call')

    (region_node,), (choice_node,) = region_nodes, choice_nodes
    regions = _literal(region_node.value, "REGION_MAPPING")
    frameworks = _literal(choice_node.value, "--framework choices")
    if not all(isinstance(s, str) for item in regions.items() for s in item):
        raise RegistryError("REGION_MAPPING must map region strings to bucket id strings")
    if not isinstance(frameworks, (list, tuple)) or not all(isinstance(f, str) for f in frameworks):
        raise RegistryError("--framework choices must be a list of strings")
    if len(regions) != len(region_node.value.keys):
        raise RegistryError("REGION_MAPPING repeats a region")
    if len(set(frameworks)) != len(frameworks):
        raise RegistryError("--framework choices repeats a framework")

    region_node.value = choice_node.value = ast.Constant(None)
    return Registry(regions, tuple(frameworks), ast.dump(tree))


def classify(base_source: str | None, head: Registry) -> Scope:
    """Decide which telemetry tests the change from base_source to head needs."""
    every = frozenset({MOCK_FRAMEWORK})
    if base_source is None:
        return Scope(f"{DLC_SCRIPT} is new", every)
    try:
        base = parse_registry(base_source)
    except (RegistryError, SyntaxError) as e:
        return Scope(f"cannot read the base registry ({e})", every)

    added = frozenset(head.frameworks) - frozenset(base.frameworks)
    removed = frozenset(base.frameworks) - frozenset(head.frameworks)
    changed = sorted(r for r, bucket in base.regions.items() if head.regions.get(r) != bucket)
    new_regions = sorted(set(head.regions) - set(base.regions))

    reasons = []
    if base.logic != head.logic:
        reasons.append("telemetry code changed outside the registry")
    if changed:
        reasons.append(f"REGION_MAPPING entries changed or removed: {changed}")
    if removed:
        reasons.append(f"--framework choices removed: {sorted(removed)}")
    # Anything but a registry addition can affect every framework.
    frameworks = added | every if reasons else added
    if added:
        reasons.append(f"--framework choices added: {sorted(added)}")
    if new_regions:
        reasons.append(f"REGION_MAPPING entries added: {new_regions}")
    return Scope(
        "; ".join(reasons) or "registry reordered or reformatted only", frameworks, removed
    )


def config_frameworks() -> dict[str, list[str]]:
    """Return metadata.framework -> the image configs that declare it."""
    configs = {}
    for path in sorted(Path(CONFIG_DIR).rglob("*.yml")):
        framework = yaml.safe_load(path.read_text())["metadata"]["framework"]
        configs.setdefault(framework, []).append(path.as_posix())
    return configs


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=check)


def is_suite_path(path: str) -> bool:
    return any(path == p or (p.endswith("/") and path.startswith(p)) for p in SUITE_PATHS)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Select the telemetry tests a PR needs.")
    parser.add_argument("--base-ref", default="HEAD^1", help="Ref to diff HEAD against.")
    args = parser.parse_args(argv)

    try:
        head = parse_registry(Path(DLC_SCRIPT).read_text())
    except (RegistryError, SyntaxError) as e:
        print(
            f"::error file={DLC_SCRIPT}::{e}; update {Path(__file__).name} to read the new layout"
        )
        return 1

    try:
        merge_base = git("merge-base", args.base_ref, "HEAD").stdout.strip()
        changed = git("diff", "--name-only", "--no-renames", merge_base, "HEAD").stdout.splitlines()
    except subprocess.CalledProcessError as e:
        print(f"::error::cannot diff HEAD against {args.base_ref}: {e.stderr.strip()}")
        return 1

    scope = Scope(f"{DLC_SCRIPT} is unchanged")
    if DLC_SCRIPT in changed:
        base = git("show", f"{merge_base}:{DLC_SCRIPT}", check=False)
        scope = classify(base.stdout if base.returncode == 0 else None, head)
    suite_changes = [f for f in changed if f != DLC_SCRIPT and is_suite_path(f)]
    if suite_changes:
        scope = dataclasses.replace(
            scope,
            reason=f"{scope.reason}; telemetry test suite changed: {suite_changes}",
            frameworks=scope.frameworks | {MOCK_FRAMEWORK},
        )

    if MOCK_FRAMEWORK in scope.frameworks and MOCK_FRAMEWORK not in head.frameworks:
        print(
            f"::error::MOCK_FRAMEWORK '{MOCK_FRAMEWORK}' is not a --framework choice; "
            f"point it at one in {Path(__file__).name}"
        )
        return 1
    configs = config_frameworks()
    # The mock can't test a removed framework, but every image still built with it
    # would silently stop reporting telemetry.
    still_used = {f: configs[f] for f in sorted(scope.removed) if f in configs}
    if still_used:
        print(
            f"::error file={DLC_SCRIPT}::removed --framework choices are still in use: {still_used}"
        )
        return 1
    # The mock needs no config, but a new framework's images need its exact name.
    for framework in sorted(scope.frameworks - {MOCK_FRAMEWORK} - set(configs)):
        print(
            f"::warning file={DLC_SCRIPT}::No config in {CONFIG_DIR} has metadata.framework "
            f"'{framework}'; check the name, as images built with another name report nothing"
        )

    frameworks = sorted(scope.frameworks)
    print(f"::notice::Telemetry tests on the mock image as {frameworks}: {scope.reason}")
    line = f"frameworks={json.dumps(frameworks, separators=(',', ':'))}"
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(line + "\n")
    else:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
