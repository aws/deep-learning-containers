#!/usr/bin/env python3
"""Select the image configs whose telemetry tests a PR needs.

Every image copies scripts/docker/telemetry/deep_learning_container.py, so a path
trigger on it reruns every framework's PR workflow. Most edits to the file are
registry additions that leave existing images' telemetry unchanged. This script
diffs the file against the PR base and narrows the telemetry test matrix:

    only new REGION_MAPPING entries     -> no configs
    new --framework choices             -> released configs of those frameworks
    any other change to the file, or a
    change to the telemetry test suite  -> released configs of every registered framework

Usage:
    python3 scripts/ci/select_telemetry_configs.py [--base-ref origin/main]

--base-ref defaults to HEAD^1, the base side of the PR merge commit that CI checks out.

Output (written to $GITHUB_OUTPUT, or stdout if not in CI):
    configs=<json-array>   [{"config_file": "...", "name": "..."}], the discover-configs shape
"""

import argparse
import ast
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
    "scripts/docker/telemetry/",
    "test/telemetry/",
)
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
    run_all: bool = False
    frameworks: frozenset[str] = frozenset()


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
    if base_source is None:
        return Scope(f"{DLC_SCRIPT} is new", run_all=True)
    try:
        base = parse_registry(base_source)
    except (RegistryError, SyntaxError) as e:
        return Scope(f"cannot read the base registry ({e})", run_all=True)

    if base.logic != head.logic:
        return Scope("telemetry code changed outside the registry", run_all=True)
    changed = sorted(r for r, bucket in base.regions.items() if head.regions.get(r) != bucket)
    if changed:
        return Scope(f"REGION_MAPPING entries changed or removed: {changed}", run_all=True)
    removed = sorted(set(base.frameworks) - set(head.frameworks))
    if removed:
        return Scope(f"--framework choices removed: {removed}", run_all=True)

    added = frozenset(head.frameworks) - frozenset(base.frameworks)
    if added:
        return Scope(f"--framework choices added: {sorted(added)}", frameworks=added)
    new_regions = sorted(set(head.regions) - set(base.regions))
    if new_regions:
        return Scope(f"only REGION_MAPPING entries added: {new_regions}")
    return Scope("registry reordered or reformatted only")


def released_configs() -> list[tuple[str, dict]]:
    """Return (framework, discover-configs entry) for every released image config."""
    released = []
    for path in sorted(Path(CONFIG_DIR).rglob("*.yml")):
        config = yaml.safe_load(path.read_text())
        # Unreleased configs are left out of PR CI, so leave them out here too.
        if config["release"]["release"]:
            entry = {"config_file": path.as_posix(), "name": config["image"]["name"]}
            released.append((config["metadata"]["framework"], entry))
    return released


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=check)


def is_suite_path(path: str) -> bool:
    return any(path == p or (p.endswith("/") and path.startswith(p)) for p in SUITE_PATHS)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Select image configs for telemetry tests.")
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

    suite_changes = [f for f in changed if f != DLC_SCRIPT and is_suite_path(f)]
    if suite_changes:
        scope = Scope(f"telemetry test suite changed: {suite_changes}", run_all=True)
    elif DLC_SCRIPT in changed:
        base = git("show", f"{merge_base}:{DLC_SCRIPT}", check=False)
        scope = classify(base.stdout if base.returncode == 0 else None, head)
    else:
        scope = Scope(f"{DLC_SCRIPT} and the telemetry test suite are unchanged")

    released = released_configs()
    wanted = set(head.frameworks) if scope.run_all else scope.frameworks
    configs = [entry for framework, entry in released if framework in wanted]
    print(f"::notice::Telemetry tests for {len(configs)} config(s): {scope.reason}")
    for config in configs:
        print(f"  {config['config_file']}")
    for framework in sorted(scope.frameworks - {framework for framework, _ in released}):
        print(
            f"::warning file={DLC_SCRIPT}::No released config in {CONFIG_DIR} has "
            f"metadata.framework '{framework}', so no image tests its telemetry"
        )

    line = f"configs={json.dumps(configs, separators=(',', ':'))}"
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(line + "\n")
    else:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
