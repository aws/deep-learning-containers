"""Telemetry paths must trigger telemetry.pr.yml and no other workflow.

Every image copies the telemetry script, so a framework PR workflow that watches
telemetry paths reruns on every region or framework registration. telemetry.pr.yml
runs select_telemetry_configs.py to test only the images a change can affect.
"""

from pathlib import Path

import select_telemetry_configs as sts
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
TELEMETRY_PR = "telemetry.pr.yml"
TELEMETRY_PREFIXES = ("scripts/docker/telemetry/", "test/telemetry/")


def _load(path):
    return yaml.safe_load(path.read_text())


def _pull_request_paths(workflow):
    # PyYAML reads the bare `on:` key as boolean True.
    on = workflow.get("on") or workflow.get(True)
    pull_request = on.get("pull_request") if isinstance(on, dict) else None
    return pull_request.get("paths", []) if isinstance(pull_request, dict) else []


def _path_filter_patterns(workflow):
    """Yield every pattern of every dorny/paths-filter step."""
    for job in workflow.get("jobs", {}).values():
        for step in job.get("steps", []) or []:
            if "paths-filter" in str(step.get("uses", "")):
                for patterns in yaml.safe_load(step["with"]["filters"]).values():
                    yield from patterns


def _covers(pattern, path):
    if pattern.endswith("/**"):
        return path.startswith(pattern[:-2])
    return pattern == path


def test_only_the_telemetry_pr_workflow_watches_telemetry_paths():
    offenders = []
    for wf_path in sorted(WORKFLOWS_DIR.glob("*.yml")):
        if wf_path.name == TELEMETRY_PR:
            continue
        workflow = _load(wf_path)
        patterns = [*_pull_request_paths(workflow), *_path_filter_patterns(workflow)]
        hits = sorted({p for p in patterns if p.startswith(TELEMETRY_PREFIXES)})
        if hits:
            offenders.append((wf_path.name, hits))
    assert not offenders, (
        f"telemetry changes are tested by {TELEMETRY_PR}; remove these paths so every "
        f"telemetry edit stops rerunning these workflows: {offenders}"
    )


def test_telemetry_pr_workflow_triggers_on_every_path_the_selector_reads():
    patterns = _pull_request_paths(_load(WORKFLOWS_DIR / TELEMETRY_PR))
    required = [
        sts.DLC_SCRIPT,
        *sts.SUITE_PATHS,
        ".github/workflows/telemetry.pr.yml",
        "scripts/ci/select_telemetry_configs.py",
        "test/ci/telemetry/",
    ]
    missing = [path for path in required if not any(_covers(p, path) for p in patterns)]
    assert not missing, f"{TELEMETRY_PR} on.pull_request.paths does not cover: {missing}"
