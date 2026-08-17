"""Consistency check: every suite a workflow invokes must exist in test-suites.yml."""

import importlib.util
import json
import re
from pathlib import Path

import yaml

# Extract the suite name out of the GitHub Actions expression, e.g.
# fromJSON(needs.check.outputs.skips)['pytorch/unit'] -> pytorch/unit
_ACCESSOR_RE = re.compile(r"outputs\.skips\)\s*(?:\[\s*['\"]([^'\"]+)['\"]\s*\]|\.([A-Za-z_]\w*))")

REPO_ROOT = Path(__file__).resolve().parents[4]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
TEST_SKIP_ACTIONS = ("check-test-pass", "record-test-pass")
CHECK_WORKFLOW = "_reusable.check-test-pass.yml"

MODULE_PATH = Path(__file__).resolve().parent.parent / "hash_suite_code.py"
spec = importlib.util.spec_from_file_location("hash_suite_code", MODULE_PATH)
hsc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hsc)


def _iter_steps(workflow):
    """Yield every step mapping across all jobs in a parsed workflow."""
    jobs = workflow.get("jobs", {}) if isinstance(workflow, dict) else {}
    for job in jobs.values():
        if not isinstance(job, dict):
            continue
        for step in job.get("steps", []) or []:
            if isinstance(step, dict):
                yield step


def _uses_test_skip_action(step):
    uses = step.get("uses", "")
    return isinstance(uses, str) and any(a in uses for a in TEST_SKIP_ACTIONS)


def _suites_from_step(step):
    """Yield the suite names that a step/job references, (skipping ${{ }} expressions)."""
    with_ = step.get("with") or {}

    suite = with_.get("suite")
    if isinstance(suite, str) and "${{" not in suite:
        yield suite

    suites = with_.get("suites")
    if isinstance(suites, str) and "${{" not in suites:
        try:
            parsed = json.loads(suites)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, list):
            yield from (s for s in parsed if isinstance(s, str))


def _check_suites(workflow):
    """Return the set of suites the check job is asked to check."""
    suites = set()
    jobs = workflow.get("jobs", {}) if isinstance(workflow, dict) else {}
    for job in jobs.values():
        uses = job.get("uses", "") if isinstance(job, dict) else ""
        if isinstance(uses, str) and CHECK_WORKFLOW in uses:
            suites.update(_suites_from_step(job))
    return suites


def _check_accessor_keys(workflow):
    """Yield suite keys read from the check output in any job `if:` condition."""
    jobs = workflow.get("jobs", {}) if isinstance(workflow, dict) else {}
    for job in jobs.values():
        cond = job.get("if") if isinstance(job, dict) else None
        if not isinstance(cond, str):
            continue
        for bracket, dotted in _ACCESSOR_RE.findall(cond):
            yield bracket or dotted


def collect_invoked_suites():
    """Return [(workflow_file, suite), ...] for every test suite invoked by a workflow."""
    invoked = []
    for wf_path in sorted(WORKFLOWS_DIR.glob("*.yml")):
        try:
            workflow = yaml.safe_load(wf_path.read_text())
        except yaml.YAMLError:
            continue
        for step in _iter_steps(workflow):
            if not _uses_test_skip_action(step):
                continue
            for suite in _suites_from_step(step):
                invoked.append((wf_path.name, suite))
        for suite in _check_suites(workflow):
            invoked.append((wf_path.name, suite))
    return invoked


def test_invoked_suites_are_configured():
    configured = set(hsc.load_config(REPO_ROOT))
    invoked = collect_invoked_suites()
    unknown = [(wf, s) for wf, s in invoked if s not in configured]
    assert not unknown, (
        "workflow steps invoke suites that are missing from .github/config/test-suites.yml "
        f"(this test will never skip): {unknown}"
    )


def test_check_accessors_are_configured_and_checked():
    """Every `skips[...]` key in a job `if:` must be a real suite AND one the check job checks."""
    configured = set(hsc.load_config(REPO_ROOT))
    problems = []
    for wf_path in sorted(WORKFLOWS_DIR.glob("*.yml")):
        try:
            workflow = yaml.safe_load(wf_path.read_text())
        except yaml.YAMLError:
            continue
        check_suites = _check_suites(workflow)
        for key in _check_accessor_keys(workflow):
            if key not in configured:
                problems.append((wf_path.name, key, "not in test-suites.yml"))
            elif key not in check_suites:
                problems.append((wf_path.name, key, "not in the check job's suites list"))
    assert not problems, f"A job that is skippable was not checked by the check job: {problems}"
