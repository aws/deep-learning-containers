"""Consistency check: every suite a workflow invokes must exist in test-suites.yml."""

import json
import re
from pathlib import Path

import hash_suite_code as suite_hasher
import pytest
import yaml

# Extract the suite name out of the GitHub Actions expression, e.g.
# fromJSON(needs.check.outputs.skips)['pytorch/unit'] -> pytorch/unit
_ACCESSOR_RE = re.compile(
    r"outputs\.skips(?:\s*\|\|\s*'[^']*')?\s*\)\s*(?:\[\s*['\"]([^'\"]+)['\"]\s*\]|\.([A-Za-z_]\w*))"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
TEST_SKIP_ACTIONS = ("check-test-pass", "record-test-pass")
CHECK_WORKFLOW = "_reusable.check-test-pass.yml"
RECORD_ACTION = "record-test-pass"

EXPECTED_RECORD_INPUTS = {
    "image-content-hash": "${{ inputs.image-content-hash }}",
    "suite-code-hash": "${{ inputs.suite-code-hash }}",
    "ci-image-uri": "${{ inputs.image-uri }}",
    "ci-images-table-account-id": "${{ vars.CI_IMAGES_TABLE_ACCOUNT_ID }}",
}


def _norm(value):
    """Collapse whitespace so GHA-expression spacing differences don't matter."""
    return " ".join(str(value).split())


@pytest.fixture(autouse=True)
def _fresh_suite_config():
    """Drop hash_suite_code's cached config so these tests always read the real
    test-suites.yml. Without this, they rely on no other test leaving a fake config
    in the module-level cache."""
    suite_hasher._suites = None
    yield
    suite_hasher._suites = None


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
    configured = set(suite_hasher.load_config())
    invoked = collect_invoked_suites()
    unknown = [(wf, s) for wf, s in invoked if s not in configured]
    assert not unknown, (
        "workflow steps invoke suites that are missing from .github/config/test-suites.yml "
        f"(this test will never skip): {unknown}"
    )


def test_check_accessors_are_configured_and_checked():
    """Every `skips[...]` key in a job `if:` must be a real suite AND one the check job checks."""
    configured = set(suite_hasher.load_config())
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


def _record_test_pass_steps():
    """Yield (workflow_file, step) for every step that invokes the record-test-pass action."""
    for wf_path in sorted(WORKFLOWS_DIR.glob("*.yml")):
        try:
            workflow = yaml.safe_load(wf_path.read_text())
        except yaml.YAMLError:
            continue
        for step in _iter_steps(workflow):
            uses = step.get("uses", "")
            if isinstance(uses, str) and RECORD_ACTION in uses:
                yield wf_path.name, step


def test_record_test_pass_inputs_are_wired():
    """Every record-test-pass call site must pass all store attributes, correctly wired."""
    problems = []
    sites = 0
    for wf_name, step in _record_test_pass_steps():
        sites += 1
        with_ = step.get("with") or {}

        suite = with_.get("suite")
        if not (isinstance(suite, str) and suite.strip()):
            problems.append((wf_name, "suite", f"missing or empty (got {suite!r})"))

        for key, expected in EXPECTED_RECORD_INPUTS.items():
            actual = with_.get(key)
            if actual is None:
                problems.append((wf_name, key, "missing"))
            elif _norm(actual) != _norm(expected):
                problems.append((wf_name, key, f"expected {expected!r}, got {actual!r}"))

    assert sites, "found no record-test-pass call sites — the step walker matched nothing"
    assert not problems, (
        "record-test-pass call sites must thread every store attribute correctly "
        f"(missing/misspelled inputs write broken or tagless rows): {problems}"
    )
