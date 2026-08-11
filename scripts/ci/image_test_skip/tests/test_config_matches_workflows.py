"""Consistency check: every suite a workflow invokes must exist in test-suites.yml."""

import importlib.util
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
TEST_SKIP_ACTIONS = ("check-test-pass", "record-test-pass")

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


def collect_invoked_suites():
    """Return [(workflow_file, suite), ...] for every literal suite a step invokes."""
    invoked = []
    for wf_path in sorted(WORKFLOWS_DIR.glob("*.yml")):
        try:
            workflow = yaml.safe_load(wf_path.read_text())
        except yaml.YAMLError:
            continue
        for step in _iter_steps(workflow):
            if not _uses_test_skip_action(step):
                continue
            suite = (step.get("with") or {}).get("suite")
            if not isinstance(suite, str):
                continue
            if "${{" in suite:  # expression-driven; can't resolve statically
                continue
            invoked.append((wf_path.name, suite))
    return invoked


def test_invoked_suites_are_configured():
    configured = set(hsc.load_config(REPO_ROOT))
    invoked = collect_invoked_suites()
    unknown = [(wf, s) for wf, s in invoked if s not in configured]
    assert not unknown, (
        "workflow steps invoke suites missing from .github/config/test-suites.yml "
        f"(would silently never skip): {unknown}"
    )


# --- detection-logic tests (synthetic workflows; prove the scan catches drift) ---

WF_LITERAL = """
jobs:
  t:
    steps:
      - uses: ./.github/actions/check-test-pass
        with:
          suite: pytorch/single_gpu
      - uses: ./.github/actions/record-test-pass
        with:
          suite: sanity
      - uses: ./.github/actions/ecr-authenticate  # unrelated step, ignored
        with:
          suite: not-a-real-suite
"""

WF_EXPRESSION = """
jobs:
  t:
    steps:
      - uses: ./.github/actions/check-test-pass
        with:
          suite: ${{ matrix.suite }}
"""


def _invoked_from_text(text):
    workflow = yaml.safe_load(text)
    out = []
    for step in _iter_steps(workflow):
        if not _uses_test_skip_action(step):
            continue
        suite = (step.get("with") or {}).get("suite")
        if isinstance(suite, str) and "${{" not in suite:
            out.append(suite)
    return out


def test_scan_collects_literal_suites_from_test_skip_steps_only():
    # Picks up both actions' suites; ignores the unrelated ecr-authenticate step.
    assert _invoked_from_text(WF_LITERAL) == ["pytorch/single_gpu", "sanity"]


def test_scan_skips_expression_driven_suites():
    assert _invoked_from_text(WF_EXPRESSION) == []
