"""Unit tests for the SageMaker endpoint instance-type fallback ladder.

CPU-only, no container, no AWS: these load the real
``vllm/sagemaker/amzn2023/test_sm_model_serving.py`` by path (so the tests stay coupled
to the shipped harness, not a copy), stub out the AWS SDK, and drive the
``deployed_model`` fixture with a fake deploy that reports whatever outcome each case
needs.

The contract under test:

  * ``instance_type`` accepts a scalar or a priority-ordered list, and the ladder is
    walked in order — the first candidate that comes up wins.
  * A capacity error (ICE) falls through to the next candidate instead of failing, and
    the partially-created resources of the failed attempt are always cleaned up.
  * If every candidate is dry the test *skips*, so an AWS capacity shortage cannot
    block the auto-release.
  * Any error that is *not* a capacity error propagates immediately. This is the
    safety property that keeps the ladder from masking a genuine image regression.

The ICE string used here is copied verbatim from a real failing run, so the token
matching in ``_is_capacity_error`` is asserted against the exact text SageMaker
produces rather than a paraphrase of it.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parent / "sagemaker/amzn2023/test_sm_model_serving.py"

# Verbatim message from a real InsufficientInstanceCapacity failure. pytest renders an
# exception as "E <Type>: <str(exc)>", which is how we know str() carries the token.
REAL_ICE_MESSAGE = (
    "Encountered unexpected failed state while waiting for Endpoint. Final Resource "
    "State: Failed. Failure Reason: Unable to provision requested ML compute capacity "
    "due to InsufficientInstanceCapacity error. Please retry using a different ML "
    "instance type or after some time. Consider configuring InstancePools in your "
    "EndpointConfig with multiple instance types for priority-based fallback to "
    "improve capacity availability."
)


class _Stub:
    """Stand-in for a sagemaker.core resource: constructs, creates, and deletes."""

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def create(cls, *args, **kwargs):
        return cls()

    def delete(self):
        pass

    def wait_for_status(self, *args, **kwargs):
        pass


def _ensure_module(name, **attrs):
    """Register a stub for ``name`` only if the real module is not importable.

    Preferring the real module when present keeps this file from polluting sys.modules
    for anything else in the same pytest session.
    """
    try:
        __import__(name)
        return
    except ImportError:
        pass
    module = types.ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    sys.modules[name] = module


def _load_harness():
    """Import the shipped endpoint-test harness by path, stubbing the AWS SDK."""
    _ensure_module("boto3", client=lambda *a, **k: _Stub())
    _ensure_module("sagemaker")
    _ensure_module("sagemaker.core")
    _ensure_module("sagemaker.core.resources", Endpoint=_Stub, EndpointConfig=_Stub, Model=_Stub)
    _ensure_module(
        "sagemaker.core.shapes",
        ContainerDefinition=_Stub,
        ModelDataSource=_Stub,
        ProductionVariant=_Stub,
        S3ModelDataSource=_Stub,
    )
    _ensure_module("test_utils", random_suffix_name=lambda prefix, n: f"{prefix}-suffix")
    _ensure_module("test_utils.constants", INFERENCE_AMI_VERSION="al2023-1", SAGEMAKER_ROLE="role")

    # Import under a private name so pytest never collects the loaded harness as tests.
    spec = importlib.util.spec_from_file_location("_sm_serving_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


harness = _load_harness()


def _fixture_func(fixture):
    """Return the undecorated function behind a pytest fixture.

    pytest refuses direct fixture calls; the original lives on ``__wrapped__`` (pytest
    >=4) or ``__pytest_wrapped__.obj`` (older). Both are checked so this test does not
    silently break on a pytest bump.
    """
    if hasattr(fixture, "__wrapped__"):
        return fixture.__wrapped__
    return fixture.__pytest_wrapped__.obj


def _model_cfg(name):
    """Load one entry from the real config file, with instance_types normalized."""
    models = harness._load_sagemaker_config(harness.CONFIG_PATH)
    return next(m for m in models if m["name"] == name)


class _Request:
    def __init__(self, cfg):
        self.param = (cfg["name"], cfg)


def _drive(monkeypatch, cfg, outcome_for):
    """Run the fixture against a fake deploy; report which candidates it attempted.

    ``outcome_for(instance_type)`` returns an Exception to raise for that candidate, or
    None to let the deploy succeed. Resource cleanup is deliberately NOT asserted here —
    this fake replaces ``_deploy_endpoint`` wholesale, so it could only ever confirm its
    own behavior. The real cleanup path is covered by
    ``test_deploy_endpoint_cleans_up_after_capacity_failure``.
    """
    attempted = []

    def fake_deploy(image_uri, model_cfg, region, instance_type):
        attempted.append(instance_type)
        outcome = outcome_for(instance_type)
        if isinstance(outcome, Exception):
            raise outcome
        return (f"endpoint-{instance_type}", _Stub(), _Stub(), _Stub())

    monkeypatch.setattr(harness, "_deploy_endpoint", fake_deploy)

    generator = _fixture_func(harness.deployed_model)(_Request(cfg), "image-uri")
    result = {"attempted": attempted}
    try:
        result["yielded"] = next(generator)
    except pytest.skip.Exception as exc:
        result["outcome"] = "skipped"
        result["reason"] = str(exc)
        return result
    except Exception as exc:
        result["outcome"] = "raised"
        result["error"] = exc
        return result

    result["outcome"] = "deployed"
    with pytest.raises(StopIteration):  # normal fixture teardown
        next(generator)
    return result


def _ice(*_args):
    return RuntimeError(REAL_ICE_MESSAGE)


def test_nemotron_config_declares_a_fallback_ladder():
    """The config that motivated this feature must actually carry a ladder."""
    candidates = _model_cfg("nemotron-nano-12b-v2")["instance_types"]
    assert len(candidates) > 1, f"expected a fallback ladder, got {candidates}"
    assert candidates[0] == "ml.g6e.xlarge", "cheapest fitting instance should be tried first"
    # g6.xlarge is an L4 24GB card and cannot hold this model; it must never be a fallback.
    assert "ml.g6.xlarge" not in candidates


def test_scalar_instance_type_normalizes_to_one_candidate():
    """Entries that declare a plain string keep their existing single-attempt behavior."""
    assert _model_cfg("minicpm5-1b")["instance_types"] == ["ml.g6.xlarge"]


@pytest.mark.parametrize(
    "message",
    [
        REAL_ICE_MESSAGE,
        "ResourceLimitExceeded: account limit for ml.g6e.xlarge endpoint usage is 0",
        "CapacityError: no capacity",
        "insufficientinstancecapacity",  # case-insensitive
    ],
)
def test_capacity_errors_are_recognized(message):
    assert harness._is_capacity_error(RuntimeError(message))


@pytest.mark.parametrize(
    "message",
    [
        "ModelError: container entrypoint exited with status 1",
        "ValidationException: unsupported instance type",
        "AccessDeniedException: not authorized to call CreateModel",
    ],
)
def test_real_failures_are_not_treated_as_capacity(message):
    assert not harness._is_capacity_error(RuntimeError(message))


def test_first_candidate_wins_when_capacity_is_available(monkeypatch):
    result = _drive(monkeypatch, _model_cfg("nemotron-nano-12b-v2"), lambda it: None)

    assert result["outcome"] == "deployed"
    assert result["attempted"] == ["ml.g6e.xlarge"], "must not shop around unnecessarily"


def test_falls_back_to_next_candidate_on_capacity_error(monkeypatch):
    cfg = _model_cfg("nemotron-nano-12b-v2")
    first, second = cfg["instance_types"][0], cfg["instance_types"][1]

    result = _drive(monkeypatch, cfg, lambda it: _ice() if it == first else None)

    assert result["outcome"] == "deployed"
    assert result["attempted"] == [first, second]
    assert result["yielded"]["endpoint_name"] == f"endpoint-{second}"


def test_release_run_skips_when_every_candidate_is_dry(monkeypatch):
    """A capacity shortage must not block the auto-release."""
    monkeypatch.setattr(harness, "RELEASE_RUN", True)
    cfg = _model_cfg("nemotron-nano-12b-v2")

    result = _drive(monkeypatch, cfg, _ice)

    assert result["outcome"] == "skipped"
    assert result["attempted"] == cfg["instance_types"], "every candidate should be tried"
    assert "No SageMaker capacity" in result["reason"]


def test_pr_run_fails_when_every_candidate_is_dry(monkeypatch):
    """Outside a release, an exhausted ladder stays visible so a human re-triggers."""
    monkeypatch.setattr(harness, "RELEASE_RUN", False)
    cfg = _model_cfg("nemotron-nano-12b-v2")

    result = _drive(monkeypatch, cfg, _ice)

    assert result["outcome"] == "raised"
    assert isinstance(result["error"], AssertionError)
    assert result["attempted"] == cfg["instance_types"], "every candidate should be tried"
    assert "No SageMaker capacity" in str(result["error"])


def test_deploy_endpoint_cleans_up_after_capacity_failure(monkeypatch):
    """A failed deploy must not leak a Model, an EndpointConfig, or a Failed Endpoint.

    Exercises the real ``_deploy_endpoint`` (not a fake): the endpoint reports ICE from
    ``wait_for_status``, which is the exact point the old code raised from — after the
    Model and EndpointConfig already existed, and during fixture setup, so teardown
    never ran and all three resources leaked on every failed attempt.
    """
    deleted = []

    def tracking_stub(label):
        class Tracked(_Stub):
            @classmethod
            def create(cls, *args, **kwargs):
                return cls()

            def delete(self):
                deleted.append(label)

        return Tracked

    endpoint_cls = tracking_stub("endpoint")

    def raise_ice(self, *args, **kwargs):
        raise RuntimeError(REAL_ICE_MESSAGE)

    endpoint_cls.wait_for_status = raise_ice

    monkeypatch.setattr(harness, "Model", tracking_stub("model"))
    monkeypatch.setattr(harness, "EndpointConfig", tracking_stub("endpoint_config"))
    monkeypatch.setattr(harness, "Endpoint", endpoint_cls)
    monkeypatch.setattr(harness, "_get_role_arn", lambda region: "arn:aws:iam::123:role/test")

    with pytest.raises(RuntimeError, match="InsufficientInstanceCapacity"):
        harness._deploy_endpoint(
            "image-uri", _model_cfg("nemotron-nano-12b-v2"), "us-west-2", "ml.g6e.xlarge"
        )

    assert sorted(deleted) == ["endpoint", "endpoint_config", "model"], (
        f"every created resource must be deleted before re-raising; deleted={deleted}"
    )


def test_non_capacity_failure_is_surfaced_immediately(monkeypatch):
    """The ladder must never mask a genuine image or config defect."""
    cfg = _model_cfg("nemotron-nano-12b-v2")
    boom = ValueError("ModelError: container entrypoint exited with status 1")

    result = _drive(monkeypatch, cfg, lambda it: boom)

    assert result["outcome"] == "raised"
    assert result["error"] is boom
    assert result["attempted"] == [cfg["instance_types"][0]], "must stop at the first failure"
