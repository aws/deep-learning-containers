"""Verify Lambda RIC and RIE are installed."""

import importlib
import os

import pytest


def test_awslambdaric_importable():
    importlib.import_module("awslambdaric")


def test_concurrency_mode_valid_when_set():
    """If the concurrency-mode knob is set, it must be a supported value."""
    mode = os.environ.get("AWS_LAMBDA_CONCURRENCY_MODE")
    if mode is not None:
        assert mode in {"thread", "process", "hybrid"}, f"unexpected concurrency mode: {mode}"


def test_multimode_ric_provides_concurrency_hooks():
    """The multi-mode RIC ships the concurrency-hooks module (source of the
    pre-fork hook decorator); the older single-mode client does not."""
    if os.environ.get("AWS_LAMBDA_CONCURRENCY_MODE") is not None:
        importlib.import_module("awslambdaric.lambda_concurrency_hooks")


def test_rie_binary_exists():
    rie = "/usr/local/bin/aws-lambda-rie"
    assert os.path.isfile(rie), f"RIE not found at {rie}"
    assert os.access(rie, os.X_OK), "RIE not executable"


def test_entrypoint_exists():
    script = "/lambda_entrypoint.sh"
    assert os.path.isfile(script), "lambda_entrypoint.sh not found"
    assert os.access(script, os.X_OK), "lambda_entrypoint.sh not executable"


def _multimode_config_provider():
    """Return the multi-mode RIC's config provider, or skip if the running image
    ships the older single-mode client (which has no concurrency-mode support)."""
    lambda_config = pytest.importorskip("awslambdaric.lambda_config")
    provider = getattr(lambda_config, "LambdaConfigProvider", None)
    if provider is None or not hasattr(provider, "concurrency_mode"):
        pytest.skip("multi-mode RIC not present in this image")
    return provider


@pytest.mark.parametrize("mode", ["process", "thread", "hybrid"])
def test_concurrency_mode_runtime_override(monkeypatch, mode):
    """A runtime env-var override — as set by `docker run -e ...` or by a Lambda
    function's environment configuration — must win over the image's baked default
    and be honored by the RIC's config resolution, for every supported mode."""
    provider = _multimode_config_provider()
    # RUNTIME_API is required for the provider to construct; value is unused here.
    monkeypatch.setenv("AWS_LAMBDA_RUNTIME_API", "127.0.0.1:9001")
    monkeypatch.setenv("AWS_LAMBDA_CONCURRENCY_MODE", mode)
    # environ defaults to the live os.environ, so this reads the overridden value.
    cfg = provider(["python", "handler.handler"])
    assert cfg.concurrency_mode == mode


def test_concurrency_mode_invalid_override_rejected(monkeypatch):
    """An unsupported override value must be rejected loudly, not silently ignored."""
    provider = _multimode_config_provider()
    monkeypatch.setenv("AWS_LAMBDA_RUNTIME_API", "127.0.0.1:9001")
    monkeypatch.setenv("AWS_LAMBDA_CONCURRENCY_MODE", "bogus")
    with pytest.raises(ValueError):
        provider(["python", "handler.handler"])


def test_ric_resolves_container_env():
    """End-to-end: assert the RIC resolves the concurrency mode actually injected
    into the container. The expected value comes from a SEPARATE var
    (EXPECTED_CONCURRENCY_MODE), not the var under test, so a failure of
    AWS_LAMBDA_CONCURRENCY_MODE to propagate cannot silently pass. CI runs this
    with the image's baked default (no override) and again under an `-e` override
    for every supported mode."""
    provider = _multimode_config_provider()
    # RUNTIME_API is required for the provider to construct; value is unused here.
    env = {**os.environ, "AWS_LAMBDA_RUNTIME_API": "127.0.0.1:9001"}
    resolved = provider(["python", "handler.handler"], environ=env).concurrency_mode
    expected = os.environ.get("EXPECTED_CONCURRENCY_MODE")
    if expected is None:
        # No injected expectation → the image's baked default must resolve.
        assert resolved == "process"
    else:
        # Prove the override propagated into the container AND the RIC honored it.
        assert os.environ.get("AWS_LAMBDA_CONCURRENCY_MODE") == expected
        assert resolved == expected
