"""Verify Lambda RIC and RIE are installed."""

import importlib
import os


def test_awslambdaric_importable():
    importlib.import_module("awslambdaric")


def test_pre_fork_hook_available():
    """register_pre_fork is GA in awslambdaric>=4.1.0; the serving handlers use it to
    start one shared engine in the parent before workers fork."""
    hooks = importlib.import_module("awslambdaric.lambda_concurrency_hooks")
    assert callable(hooks.register_pre_fork)


def test_rie_binary_exists():
    rie = "/usr/local/bin/aws-lambda-rie"
    assert os.path.isfile(rie), f"RIE not found at {rie}"
    assert os.access(rie, os.X_OK), "RIE not executable"


def test_entrypoint_exists():
    # Core images ship lambda_entrypoint.sh; serving variants (sglang, vllm) ship
    # their own dedicated entrypoint. Accept whichever the image provides.
    candidates = ["/lambda_entrypoint.sh", "/sglang_entrypoint.sh", "/vllm_entrypoint.sh"]
    script = next((s for s in candidates if os.path.isfile(s)), None)
    assert script is not None, f"no entrypoint found (looked for {candidates})"
    assert os.access(script, os.X_OK), f"{script} not executable"
