"""Verify Lambda RIC and RIE are installed."""

import importlib
import os


def test_awslambdaric_importable():
    importlib.import_module("awslambdaric")


def test_rie_binary_exists():
    rie = "/usr/local/bin/aws-lambda-rie"
    assert os.path.isfile(rie), f"RIE not found at {rie}"
    assert os.access(rie, os.X_OK), "RIE not executable"


def test_entrypoint_exists():
    script = "/lambda_entrypoint.sh"
    assert os.path.isfile(script), "lambda_entrypoint.sh not found"
    assert os.access(script, os.X_OK), "lambda_entrypoint.sh not executable"
