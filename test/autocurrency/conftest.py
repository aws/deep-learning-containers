"""Pytest fixtures for the autocurrency CI tooling tests.

`agent-fix.py` is not importable by name (the hyphen is not a valid identifier),
so it is loaded from its path via importlib. It imports boto3 at module scope
purely for the Bedrock client; none of the code under test here touches AWS, so
a stub stands in when boto3 is absent, keeping these unit tests runnable without
the AWS dependencies.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

AGENT_FIX_PATH = (
    Path(__file__).parent.parent.parent / "scripts" / "ci" / "autocurrency" / "agent-fix.py"
)


def _load_agent_fix() -> ModuleType:
    try:
        import boto3  # noqa: F401
    except ImportError:
        sys.modules["boto3"] = ModuleType("boto3")

    spec = importlib.util.spec_from_file_location("agent_fix", AGENT_FIX_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def agent_fix() -> ModuleType:
    """The agent-fix module, loaded from its path."""
    return _load_agent_fix()
