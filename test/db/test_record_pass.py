"""Unit tests for the record-test-pass action's URI→tag extraction."""

import sys
from pathlib import Path

# record_pass.py lives in the composite action dir, not on the default path.
_ACTION_DIR = (
    Path(__file__).resolve().parents[2] / ".github" / "actions" / "record-test-pass"
)
sys.path.insert(0, str(_ACTION_DIR))

import record_pass  # noqa: E402

FULL_URI = "123456789012.dkr.ecr.us-west-2.amazonaws.com/ci:sglang-ec2-amzn2023-0.5.12.dlc1-gpu-py312-cu130-pr-6622"
BARE_TAG = "sglang-ec2-amzn2023-0.5.12.dlc1-gpu-py312-cu130-pr-6622"


def test_extracts_bare_tag_from_full_uri():
    assert record_pass._tag_from_uri(FULL_URI) == BARE_TAG


def test_empty_uri_yields_empty_tag():
    assert record_pass._tag_from_uri("") == ""


def test_bare_tag_passes_through_unchanged():
    # Defensive: if only a tag (no registry/colon) is ever passed, keep it as-is.
    assert record_pass._tag_from_uri(BARE_TAG) == BARE_TAG
