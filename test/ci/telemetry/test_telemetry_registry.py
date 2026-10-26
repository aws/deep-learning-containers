"""The registry in deep_learning_container.py must stay readable and well-formed.

A PR that only adds a region runs no image test, so these checks are what
validate the new entry before it ships in every image.
"""

import re
from pathlib import Path

import pytest
import select_telemetry_tests as stt

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def registry():
    return stt.parse_registry((REPO_ROOT / stt.DLC_SCRIPT).read_text())


def test_selector_can_read_the_registry(registry):
    assert registry.regions
    assert registry.frameworks


def test_regions_are_aws_region_codes(registry):
    bad = [r for r in registry.regions if not re.fullmatch(r"[a-z]+(-[a-z]+)+-\d+", r)]
    assert not bad, f"REGION_MAPPING keys must be AWS region codes like 'us-west-2': {bad}"


def test_region_bucket_ids_are_unique_8_digit_hex(registry):
    bad = {r: b for r, b in registry.regions.items() if not re.fullmatch(r"[0-9a-f]{8}", b)}
    assert not bad, f"REGION_MAPPING values must be 8 lowercase hex digits: {bad}"
    seen = {}
    for region, bucket in registry.regions.items():
        seen.setdefault(bucket, []).append(region)
    shared = {b: rs for b, rs in seen.items() if len(rs) > 1}
    assert not shared, f"each region needs its own telemetry bucket id: {shared}"


def test_framework_names_are_safe_to_substitute(registry):
    # Dockerfiles sed the name into bash_telemetry.sh.template, which passes it to argparse.
    bad = [f for f in registry.frameworks if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", f)]
    assert not bad, f"--framework choices must match [a-z0-9][a-z0-9_-]*: {bad}"


def test_mock_framework_is_registered(registry):
    # Code changes test the mock image as this framework, which argparse must accept.
    assert stt.MOCK_FRAMEWORK in registry.frameworks
