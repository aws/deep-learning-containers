"""Unit tests for hash_image_content.

image_content_hash = sha256 of the ordered list of RootFS.Layers DiffIDs, read
from the pushed image's registry config (no layer pull). These tests exercise
the pure hashing + parsing core.
"""

import json

import hash_image_content as image_hasher
import pytest


DIFF_IDS = [
    "sha256:aaaa000000000000000000000000000000000000000000000000000000000000",
    "sha256:bbbb000000000000000000000000000000000000000000000000000000000000",
    "sha256:cccc000000000000000000000000000000000000000000000000000000000000",
]


def test_hash_of_diff_ids_is_deterministic():
    h1 = image_hasher.hash_diff_ids(DIFF_IDS)
    h2 = image_hasher.hash_diff_ids(DIFF_IDS)
    assert h1 == h2
    assert h1.startswith("sha256:")


def test_hash_is_order_sensitive():
    reordered = [DIFF_IDS[1], DIFF_IDS[0], DIFF_IDS[2]]
    assert image_hasher.hash_diff_ids(DIFF_IDS) != image_hasher.hash_diff_ids(reordered)


def test_hash_changes_when_a_layer_changes():
    changed = DIFF_IDS[:-1] + [
        "sha256:dddd000000000000000000000000000000000000000000000000000000000000"
    ]
    assert image_hasher.hash_diff_ids(DIFF_IDS) != image_hasher.hash_diff_ids(changed)


def test_empty_diff_ids_raises():
    with pytest.raises(ValueError):
        image_hasher.hash_diff_ids([])


def test_extract_diff_ids_from_single_platform_config():
    config = {"rootfs": {"type": "layers", "diff_ids": DIFF_IDS}}
    assert image_hasher.extract_diff_ids(json.dumps(config), platform="linux/amd64") == DIFF_IDS


def test_extract_diff_ids_rejects_multiarch_index():
    payload = {
        "linux/amd64": {"rootfs": {"diff_ids": DIFF_IDS}},
        "linux/arm64": {"rootfs": {"diff_ids": ["sha256:ffff"]}},
    }
    with pytest.raises(ValueError):
        image_hasher.extract_diff_ids(json.dumps(payload), platform="linux/amd64")


def test_extract_diff_ids_missing_rootfs_raises():
    with pytest.raises(ValueError):
        image_hasher.extract_diff_ids(json.dumps({"config": {}}), platform="linux/amd64")


def test_single_config_wrong_arch_raises():
    config = {"architecture": "arm64", "os": "linux", "rootfs": {"diff_ids": DIFF_IDS}}
    with pytest.raises(image_hasher.PlatformNotFoundError):
        image_hasher.extract_diff_ids(json.dumps(config), platform="linux/amd64")


def test_single_config_wrong_os_raises():
    config = {"architecture": "amd64", "os": "windows", "rootfs": {"diff_ids": DIFF_IDS}}
    with pytest.raises(image_hasher.PlatformNotFoundError):
        image_hasher.extract_diff_ids(json.dumps(config), platform="linux/amd64")


def test_single_config_matching_platform_ok():
    config = {"architecture": "amd64", "os": "linux", "rootfs": {"diff_ids": DIFF_IDS}}
    assert image_hasher.extract_diff_ids(json.dumps(config), platform="linux/amd64") == DIFF_IDS


def test_single_config_without_platform_fields_proceeds():
    config = {"rootfs": {"diff_ids": DIFF_IDS}}
    assert image_hasher.extract_diff_ids(json.dumps(config), platform="linux/amd64") == DIFF_IDS


def test_compute_end_to_end_with_stubbed_inspect(monkeypatch):
    config = {"rootfs": {"type": "layers", "diff_ids": DIFF_IDS}}
    monkeypatch.setattr(image_hasher, "_inspect_image_config", lambda uri: json.dumps(config))
    got = image_hasher.compute_image_content_hash(
        "123.dkr.ecr.us-west-2.amazonaws.com/ci:tag", platform="linux/amd64"
    )
    assert got == image_hasher.hash_diff_ids(DIFF_IDS)
