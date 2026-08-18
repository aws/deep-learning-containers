"""Unit tests for hash_image_content.

image_content_hash = sha256 over the ordered RootFS.Layers DiffIDs and the
runtime config (Env, Cmd, Entrypoint, Labels, ...), read from the pushed image's
registry config (no layer pull). These tests exercise the pure hashing + parsing core.
"""

import json

import hash_image_content as image_hasher
import pytest


DIFF_IDS = [
    "sha256:aaaa000000000000000000000000000000000000000000000000000000000000",
    "sha256:bbbb000000000000000000000000000000000000000000000000000000000000",
    "sha256:cccc000000000000000000000000000000000000000000000000000000000000",
]


def test_content_hash_is_deterministic():
    h1 = image_hasher.content_hash(DIFF_IDS, {})
    h2 = image_hasher.content_hash(DIFF_IDS, {})
    assert h1 == h2
    assert h1.startswith("sha256:")


def test_hash_is_order_sensitive():
    reordered = [DIFF_IDS[1], DIFF_IDS[0], DIFF_IDS[2]]
    assert image_hasher.content_hash(DIFF_IDS, {}) != image_hasher.content_hash(reordered, {})


def test_hash_changes_when_a_layer_changes():
    changed = DIFF_IDS[:-1] + [
        "sha256:dddd000000000000000000000000000000000000000000000000000000000000"
    ]
    assert image_hasher.content_hash(DIFF_IDS, {}) != image_hasher.content_hash(changed, {})


def test_empty_diff_ids_raises():
    with pytest.raises(ValueError):
        image_hasher.content_hash([], {})


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
    payload = {
        "config": {"Env": ["PATH=/usr/bin"], "Cmd": ["/bin/sh"]},
        "rootfs": {"type": "layers", "diff_ids": DIFF_IDS},
    }
    monkeypatch.setattr(image_hasher, "_inspect_image_config", lambda uri: json.dumps(payload))
    got = image_hasher.compute_image_content_hash(
        "123.dkr.ecr.us-west-2.amazonaws.com/ci:tag", platform="linux/amd64"
    )
    assert got == image_hasher.content_hash(DIFF_IDS, payload["config"])


def test_hash_ignores_build_timestamps(monkeypatch):
    """created + history are build timestamps: two rebuilds with identical
    content (same diff_ids + config) must hash the same regardless of them."""
    base = {
        "config": {"Env": ["PATH=/usr/bin"], "Cmd": ["/bin/sh"]},
        "rootfs": {"type": "layers", "diff_ids": DIFF_IDS},
    }
    first = {
        **base,
        "created": "2026-08-18T15:51:20.705516406-07:00",
        "history": [{"created": "2026-08-18T15:51:20.705516406-07:00", "created_by": "RUN x"}],
    }
    second = {
        **base,
        "created": "2026-08-18T15:59:59.000000000-07:00",
        "history": [{"created": "2026-08-18T15:59:59.000000000-07:00", "created_by": "RUN x"}],
    }
    monkeypatch.setattr(image_hasher, "_inspect_image_config", lambda uri: json.dumps(first))
    h_first = image_hasher.compute_image_content_hash("img:tag", platform="linux/amd64")
    monkeypatch.setattr(image_hasher, "_inspect_image_config", lambda uri: json.dumps(second))
    h_second = image_hasher.compute_image_content_hash("img:tag", platform="linux/amd64")
    assert h_first == h_second
