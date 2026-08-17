"""Unit tests for the batch test-pass check (.github/actions/check-test-pass/check_test_pass.py)."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[4]
CHECK_TEST_PASS_PATH = REPO_ROOT / ".github" / "actions" / "check-test-pass" / "check_test_pass.py"
spec = importlib.util.spec_from_file_location("check_test_pass", CHECK_TEST_PASS_PATH)
check_test_pass = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_test_pass)


SGLANG_SUITES = ["sanity", "security", "sglang/upstream", "sglang/model"]
ELIGIBLE = {"sanity", "sglang/upstream", "sglang/model"}
CODE_HASHES = {"sanity": "h1", "sglang/upstream": "h2", "sglang/model": "h3"}


def _fake_helpers(monkeypatch, *, eligible, image_hash, code_hashes, skippable):
    """Patch check_test_pass._load_helpers to return controllable fakes."""
    hic = SimpleNamespace(compute_image_content_hash=lambda uri, platform=None: image_hash)
    hsc = SimpleNamespace(
        is_skip_eligible=lambda root, suite: suite in eligible,
        hash_suite_code=lambda root, suite: code_hashes[suite],
    )
    captured = {}

    def fake_check_test_pass(image_content_hash, suite_code_hashes, client=None):
        captured["image_content_hash"] = image_content_hash
        captured["suite_code_hashes"] = suite_code_hashes
        return set(skippable)

    store = SimpleNamespace(check_test_pass=fake_check_test_pass)
    monkeypatch.setattr(check_test_pass, "_load_helpers", lambda root: (hic, hsc, store))
    return captured


def test_non_eligible_suites_are_omitted(monkeypatch):
    _fake_helpers(
        monkeypatch,
        eligible=ELIGIBLE,
        image_hash="sha256:img",
        code_hashes=CODE_HASHES,
        skippable=[],
    )
    skips, _, _ = check_test_pass.compute_skips("/repo", "img:uri", SGLANG_SUITES)
    assert "security" not in skips
    assert set(skips) == ELIGIBLE


def test_skippable_suites_map_to_true(monkeypatch):
    _fake_helpers(
        monkeypatch,
        eligible=ELIGIBLE,
        image_hash="sha256:img",
        code_hashes=CODE_HASHES,
        skippable={"sanity", "sglang/model"},
    )
    skips, _, _ = check_test_pass.compute_skips("/repo", "img:uri", SGLANG_SUITES)
    assert skips == {"sanity": True, "sglang/upstream": False, "sglang/model": True}


def test_no_eligible_suites_returns_empty(monkeypatch):
    _fake_helpers(
        monkeypatch, eligible=set(), image_hash="sha256:img", code_hashes={}, skippable=[]
    )
    assert check_test_pass.compute_skips("/repo", "img:uri", SGLANG_SUITES) == ({}, "", {})


def test_check_test_pass_receives_deduped_code_hashes(monkeypatch):
    captured = _fake_helpers(
        monkeypatch,
        eligible=ELIGIBLE,
        image_hash="sha256:img",
        code_hashes=CODE_HASHES,
        skippable=[],
    )
    # A repeated suite must be hashed and queried only once.
    check_test_pass.compute_skips("/repo", "img:uri", ["sanity", *ELIGIBLE])
    assert captured["image_content_hash"] == "sha256:img"
    assert captured["suite_code_hashes"] == CODE_HASHES


def test_main_fails_open_on_error(monkeypatch, capsys):
    def boom(*a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(check_test_pass, "compute_skips", boom)
    rc = check_test_pass.main(
        ["--image-uri", "img", "--suites", '["sanity", "security"]', "--repo-root", "/repo"]
    )
    out = capsys.readouterr().out.strip()
    assert rc == 0
    assert json.loads(out) == {"skips": {}, "image_content_hash": "", "suite_code_hashes": {}}


def test_main_emits_compact_json(monkeypatch, capsys):
    monkeypatch.setattr(
        check_test_pass,
        "compute_skips",
        lambda *a, **k: (
            {"sanity": True, "pytorch/unit": False},
            "sha256:img",
            {"sanity": "h1", "pytorch/unit": "h2"},
        ),
    )
    rc = check_test_pass.main(["--image-uri", "img", "--suites", '["sanity"]', "--repo-root", "/repo"])
    out = capsys.readouterr().out.strip()
    assert rc == 0
    result = json.loads(out)
    assert result["skips"] == {"sanity": True, "pytorch/unit": False}
    assert result["image_content_hash"] == "sha256:img"
    assert result["suite_code_hashes"] == {"sanity": "h1", "pytorch/unit": "h2"}
    assert " " not in out  # compact, single-line for $GITHUB_OUTPUT
