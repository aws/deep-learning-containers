"""Unit tests for hash_suite_code.

suite_code_hash must be a deterministic, order-independent function of the
resolved file set's contents + relative paths, and must change iff a captured
file's bytes or the set membership changes.
"""

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parent.parent / "hash_suite_code.py"
spec = importlib.util.spec_from_file_location("hash_suite_code", MODULE_PATH)
hsc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hsc)


@pytest.fixture
def repo(tmp_path):
    """A fake repo root with a test-suites.yml and a small test tree."""
    (tmp_path / ".github" / "config").mkdir(parents=True)
    (tmp_path / "test" / "suitea" / "sub").mkdir(parents=True)
    (tmp_path / "test" / "suiteb").mkdir(parents=True)
    (tmp_path / "test" / "suitea" / "a.py").write_text("print('a')\n")
    (tmp_path / "test" / "suitea" / "sub" / "b.py").write_text("print('b')\n")
    (tmp_path / "test" / "suitea" / "top.ini").write_text("[x]\n")
    (tmp_path / "test" / "suiteb" / "c.py").write_text("print('c')\n")
    (tmp_path / ".github" / "config" / "test-suites.yml").write_text(
        "suites:\n"
        "  suitea:\n"
        "    skip_eligible: true\n"
        "    code_paths:\n"
        "      - test/suitea/**\n"
        "      - test/suitea/top.ini\n"
        "  suiteb:\n"
        "    skip_eligible: true\n"
        "    code_paths:\n"
        "      - test/suiteb/**\n"
        "  security:\n"
        "    skip_eligible: false\n"
        "    code_paths:\n"
        "      - test/security/**\n"
    )
    return tmp_path


def test_hash_is_deterministic(repo):
    h1 = hsc.hash_suite_code(repo, "suitea")
    h2 = hsc.hash_suite_code(repo, "suitea")
    assert h1 == h2
    assert h1.startswith("sha256:")


def test_hash_changes_when_a_captured_file_changes(repo):
    before = hsc.hash_suite_code(repo, "suitea")
    (repo / "test" / "suitea" / "a.py").write_text("print('changed')\n")
    after = hsc.hash_suite_code(repo, "suitea")
    assert before != after


def test_hash_changes_when_a_file_is_added_to_the_subtree(repo):
    before = hsc.hash_suite_code(repo, "suitea")
    (repo / "test" / "suitea" / "sub" / "new.py").write_text("print('new')\n")
    after = hsc.hash_suite_code(repo, "suitea")
    assert before != after


def test_hash_changes_when_a_captured_file_is_removed(repo):
    before = hsc.hash_suite_code(repo, "suitea")
    (repo / "test" / "suitea" / "sub" / "b.py").unlink()
    after = hsc.hash_suite_code(repo, "suitea")
    assert before != after


def test_unrelated_suite_change_does_not_affect_hash(repo):
    before = hsc.hash_suite_code(repo, "suitea")
    (repo / "test" / "suiteb" / "c.py").write_text("print('unrelated')\n")
    after = hsc.hash_suite_code(repo, "suitea")
    assert before == after


def test_distinct_suites_hash_differently(repo):
    assert hsc.hash_suite_code(repo, "suitea") != hsc.hash_suite_code(repo, "suiteb")


def test_unknown_suite_raises(repo):
    with pytest.raises(KeyError):
        hsc.hash_suite_code(repo, "does-not-exist")


def test_is_skip_eligible(repo):
    assert hsc.is_skip_eligible(repo, "suitea") is True
    assert hsc.is_skip_eligible(repo, "security") is False


def test_unknown_suite_is_not_skip_eligible(repo):
    assert hsc.is_skip_eligible(repo, "does-not-exist") is False


def test_moving_a_file_changes_hash(repo):
    before = hsc.hash_suite_code(repo, "suitea")
    src = repo / "test" / "suitea" / "a.py"
    content = src.read_text()
    src.unlink()
    (repo / "test" / "suitea" / "renamed.py").write_text(content)
    after = hsc.hash_suite_code(repo, "suitea")
    assert before != after
