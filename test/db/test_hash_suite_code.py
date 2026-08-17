"""Unit tests for hash_suite_code.

suite_code_hash must be a deterministic, order-independent function of the
resolved file set's contents + relative paths, and must change iff a captured
file's bytes or the set membership changes.
"""

import hash_suite_code as suite_hasher
import pytest


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
    h1 = suite_hasher.hash_suite_code(repo, "suitea")
    h2 = suite_hasher.hash_suite_code(repo, "suitea")
    assert h1 == h2
    assert h1.startswith("sha256:")


def test_hash_changes_when_a_captured_file_changes(repo):
    before = suite_hasher.hash_suite_code(repo, "suitea")
    (repo / "test" / "suitea" / "a.py").write_text("print('changed')\n")
    after = suite_hasher.hash_suite_code(repo, "suitea")
    assert before != after


def test_hash_changes_when_a_file_is_added_to_the_subtree(repo):
    before = suite_hasher.hash_suite_code(repo, "suitea")
    (repo / "test" / "suitea" / "sub" / "new.py").write_text("print('new')\n")
    after = suite_hasher.hash_suite_code(repo, "suitea")
    assert before != after


def test_hash_changes_when_a_captured_file_is_removed(repo):
    before = suite_hasher.hash_suite_code(repo, "suitea")
    (repo / "test" / "suitea" / "sub" / "b.py").unlink()
    after = suite_hasher.hash_suite_code(repo, "suitea")
    assert before != after


def test_unrelated_suite_change_does_not_affect_hash(repo):
    before = suite_hasher.hash_suite_code(repo, "suitea")
    (repo / "test" / "suiteb" / "c.py").write_text("print('unrelated')\n")
    after = suite_hasher.hash_suite_code(repo, "suitea")
    assert before == after


def test_distinct_suites_hash_differently(repo):
    assert suite_hasher.hash_suite_code(repo, "suitea") != suite_hasher.hash_suite_code(repo, "suiteb")


def test_unknown_suite_raises(repo):
    with pytest.raises(KeyError):
        suite_hasher.hash_suite_code(repo, "does-not-exist")


def test_suite_matching_no_files_raises(repo):
    with pytest.raises(ValueError):
        suite_hasher.hash_suite_code(repo, "security")


def test_is_skip_eligible(repo):
    assert suite_hasher.is_skip_eligible(repo, "suitea") is True
    assert suite_hasher.is_skip_eligible(repo, "security") is False


def test_unknown_suite_is_not_skip_eligible(repo):
    assert suite_hasher.is_skip_eligible(repo, "does-not-exist") is False


def test_moving_a_file_changes_hash(repo):
    before = suite_hasher.hash_suite_code(repo, "suitea")
    src = repo / "test" / "suitea" / "a.py"
    content = src.read_text()
    src.unlink()
    (repo / "test" / "suitea" / "renamed.py").write_text(content)
    after = suite_hasher.hash_suite_code(repo, "suitea")
    assert before != after
