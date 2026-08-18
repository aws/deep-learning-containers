"""Compute suite_code_hash for a test suite.

suite_code_hash is sha256 over the file set from .github/config/test-suites.yml —
each file's repo-relative path plus its bytes, sorted by path so the result is
order-independent. Editing, adding, moving, or removing a captured file changes
the hash.
"""

import hashlib
from pathlib import Path

import yaml

CONFIG_REL = ".github/config/test-suites.yml"


def load_config(repo_root):
    """Load the test-skip suite config as a dict keyed by suite name."""
    config_path = Path(repo_root) / CONFIG_REL
    data = yaml.safe_load(config_path.read_text())
    return data.get("suites", {})


def resolve_files(repo_root, code_paths):
    """Resolve code_paths globs to a sorted, de-duplicated list of existing files.
    Paths are specified relative to repo_root.
    """
    repo_root = Path(repo_root)
    matched = set()
    for pattern in code_paths:
        # pathlib's trailing `**` matches directories only; `**/*` is what
        # captures files recursively.
        if pattern.endswith("/**"):
            pattern = pattern + "/*"
        for path in repo_root.glob(pattern):
            if path.is_file():
                matched.add(path)
    return sorted(matched, key=lambda p: p.relative_to(repo_root).as_posix())


def hash_suite_code(repo_root, suite):
    """Return the suite_code_hash (``sha256:<hex>``) for one suite.

    Raises KeyError if the suite is not in the config, or ValueError if its
    code_paths match no files.
    """
    suites = load_config(repo_root)
    if suite not in suites:
        raise KeyError(f"unknown suite: {suite!r} (not in {CONFIG_REL})")
    code_paths = suites[suite].get("code_paths", [])
    files = resolve_files(repo_root, code_paths)
    if not files:
        raise ValueError(f"suite {suite!r} matched no files (code_paths={code_paths!r})")

    digest = hashlib.sha256()
    repo_root = Path(repo_root)
    for path in files:
        rel = path.relative_to(repo_root).as_posix()
        # Bind path to content so a rename (same bytes, new path) is a real change.
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def is_skip_eligible(repo_root, suite):
    """Return True iff the suite exists and is marked skip_eligible in the config.
    An unknown suite is treated as not eligible.
    """
    suite_cfg = load_config(repo_root).get(suite)
    return bool(suite_cfg and suite_cfg.get("skip_eligible"))
