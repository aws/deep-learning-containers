"""Pytest configuration for autocurrency tests.

Adds the repository root to sys.path so that
``from scripts.autocurrency.docs_pr_functions import ...`` works.
"""

import sys
from pathlib import Path

# Ensure repo root is on sys.path
_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
