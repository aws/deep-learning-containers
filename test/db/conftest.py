"""Make the test-skip modules importable by the db unit tests.

The modules under test live in ``scripts/ci/test_skip/`` and are imported at
runtime by the check/record actions via ``sys.path`` insertion, so that
directory is not a Python package. Adding it to ``sys.path`` here lets the tests
use plain ``import hash_image_content`` etc. instead of per-file importlib glue.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "ci" / "test_skip"))
