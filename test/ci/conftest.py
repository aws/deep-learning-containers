"""Make the build-image action's scripts importable by the CI unit tests.

The modules under test live in ``.github/actions/build-image/``, which is not a
Python package (the action invokes them as scripts). Adding that directory to
``sys.path`` lets the tests use a plain ``import resolve_build_args``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".github" / "actions" / "build-image"))
