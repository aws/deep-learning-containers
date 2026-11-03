"""Make scripts/ci and the telemetry script importable by the telemetry unit tests.

Neither directory is a Python package, so put both on ``sys.path`` and let the
tests use a plain ``import select_telemetry_tests`` or ``import deep_learning_container``.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ci"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "docker" / "telemetry"))
