"""Make scripts/ci importable by the telemetry selection unit tests.

``scripts/ci`` is not a Python package, so put it on ``sys.path`` and let the
tests use a plain ``import select_telemetry_configs``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "ci"))
