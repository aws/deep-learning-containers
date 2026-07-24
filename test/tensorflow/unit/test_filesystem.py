"""TF-specific: verify tensorflow is installed inside the venv."""

import glob


def test_venv_has_tensorflow():
    """tensorflow / tensorflow_cpu is installed inside /opt/venv (not system site-packages)."""
    matches = glob.glob("/opt/venv/lib/python*/site-packages/tensorflow*")
    assert matches, "no tensorflow* directory under /opt/venv site-packages"
