import pytest

from test.test_utils import run_subprocess_cmd


def test_pip_check(image):
    """
    Test to run pip sanity tests
    """
    # Add null entrypoint to ensure command exits immediately
    run_subprocess_cmd(f"docker run --entrypoint='' {image} pip check")
