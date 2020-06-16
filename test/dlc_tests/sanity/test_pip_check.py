import pytest

from invoke.context import Context


@pytest.mark.canary("Run pip check test regularly on production images")
def test_pip_check(image):
    """
    Test to run pip sanity tests
    """
    # Add null entrypoint to ensure command exits immediately
    ctx = Context()
    ctx.run(f"docker run --entrypoint='' {image} pip check")
