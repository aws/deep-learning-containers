import pytest

from test.test_utils import run_subprocess_cmd


@pytest.mark.usefixtures("pull_images")
def test_pip_check(image):
    """
    Test to run pip sanity tests
    """
    if "tensorflow-inference" in image:
        pytest.xfail(
            reason="Tensorflow serving api requires tensorflow, but we explicitly do not install"
            "tensorflow in serving containers."
        )

    # Add null entrypoint to ensure command exits immediately
    run_subprocess_cmd(f"docker run --entrypoint='' {image} pip check")
