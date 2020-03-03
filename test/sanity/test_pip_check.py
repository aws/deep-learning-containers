import pytest

from test.test_utils import utils


@pytest.mark.usefixtures("pull_images")
def test_pip_check(image, run_subprocess_cmd):
    """
    Test to run pip sanity tests
    """
    if "tensorflow-inference" in image:
        pytest.xfail(
            reason="Tensorflow serving api requires tensorflow, but we explicitly do not install"
            "tensorflow in serving containers."
        )

    # Add null entrypoint to ensure command exits immediately
    utils.run_subprocess_cmd(f"docker run --entrypoint='' {image} pip check")
