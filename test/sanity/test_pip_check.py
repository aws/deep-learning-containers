import subprocess

import pytest


@pytest.mark.usefixtures("pull_images")
def test_pip_check(image):
    """
    Test to run pip sanity tests
    """
    if "tensorflow-inference" in image:
        pytest.xfail(reason='Tensorflow serving api requires tensorflow, but we explicitly do not install'
                            'tensorflow in serving containers.')

    # Add null entrypoint to ensure command exits immediately
    cmd = subprocess.run(f"docker run --entrypoint='' {image} pip check", stdout=subprocess.PIPE, shell=True)
    if cmd.returncode:
        pytest.fail(f"{cmd.stdout.decode()}")
