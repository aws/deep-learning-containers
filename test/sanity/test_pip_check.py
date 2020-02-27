import subprocess

import pytest


def test_pip_check(image):
    """
    Test to run pip sanity tests
    """
    if "tensorflow-inference" in image:
        pytest.xfail(reason='Tensorflow serving api requires tensorflow, but we explicitly do not install'
                            'tensorflow in serving containers.')

    # Add null entrypoint to ensure command exits immediately
    subprocess.run(["docker", "run", "-it", "--entrypoint=''", image, 'pip', 'check'], capture_output=True, check=True)
