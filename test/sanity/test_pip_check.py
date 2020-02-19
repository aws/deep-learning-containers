import pytest


def test_pip_check(image, docker_client):
    """
    Test to run pip sanity tests
    """
    if "tensorflow-inference" in image:
        pytest.xfail(reason='Tensorflow serving api requires tensorflow, but we explicitly do not install'
                            'tensorflow in serving containers.')

    # Add null entrypoint to ensure command exits immediately
    docker_client.containers.run(image, command="pip check", entrypoint='')
