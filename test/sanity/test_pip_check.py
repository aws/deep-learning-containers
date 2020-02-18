import pytest


def test_pip_check(image, docker_client):
    """
    Test to run pip sanity tests
    """
    # Add null entrypoint to ensure command exits immediately
    docker_client.containers.run(image, command="pip check", entrypoint='')
