import pytest


def test_pip_check(image, docker_client):
    """
    Test to run pip sanity tests
    """
    docker_client.containers.run(image, "pip check")

