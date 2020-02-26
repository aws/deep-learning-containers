import os

import pytest
import docker


# Ignore legacy test collection
collect_ignore = [os.path.join('container_tests', '*')]


def pytest_addoption(parser):
    parser.addoption(
        "--images",
        required=True,
        nargs='+',
        help="Specify image(s) to run"
    )


@pytest.fixture(scope="session")
def docker_client():
    return docker.from_env()


def pytest_generate_tests(metafunc):
    if "image" in metafunc.fixturenames:
        metafunc.parametrize("image", metafunc.config.getoption("--images"))
