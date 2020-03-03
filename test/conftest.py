import os

import pytest
import docker


# Ignore container_tests collection, as they will be called separately from test functions
collect_ignore = [os.path.join('container_tests', '*')]


def pytest_addoption(parser):
    parser.addoption(
        "--images",
        required=True,
        default=os.getenv('DLC_IMAGES'),
        nargs='+',
        help="Specify image(s) to run"
    )


@pytest.fixture(scope="session")
def docker_client():
    return docker.from_env()


@pytest.fixture(scope="session")
def dlc_images(request):
    return request.config.getoption("--images")


@pytest.fixture(scope="session")
def pull_images(docker_client, dlc_images):
    for image in dlc_images:
        docker_client.images.pull(image)


def pytest_generate_tests(metafunc):
    if "image" in metafunc.fixturenames:
        metafunc.parametrize("image", metafunc.config.getoption("--images"))
