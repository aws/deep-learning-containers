import os

import pytest
import docker

from test.test_utils import run_subprocess_cmd


# Ignore container_tests collection, as they will be called separately from test functions
collect_ignore = [os.path.join("container_tests", "*")]


def pytest_addoption(parser):
    parser.addoption(
        "--images",
        default=os.getenv("DLC_IMAGES").split(" "),
        nargs="+",
        help="Specify image(s) to run",
    )


@pytest.fixture(scope="session")
def docker_client():
    run_subprocess_cmd(
        f"$(aws ecr get-login --no-include-email --region {os.getenv('AWS_REGION', 'us-west-2')})",
        failure="Failed to log into ECR.",
    )
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
