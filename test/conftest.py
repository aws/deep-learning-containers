import os
import subprocess

import pytest
import docker


# Ignore container_tests collection, as they will be called separately from test functions
collect_ignore = [os.path.join('container_tests', '*')]


def pytest_addoption(parser):
    parser.addoption(
        "--images",
        default=os.getenv('DLC_IMAGES'),
        nargs='+',
        help="Specify image(s) to run"
    )


@pytest.fixture(scope="session")
def docker_client():
    cmd = subprocess.run(f"$(aws ecr get-login --no-include-email --region {os.getenv('AWS_REGION', 'us-west-2')})",
                         stdout=subprocess.PIPE,
                         shell=True)
    if cmd.returncode:
        pytest.fail(f"Failed to log into ECR. Error log:\n{cmd.stdout.decode()}")
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
