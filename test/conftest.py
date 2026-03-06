"""Common pytest fixtures for all tests under module test/"""

import pytest
from test_utils.aws import AWSSessionManager
from test_utils.constants import DEFAULT_REGION


def pytest_addoption(parser):
    parser.addoption("--image-uri", action="store", help="Image URI to be tested")
    parser.addoption(
        "--region", action="store", default=DEFAULT_REGION, help="AWS Region to test image on AWS"
    )


@pytest.fixture(scope="session")
def image_uri(request):
    return request.config.getoption("--image-uri")


@pytest.fixture(scope="session")
def region(request):
    return request.config.getoption("--region")


@pytest.fixture(scope="session")
def aws_session(region):
    return AWSSessionManager(region)
