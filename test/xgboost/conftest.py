"""Shared pytest configuration for XGBoost tests."""

import pytest
from test_utils.constants import SAGEMAKER_ROLE


def pytest_addoption(parser):
    parser.addoption(
        "--image-uri",
        action="store",
        required=True,
        help="XGBoost container image URI to test",
    )
    parser.addoption(
        "--region",
        action="store",
        default="us-west-2",
        help="AWS region",
    )


@pytest.fixture(scope="session")
def image_uri(request):
    return request.config.getoption("--image-uri")


@pytest.fixture(scope="session")
def region(request):
    return request.config.getoption("--region")


@pytest.fixture(scope="session")
def role():
    return SAGEMAKER_ROLE
