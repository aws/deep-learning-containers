"""Shared pytest configuration for Scikit-learn tests."""

import pytest
from test_utils.constants import SAGEMAKER_ROLE


def pytest_addoption(parser):
    parser.addoption(
        "--sklearn-version",
        default="1.4.2",
        help="Scikit-learn version under test (e.g. 1.4.2)",
    )


@pytest.fixture(scope="session")
def sklearn_version(request):
    return request.config.getoption("--sklearn-version")


@pytest.fixture(scope="session")
def role():
    return SAGEMAKER_ROLE
