"""Shared pytest configuration for XGBoost tests."""

import pytest
from test_utils.constants import SAGEMAKER_ROLE


@pytest.fixture(scope="session")
def role():
    return SAGEMAKER_ROLE
