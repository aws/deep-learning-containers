import pytest
from test.test_utils import ec2 as ec2_utils


@pytest.fixture(scope="session")
def sglang():
    """
    SGLang fixture - parametrized automatically by pytest_generate_tests in main conftest.py
    This fixture will be populated with image URIs that match 'sglang' in the repository name.
    The repo name is just "sglang", not "sglang-inference".
    """
    pass


@pytest.fixture(scope="session")
def gpu_only():
    """SGLang requires GPU"""
    pass
