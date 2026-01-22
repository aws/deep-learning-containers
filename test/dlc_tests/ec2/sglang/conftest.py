import pytest
from test.test_utils import ec2 as ec2_utils


@pytest.fixture(scope="function")
def sglang_inference(request):
    """
    Fixture to provide SGLang image URI
    """
    return request.config.getoption("--image-uri")


@pytest.fixture(scope="function")
def gpu_only():
    """SGLang requires GPU"""
    return True
