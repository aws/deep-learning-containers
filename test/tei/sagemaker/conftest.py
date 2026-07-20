import os

import pytest


@pytest.fixture(scope="session")
def image_uri() -> str:
    uri = os.environ.get("TEST_IMAGE_URI")
    if not uri:
        pytest.skip("TEST_IMAGE_URI not set")
    return uri


@pytest.fixture(scope="session")
def role_arn() -> str:
    arn = os.environ.get("SM_ROLE_ARN")
    if not arn:
        pytest.skip("SM_ROLE_ARN not set")
    return arn


@pytest.fixture(scope="session")
def device_type() -> str:
    device = os.environ.get("TEST_DEVICE_TYPE", "").lower()
    if device not in ("cpu", "gpu"):
        pytest.skip(f"TEST_DEVICE_TYPE must be 'cpu' or 'gpu', got: {device!r}")
    return device
