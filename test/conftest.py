import pytest


def pytest_addoption(parser):
    parser.addoption("--image-uri", action="store", help="Image URI to be tested")


@pytest.fixture(scope="session")
def image_uri(request):
    return request.config.getoption("--image-uri")
