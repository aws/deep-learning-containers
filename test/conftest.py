def pytest_addoption(parser):
    parser.addoption("--image-uri", action="store", help="Image URI to be tested")
