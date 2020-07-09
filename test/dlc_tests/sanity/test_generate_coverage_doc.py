import os

import pytest
from invoke.context import Context

from test.test_utils import TEST_COVERAGE_FILE


@pytest.mark.integration("Generating this coverage doc")
def test_generate_coverage_doc():
    """
    Test generating the test coverage doc
    """
    test_coverage_file = TEST_COVERAGE_FILE
    ctx = Context()
    # Set DLC_TESTS to 'test' to avoid image names affecting function metadata (due to parametrization)
    ctx.run("export DLC_TESTS='test' && pytest --collect-only  --generate-coverage-doc --ignore=container_tests/")

    # Ensure that the coverage report is created
    assert os.path.exists(test_coverage_file), f"Cannot find test coverage report file {test_coverage_file}"
