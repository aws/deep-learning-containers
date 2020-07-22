import os

import pytest
from invoke.context import Context

from test.test_utils import TEST_COVERAGE_FILE, is_pr_context, PR_ONLY_REASON


@pytest.mark.skipif(not is_pr_context(), reason=PR_ONLY_REASON)
@pytest.mark.integration("Generating this coverage doc")
def test_generate_coverage_doc():
    """
    Test generating the test coverage doc
    """
    test_coverage_file = TEST_COVERAGE_FILE
    ctx = Context()
    # Set DLC_TESTS to 'test' to avoid image names affecting function metadata (due to parametrization)
    # Set CODEBUILD_RESOLVED_SOURCE_VERSION to test for ease of running this test locally
    ctx.run("export DLC_TESTS='test' && export CODEBUILD_RESOLVED_SOURCE_VERSION='test' && export BUILD_CONTEXT=''"
            "&& pytest --collect-only  --generate-coverage-doc --ignore=container_tests/", hide=True)

    # Ensure that the coverage report is created
    assert os.path.exists(test_coverage_file), f"Cannot find test coverage report file {test_coverage_file}"
