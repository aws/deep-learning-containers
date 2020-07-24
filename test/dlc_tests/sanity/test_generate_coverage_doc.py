import os

import boto3
import pytest
from invoke.context import Context

from test.test_utils import is_pr_context, is_canary_context
from test.test_utils.test_reporting import get_test_coverage_file_path


def coverage_doc_skip_condition():
    # Run this only in us-west-2
    if is_canary_context() and os.getenv("AWS_REGION") == "us-west-2":
        return False
    elif is_pr_context():
        return False
    return True


@pytest.mark.skipif(coverage_doc_skip_condition(), reason="Run only in PR context or us-west-2 canary context")
@pytest.mark.integration("Generating this coverage doc")
@pytest.mark.canary("runs only in us-west-2")
def test_generate_coverage_doc():
    """
    Test generating the test coverage doc
    """
    test_coverage_file = get_test_coverage_file_path()
    ctx = Context()
    # Set DLC_TESTS to 'test' to avoid image names affecting function metadata (due to parametrization)
    # Set CODEBUILD_RESOLVED_SOURCE_VERSION to test for ease of running this test locally
    ctx.run("export DLC_TESTS='test' && export CODEBUILD_RESOLVED_SOURCE_VERSION='test' && export BUILD_CONTEXT=''"
            "&& pytest -s --collect-only  --generate-coverage-doc --ignore=container_tests/", hide=True)

    # Ensure that the coverage report is created
    assert os.path.exists(test_coverage_file), f"Cannot find test coverage report file {test_coverage_file}"

    # Write test coverage file to S3
    if is_canary_context():
        client = boto3.client("s3")
        with open(test_coverage_file, "rb") as test_file:
            client.put_object(Bucket="dlc-test-report", Key=os.path.basename(test_coverage_file), Body=test_file)
