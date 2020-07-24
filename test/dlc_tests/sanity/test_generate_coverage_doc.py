import os

import boto3
import pytest

from botocore.exceptions import ClientError
from invoke.context import Context

from test.test_utils import is_pr_context, is_canary_context, LOGGER
from test.test_utils.test_reporting import get_test_coverage_file_path


TEST_COVERAGE_REPORT_REGION = "us-west-2"
TEST_COVERAGE_REPORT_BUCKET = f"dlc-test-coverage-reports-{TEST_COVERAGE_REPORT_REGION}"


def coverage_doc_skip_condition():
    # Run this only in us-west-2
    if is_canary_context() and os.getenv("AWS_REGION") == TEST_COVERAGE_REPORT_REGION:
        return False
    elif is_pr_context():
        return False
    return True


@pytest.mark.skipif(
    coverage_doc_skip_condition(), reason=f"Run only in PR context or {TEST_COVERAGE_REPORT_REGION} canary context"
)
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
    ctx.run(
        "export DLC_TESTS='test' && export CODEBUILD_RESOLVED_SOURCE_VERSION='test' && export BUILD_CONTEXT=''"
        "&& pytest -s --collect-only  --generate-coverage-doc --ignore=container_tests/",
        hide=True,
    )

    # Ensure that the coverage report is created
    assert os.path.exists(test_coverage_file), f"Cannot find test coverage report file {test_coverage_file}"

    # Write test coverage file to S3
    report_region = TEST_COVERAGE_REPORT_REGION
    report_bucket = TEST_COVERAGE_REPORT_BUCKET
    if is_canary_context() and os.getenv("AWS_REGION") == report_region:
        client = boto3.client("s3")
        with open(test_coverage_file, "rb") as test_file:
            try:
                client.put_object(Bucket=report_bucket, Key=os.path.basename(test_coverage_file), Body=test_file)
            except ClientError as e:
                LOGGER.error(f"Unable to upload report to bucket {report_bucket}. Error: {e}")
                raise
