import os

import boto3
import pytest

from botocore.exceptions import ClientError
from invoke.context import Context

from test.test_utils import LOGGER, is_mainline_context
from test.test_utils.test_reporting import get_test_coverage_file_path


TEST_COVERAGE_REPORT_BUCKET = f"dlc-test-coverage-reports"


@pytest.mark.integration("Generating this coverage doc")
@pytest.mark.model("N/A")
def test_generate_coverage_doc():
    """
    Test generating the test coverage doc
    """
    test_coverage_file = get_test_coverage_file_path()
    ctx = Context()
    # Set DLC_IMAGES to 'test' to avoid image names affecting function metadata (due to parametrization)
    # Set CODEBUILD_RESOLVED_SOURCE_VERSION to test for ease of running this test locally
    ctx.run(
        "export DLC_IMAGES='test' && export CODEBUILD_RESOLVED_SOURCE_VERSION='test' && export BUILD_CONTEXT=''"
        "&& pytest -s --collect-only  --generate-coverage-doc --ignore=container_tests/",
        hide=True,
    )

    # Ensure that the coverage report is created
    assert os.path.exists(test_coverage_file), f"Cannot find test coverage report file {test_coverage_file}"

    # Write test coverage file to S3
    if is_mainline_context():
        report_bucket = TEST_COVERAGE_REPORT_BUCKET
        client = boto3.client("s3")
        with open(test_coverage_file, "rb") as test_file:
            try:
                client.put_object(Bucket=report_bucket, Key=os.path.basename(test_coverage_file), Body=test_file)
            except ClientError as e:
                LOGGER.error(f"Unable to upload report to bucket {report_bucket}. Error: {e}")
                raise
