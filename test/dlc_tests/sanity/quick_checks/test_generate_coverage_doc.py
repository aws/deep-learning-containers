import os

import boto3
import pytest

from botocore.exceptions import ClientError
from invoke.context import Context

from test.test_utils import LOGGER, is_mainline_context, is_graviton_architecture
from test.test_utils.test_reporting import get_test_coverage_file_path


ACCOUNT_ID = os.getenv("ACCOUNT_ID", boto3.client("sts").get_caller_identity().get("Account"))
TEST_COVERAGE_REPORT_BUCKET = f"dlc-test-coverage-reports-{ACCOUNT_ID}"


@pytest.mark.quick_checks
@pytest.mark.integration("Generating this coverage doc")
@pytest.mark.model("N/A")
# @pytest.mark.skipif(
#     is_mainline_context and is_graviton_architecture,
#     reason="Skipping the test for Graviton image build in mainline context as ARM image is used as a base",
# )
def test_generate_coverage_doc():
    """
    Test generating the test coverage doc
    """

    test_coverage_file = get_test_coverage_file_path()
    ctx = Context()
    # Set DLC_IMAGES to 'test' to avoid image names affecting function metadata (due to parametrization)
    # Set CODEBUILD_RESOLVED_SOURCE_VERSION to test for ease of running this test locally
    ctx.run(
        "export DLC_IMAGES='' && export CODEBUILD_RESOLVED_SOURCE_VERSION='test' && export BUILD_CONTEXT=''"
        "&& pytest -s --collect-only  --generate-coverage-doc --ignore=container_tests/",
        hide=True,
    )

    # Ensure that the coverage report is created
    assert os.path.exists(test_coverage_file), f"Cannot find test coverage report file {test_coverage_file}"

    # Write test coverage file to S3
    if is_mainline_context():
        client = boto3.client("s3")
        with open(test_coverage_file, "rb") as test_file:
            try:
                client.put_object(
                    Bucket=TEST_COVERAGE_REPORT_BUCKET, Key=os.path.basename(test_coverage_file), Body=test_file
                )
            except ClientError as e:
                LOGGER.error(f"Unable to upload report to bucket {TEST_COVERAGE_REPORT_BUCKET}. Error: {e}")
                raise
