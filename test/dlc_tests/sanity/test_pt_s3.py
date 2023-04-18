import pytest

from invoke.context import Context
from packaging.version import Version

from test.test_utils import (
    get_container_name,
    get_framework_and_version_from_tag,
    run_cmd_on_container,
    start_container,
)


@pytest.mark.usefixtures("feature_s3_plugin_present")
@pytest.mark.usefixtures("sagemaker", "huggingface")
@pytest.mark.integration("pt_s3_plugin_sanity")
@pytest.mark.model("N/A")
def test_pt_s3_sanity(pytorch_training, outside_versions_skip):
    """
    Check that the internally built PT S3 binary is properly installed.
    :param pytorch_training: framework fixture for pytorch training
    """
    outside_versions_skip(pytorch_training, "1.8.0", "1.12.1")
    ctx = Context()
    container_name = get_container_name("pt-s3", pytorch_training)
    start_container(container_name, pytorch_training, ctx)
    s3_path = "s3://pt-s3plugin-test-data-west2/test_0.JPEG"
    run_cmd_on_container(
        container_name,
        ctx,
        f'import awsio; print(awsio.__version__); from awsio.python.lib.io.s3.s3dataset import file_exists; print(file_exists("'
        + s3_path
        + '"))',
        executable="python",
    )
