import pytest

from invoke.context import Context
from packaging.version import Version

from test.test_utils import (
    get_container_name,
    get_framework_and_version_from_tag,
    get_processor_from_image_uri,
    run_cmd_on_container,
    start_container,
)


@pytest.mark.model("N/A")
def test_pt_s3_sanity(pytorch_training):
    """
    Check that the internally built PT S3 binary is properly installed.
    :param pytorch_training: framework fixture for pytorch training
    """
    _, framework_version = get_framework_and_version_from_tag(pytorch_training)
    if Version(framework_version) == Version("1.5.1") and get_processor_from_image_uri(pytorch_training) == "gpu":
        pytest.skip("Skipping this test for PT 1.5.1 GPU Training DLC images")
    ctx = Context()
    container_name = get_container_name("pt-s3", pytorch_training)
    start_container(container_name, pytorch_training, ctx)
    run_cmd_on_container(
        container_name, ctx, f"import awsio; print(awsio.__version__); from awsio.python.lib.io.s3.s3dataset import file_exists; print(f\"file_exists: {file_exists('s3://pt-s3plugin-test-data-west2/test_0.JPEG')}\") ", executable="python"
    )

