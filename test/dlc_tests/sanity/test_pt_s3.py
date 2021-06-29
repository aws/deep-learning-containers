import pytest

from invoke.context import Context

from test.test_utils import (
    can_run_pytorch_s3_plugin_test,
    get_container_name,
    get_framework_and_version_from_tag,
    run_cmd_on_container,
    start_container,
)


@pytest.mark.integration("pt_s3_plugin_sanity")
@pytest.mark.model("N/A")
def test_pt_s3_sanity(pytorch_training, pt17_and_above_only):
    """
    Check that the internally built PT S3 binary is properly installed.
    :param pytorch_training: framework fixture for pytorch training
    """
    _, image_framework_version = get_framework_and_version_from_tag(pytorch_training)
    if not can_run_pytorch_s3_plugin_test(image_framework_version):
        pytest.skip(f"S3 plugin is not supported on {image_framework_version}")

    ctx = Context()
    container_name = get_container_name("pt-s3", pytorch_training)
    start_container(container_name, pytorch_training, ctx)
    s3_path = 's3://pt-s3plugin-test-data-west2/test_0.JPEG'
    run_cmd_on_container(
        container_name,
        ctx,
        (
            f"import awsio; "
            f"print(awsio.__version__); "
            f"from awsio.python.lib.io.s3.s3dataset import file_exists; "
            f"print(file_exists(\""+s3_path+"\"))"
        ),
        executable="python"
    )
