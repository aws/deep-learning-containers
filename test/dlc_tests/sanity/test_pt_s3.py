import pytest

from invoke.context import Context
from packaging.version import Version
from packaging.specifiers import SpecifierSet
from test.test_utils import (
    get_container_name,
    get_framework_and_version_from_tag,
    get_processor_from_image_uri,
    run_cmd_on_container,
    start_container,
)

def can_run_s3_plugin(framework_version):
    return Version(framework_version) in SpecifierSet(">=1.6,<1.9")

def validate_or_skip_s3_plugin(framework_version):
    if not can_run_s3_plugin(framework_version):
        pytest.skip(f"S3 plugin is not supported on {framework_version}")

@pytest.mark.integration("pt_s3_plugin_sanity")
@pytest.mark.model("N/A")
def test_pt_s3_sanity(pytorch_training, pt16_and_above_only):
    """
    Check that the internally built PT S3 binary is properly installed.
    :param pytorch_training: framework fixture for pytorch training
    """
    _, framework_version = get_framework_and_version_from_tag(pytorch_training)
    validate_or_skip_s3_plugin(framework_version)
    ctx = Context()
    container_name = get_container_name("pt-s3", pytorch_training)
    start_container(container_name, pytorch_training, ctx)
    s3_path = 's3://pt-s3plugin-test-data-west2/test_0.JPEG'
    run_cmd_on_container(
        container_name, ctx, f"import awsio; print(awsio.__version__); from awsio.python.lib.io.s3.s3dataset import file_exists; print(file_exists(\""+s3_path+"\"))", executable="python"
    )

