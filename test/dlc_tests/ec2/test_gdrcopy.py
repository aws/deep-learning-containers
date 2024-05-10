import os
import pytest

from packaging.version import Version
from packaging.specifiers import SpecifierSet

import test.test_utils as test_utils
from test.test_utils import (
    CONTAINER_TESTS_PREFIX,
    is_pr_context,
    get_framework_and_version_from_tag,
)

from test.test_utils.ec2 import (
    get_efa_ec2_instance_type,
    filter_efa_instance_type,
    execute_ec2_training_test,
    are_heavy_instance_ec2_tests_enabled,
)

GDRCOPY_SANITY_TEST_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "gdrcopy", "test_gdrcopy.sh")
GDRCOPY_SANITY_DEV_TEST_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "gdrcopy", "test_gdrcopy_dev.sh")
EC2_EFA_GPU_INSTANCE_TYPE_AND_REGION = get_efa_ec2_instance_type(
    default="p4d.24xlarge",
    filter_function=filter_efa_instance_type,
)


def get_gdrcopy_sanity_test_cmd(pytorch_training):
    # GDRCopy v2.4 and above uses `test_gdrcopy.sh` which is currently available in PT 2.2
    # GDRCopy v2.3 and below uses `test_gdrcopy_dev.sh` which is currently available in PT 1.13, 2.1
    _, image_framework_version = get_framework_and_version_from_tag(pytorch_training)
    return (
        GDRCOPY_SANITY_TEST_CMD
        if Version(image_framework_version) in SpecifierSet(">=2.2")
        else GDRCOPY_SANITY_DEV_TEST_CMD
    )


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.processor("gpu")
@pytest.mark.model("N/A")
@pytest.mark.team("conda")
@pytest.mark.integration("gdrcopy")
@pytest.mark.parametrize("ec2_instance_type,region", EC2_EFA_GPU_INSTANCE_TYPE_AND_REGION)
@pytest.mark.skipif(
    is_pr_context() and not are_heavy_instance_ec2_tests_enabled(),
    reason="Skip GDRCopy test in PR context unless explicitly enabled",
)
def test_gdrcopy(
    pytorch_training, ec2_connection, ec2_instance_type, region, gpu_only, pt113_and_above_only
):
    if test_utils.is_image_incompatible_with_instance_type(pytorch_training, ec2_instance_type):
        pytest.skip(
            f"Image {pytorch_training} is incompatible with instance type {ec2_instance_type}"
        )
    _, framework_version = test_utils.get_framework_and_version_from_tag(pytorch_training)
    framework_version = Version(framework_version)
    if test_utils.is_ec2_image(pytorch_training) and framework_version == Version("1.13.1"):
        pytest.skip(
            f"Image {pytorch_training} does not support GDR Copy"
        )
    gdrcopy_test_path = get_gdrcopy_sanity_test_cmd(pytorch_training)
    execute_ec2_training_test(
        ec2_connection, pytorch_training, gdrcopy_test_path, enable_gdrcopy=True
    )
