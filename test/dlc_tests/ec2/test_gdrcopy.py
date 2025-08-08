import os
import pytest

import test.test_utils as test_utils
from test.test_utils import CONTAINER_TESTS_PREFIX, is_pr_context

from test.test_utils.ec2 import (
    get_efa_ec2_instance_type,
    filter_efa_instance_type,
    execute_ec2_training_test,
    are_heavy_instance_ec2_tests_enabled,
)

GDRCOPY_SANITY_TEST_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "gdrcopy", "test_gdrcopy.sh")
EC2_EFA_GPU_INSTANCE_TYPE_AND_REGION = get_efa_ec2_instance_type(
    default="p4d.24xlarge",
    filter_function=filter_efa_instance_type,
)


@pytest.mark.skip_serialized_release_pt_test
@pytest.mark.processor("gpu")
@pytest.mark.model("N/A")
@pytest.mark.team("conda")
@pytest.mark.integration("gdrcopy")
@pytest.mark.parametrize("ec2_instance_type,region", EC2_EFA_GPU_INSTANCE_TYPE_AND_REGION)
@pytest.mark.skipif(
    is_pr_context() and not are_heavy_instance_ec2_tests_enabled(),
    reason="Skip GDRCopy test in PR context unless explicitly enabled",
)
def test_gdrcopy(pytorch_training, ec2_connection, ec2_instance_type):
    if test_utils.is_image_incompatible_with_instance_type(pytorch_training, ec2_instance_type):
        pytest.skip(
            f"Image {pytorch_training} is incompatible with instance type {ec2_instance_type}"
        )
    if test_utils.is_image_incompatible_with_AL2023_for_gdrcopy(pytorch_training):
        pytest.skip(f"Image {pytorch_training} is incompatible with AL2023 base DLAMI.")
    execute_ec2_training_test(
        ec2_connection, pytorch_training, GDRCOPY_SANITY_TEST_CMD, enable_gdrcopy=True
    )
