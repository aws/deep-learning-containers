import os
import pytest

import test.test_utils as test_utils
from test.test_utils import CONTAINER_TESTS_PREFIX, is_pr_context, is_efa_dedicated
from test.test_utils.ec2 import (
    get_efa_ec2_instance_type,
    filter_efa_instance_type,
    execute_ec2_training_test,
)

GDRCOPY_SANITY_TEST_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "gdrcopy", "test_gdrcopy.sh")
EC2_EFA_GPU_INSTANCE_TYPE_AND_REGION = get_efa_ec2_instance_type(
    default="p4d.24xlarge",
    filter_function=filter_efa_instance_type,
)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.processor("gpu")
@pytest.mark.model("N/A")
@pytest.mark.integration("gdrcopy")
@pytest.mark.parametrize("ec2_instance_type,region", EC2_EFA_GPU_INSTANCE_TYPE_AND_REGION)
@pytest.mark.parametrize(
    "ec2_instance_ami", [test_utils.UBUNTU_20_BASE_DLAMI_US_WEST_2], indirect=True
)
@pytest.mark.skipif(
    is_pr_context() and not is_efa_dedicated(),
    reason="Skip GDRCopy test in PR context unless explicitly enabled",
)
def test_gdrcopy(
    pytorch_training, ec2_connection, ec2_instance_type, region, gpu_only, pt113_and_above_only
):
    if test_utils.is_image_incompatible_with_instance_type(pytorch_training, ec2_instance_type):
        pytest.skip(
            f"Image {pytorch_training} is incompatible with instance type {ec2_instance_type}"
        )
    execute_ec2_training_test(
        ec2_connection, pytorch_training, GDRCOPY_SANITY_TEST_CMD, enable_gdrcopy=True
    )
