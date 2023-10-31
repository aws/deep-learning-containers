import os

import pytest

from packaging.version import Version
from packaging.specifiers import SpecifierSet

import test.test_utils.ec2 as ec2_utils
from test.test_utils import (
    CONTAINER_TESTS_PREFIX,
    is_pr_context,
    is_efa_dedicated,
    get_framework_and_version_from_tag,
    get_cuda_version_from_tag,
)
from test.test_utils.ec2 import get_efa_ec2_instance_type, filter_efa_instance_type

PT_TE_TESTS_CMD = os.path.join(
    CONTAINER_TESTS_PREFIX, "transformerengine", "testPTTransformerEngine"
)


EC2_EFA_GPU_INSTANCE_TYPE_AND_REGION = get_efa_ec2_instance_type(
    default="p4d.24xlarge",
    filter_function=filter_efa_instance_type,
)


@pytest.mark.processor("gpu")
@pytest.mark.model("N/A")
@pytest.mark.integration("transformerengine")
@pytest.mark.usefixtures("sagemaker")
@pytest.mark.allow_p4de_use
@pytest.mark.parametrize("ec2_instance_type,region", EC2_EFA_GPU_INSTANCE_TYPE_AND_REGION)
@pytest.mark.skipif(
    is_pr_context() and not is_efa_dedicated(),
    reason="Skip heavy instance test in PR context unless explicitly enabled",
)
@pytest.mark.team("conda")
def test_pytorch_transformerengine(
    pytorch_training, ec2_connection, region, ec2_instance_type, gpu_only, py3_only
):
    _, image_framework_version = get_framework_and_version_from_tag(pytorch_training)
    image_framework_version = Version(image_framework_version)
    image_cuda_version = get_cuda_version_from_tag(pytorch_training)
    if not (image_framework_version in SpecifierSet(">=2.0") and image_cuda_version >= "cu121"):
        pytest.skip("Test requires CUDA12 and PT 2.0 or greater")

    ec2_utils.execute_ec2_training_test(ec2_connection, pytorch_training, PT_TE_TESTS_CMD)
