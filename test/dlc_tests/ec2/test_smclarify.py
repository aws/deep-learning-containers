import os

import pytest

from packaging.version import Version

from test.test_utils import CONTAINER_TESTS_PREFIX, get_cuda_version_from_tag
from test.test_utils.ec2 import get_ec2_instance_type
from test.dlc_tests.ec2 import smclarify_cases

SMCLARIFY_SCRIPT = os.path.join(CONTAINER_TESTS_PREFIX, "test_smclarify_bias_metrics.py")

SMCLARIFY_EC2_GPU_INSTANCE_TYPE = get_ec2_instance_type(default="g5.8xlarge", processor="gpu")
SMCLARIFY_EC2_CPU_INSTANCE_TYPE = get_ec2_instance_type(default="c5.2xlarge", processor="cpu")


# Adding separate tests to run on cpu instance for cpu image and gpu instance for gpu image.
# But the test behavior doesn't change for cpu or gpu image type.
@pytest.mark.skip_serialized_release_pt_test
@pytest.mark.usefixtures("sagemaker_only")
@pytest.mark.integration("smclarify_cpu")
@pytest.mark.model("N/A")
@pytest.mark.team("smclarify")
@pytest.mark.parametrize("ec2_instance_type", SMCLARIFY_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_smclarify_metrics_cpu(
    training,
    ec2_connection,
    ec2_instance_type,
    cpu_only,
    py3_only,
    tf23_and_above_only,
    mx18_and_above_only,
    pt16_and_above_only,
):
    smclarify_cases.smclarify_metrics_cpu(training, ec2_connection)


@pytest.mark.skip_serialized_release_pt_test
@pytest.mark.usefixtures("sagemaker_only")
@pytest.mark.integration("smclarify_gpu")
@pytest.mark.model("N/A")
@pytest.mark.team("smclarify")
@pytest.mark.parametrize("ec2_instance_type", SMCLARIFY_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_smclarify_metrics_gpu(
    training,
    ec2_connection,
    ec2_instance_type,
    gpu_only,
    py3_only,
    tf23_and_above_only,
    mx18_and_above_only,
    pt16_and_above_only,
):
    image_cuda_version = get_cuda_version_from_tag(training)
    if Version(image_cuda_version.strip("cu")) < Version("110"):
        pytest.skip("SmClarify is currently installed in cuda 11 gpu images and above")
    smclarify_cases.smclarify_metrics_gpu(training, ec2_connection)
