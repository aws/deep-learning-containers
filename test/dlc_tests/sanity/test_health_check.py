import os
import pytest

from invoke.context import Context
from test import test_utils

from test.test_utils import (
    CONTAINER_TESTS_PREFIX,
    LOGGER,
    ec2 as ec2_utils,
    is_pr_context,
    is_functionality_sanity_test_enabled,
)

EFA_LOCAL_TEST_CMD = os.path.join(
    CONTAINER_TESTS_PREFIX, "healthcheck_tests", "efa_checker_test.sh"
)
NCCL_LOCAL_TEST_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "healthcheck_tests", "nccl_test.sh")
PT_EC2_MULTI_GPU_INSTANCE_TYPE = ec2_utils.get_ec2_instance_type(
    default="g3.8xlarge",
    processor="gpu",
    filter_function=ec2_utils.filter_only_multi_gpu,
)


@pytest.mark.skip(
    "EFA healthcheck binaries are not maintained by DLC, we will skip these tests moving foward unless binaries are added otherwise."
)
@pytest.mark.usefixtures("sagemaker_only", "functionality_sanity")
@pytest.mark.usefixtures("pt201_and_above_only")
@pytest.mark.processor("gpu")
@pytest.mark.parametrize("ec2_instance_type", ["p3.2xlarge"], indirect=True)
@pytest.mark.model("N/A")
@pytest.mark.timeout(1200)
@pytest.mark.integration("health_check")
@pytest.mark.skipif(
    is_pr_context() and not is_functionality_sanity_test_enabled(),
    reason="Skip functionality sanity test in PR context if explicitly disabled",
)
def test_health_check_local_efa(pytorch_training, ec2_connection, gpu_only):
    # Run local EFA test on Pytorch DLC
    account_id = test_utils.get_account_id_from_image_uri(pytorch_training)
    image_region = test_utils.get_region_from_image_uri(pytorch_training)
    local_efa_timeout = 120
    LOGGER.info(f"test_health_check_local_efa pulling image: {pytorch_training}")
    test_utils.login_to_ecr_registry(ec2_connection, account_id, image_region)
    ec2_connection.run(f"docker pull {pytorch_training}", hide="out")

    image = pytorch_training
    container_name = test_utils.get_container_name("health_check", image)
    container_test_local_dir = os.path.join("$HOME", "container_tests")
    bin_bash_cmd = "--entrypoint /bin/bash "
    LOGGER.info(f"test_health_check_local_efa starting docker image: {pytorch_training}")
    ec2_connection.run(
        f"docker run --runtime=nvidia --gpus all --name {container_name} "
        f"-v {container_test_local_dir}:{os.path.join(os.sep, 'test')} -itd {bin_bash_cmd}{pytorch_training}",
        hide=False,
    )

    LOGGER.info(f"test_health_check_local_efa run {EFA_LOCAL_TEST_CMD} on container")
    executable = os.path.join(os.sep, "bin", "bash")
    execution_command = (
        f"docker exec --user root {container_name} {executable} -c '{EFA_LOCAL_TEST_CMD}'"
    )

    run_output = ec2_connection.run(execution_command, hide=False, timeout=local_efa_timeout)
    if not run_output.ok:
        raise RuntimeError(f"Image {image} EFA test {EFA_LOCAL_TEST_CMD} failed: {run_output} ")
