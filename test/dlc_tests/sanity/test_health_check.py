import os
import pytest

from invoke.context import Context
from test import test_utils

from test.test_utils import (
    CONTAINER_TESTS_PREFIX,
    LOGGER,
    get_container_name,
    run_cmd_on_container,
    start_container,
)

DCGM_TEST_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "healthcheck_tests", "dcgm_test.sh")
EFA_LOCAL_TEST_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "healthcheck_tests", "efa_checker_test.sh")
NCCL_LOCAL_TEST_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "healthcheck_tests", "nccl_test.sh")

"""
@pytest.mark.usefixtures("sagemaker_only")
@pytest.mark.usefixtures("gpu_only")
@pytest.mark.usefixtures("pt201_and_above_only")
@pytest.mark.processor("gpu")
@pytest.mark.parametrize("ec2_instance_type", ["p3.16xlarge"], indirect=True)
@pytest.mark.model("N/A")
@pytest.mark.integration("health_check")
def test_health_check_dcgm(pytorch_training):
    try:
        image = pytorch_training
        ctx = Context()
        container_name = test_utils.get_container_name("health_check", image)
        test_utils.start_container(container_name, image, ctx)
        command_output = test_utils.run_cmd_on_container(container_name, ctx, DCGM_TEST_CMD, warn=True)
        command_stdout = command_output.stdout.strip()
        if command_output.return_code != 0:
            raise RuntimeError(
                f"Image {image} DCGM test {DCGM_TEST_CMD} failed: {command_output} "
            )
    finally:
        test_utils.stop_and_remove_container(container_name, ctx)
"""


@pytest.mark.usefixtures("sagemaker_only")
@pytest.mark.usefixtures("pt201_and_above_only")
@pytest.mark.processor("gpu")
@pytest.mark.parametrize("ec2_instance_type", ["p3.8xlarge"], indirect=True)
@pytest.mark.model("N/A")
@pytest.mark.timeout(1200)
@pytest.mark.integration("health_check")
def test_health_check_dcgm(gpu, ec2_connection):
    """
    Run local DCGM test on Pytorch DLC
    """
    LOGGER.info(f"test_health_check_dcgm pulling: {gpu}")
    account_id = test_utils.get_account_id_from_image_uri(gpu)
    image_region = test_utils.get_region_from_image_uri(gpu)
    test_utils.login_to_ecr_registry(ec2_connection, account_id, image_region)
    ec2_connection.run(f"{DCGM_TEST_CMD}")

"""
@pytest.mark.usefixtures("sagemaker_only")
@pytest.mark.usefixtures("gpu_only")
@pytest.mark.usefixtures("pt201_and_above_only")
@pytest.mark.parametrize("ec2_instance_type", ["p3.16xlarge"], indirect=True)
@pytest.mark.model("N/A")
@pytest.mark.integration("health_check")
def test_health_check_local_nccl(pytorch_training):
    try:
        image = pytorch_training
        ctx = Context()
        container_name = test_utils.get_container_name("health_check", image)
        test_utils.start_container(container_name, image, ctx)
        command_output = test_utils.run_cmd_on_container(container_name, ctx, NCCL_LOCAL_TEST_CMD, warn=True)
        command_stdout = command_output.stdout.strip()
        if command_output.return_code != 0:
            raise RuntimeError(
                f"Image {image} local NCCL test {NCCL_LOCAL_TEST_CMD} failed: {command_output} "
            )
    finally:
        test_utils.stop_and_remove_container(container_name, ctx)
"""

@pytest.mark.usefixtures("sagemaker_only")
@pytest.mark.usefixtures("pt201_and_above_only")
@pytest.mark.processor("gpu")
@pytest.mark.parametrize("ec2_instance_type", ["p3.8xlarge"], indirect=True)
@pytest.mark.model("N/A")
@pytest.mark.timeout(600)
@pytest.mark.integration("health_check")
def test_health_check_local_nccl(gpu, ec2_connection):
    """
    Run local DCGM test on Pytorch DLC
    """
    LOGGER.info(f"test_health_check_local_nccl pulling: {gpu}")
    account_id = test_utils.get_account_id_from_image_uri(gpu)
    image_region = test_utils.get_region_from_image_uri(gpu)
    test_utils.login_to_ecr_registry(ec2_connection, account_id, image_region)
    ec2_connection.run(f"{NCCL_LOCAL_TEST_CMD}")


"""
@pytest.mark.usefixtures("sagemaker_only")
@pytest.mark.usefixtures("gpu_only")
@pytest.mark.usefixtures("pt201_and_above_only")
@pytest.mark.parametrize("ec2_instance_type", ["p3.16xlarge"], indirect=True)
@pytest.mark.model("N/A")
@pytest.mark.integration("health_check")
def test_health_check_local_efa(pytorch_training, ec2_client, ec2_instance, ec2_connection):
    try:
        image = pytorch_training
        ctx = Context()
        container_name = test_utils.get_container_name("health_check", image)
        test_utils.start_container(container_name, image, ctx)

        ec2_connection.run(f"{EFA_LOCAL_TEST_CMD}")
        command_output = test_utils.run_cmd_on_container(container_name, ctx, EFA_LOCAL_TEST_CMD, warn=True)
        command_stdout = command_output.stdout.strip()
        if command_output.return_code != 0:
            raise RuntimeError(
                f"Image {image} local EFA test {EFA_LOCAL_TEST_CMD} failed: {command_output} "
            )
    finally:
        test_utils.stop_and_remove_container(container_name, ctx)
"""

@pytest.mark.usefixtures("sagemaker_only")
@pytest.mark.usefixtures("pt201_and_above_only")
@pytest.mark.processor("gpu")
@pytest.mark.parametrize("ec2_instance_type", ["p3.8xlarge"], indirect=True)
@pytest.mark.model("N/A")
@pytest.mark.timeout(1200)
@pytest.mark.integration("health_check")
def test_health_check_local_efa(gpu, ec2_connection):
    """
    Run local DCGM test on Pytorch DLC
    """
    LOGGER.info(f"test_health_check_local efa pulling: {gpu}")
    account_id = test_utils.get_account_id_from_image_uri(gpu)
    image_region = test_utils.get_region_from_image_uri(gpu)
    test_utils.login_to_ecr_registry(ec2_connection, account_id, image_region)
    ec2_connection.run(f"{EFA_LOCAL_TEST_CMD }")