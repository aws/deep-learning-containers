import os
import pytest

from invoke.context import Context
from test import test_utils

from test.test_utils import (
    CONTAINER_TESTS_PREFIX,
    get_container_name,
    run_cmd_on_container,
    start_container,
)

DCGM_TEST_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "healthcheck_tests", "dcgm_test.sh")
EFA_LOCAL_TEST_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "healthcheck_tests", "efa_checker_test.sh")
NCCL_LOCAL_TEST_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "healthcheck_tests", "nccl_test.sh")

@pytest.mark.usefixtures("sagemaker_only")
@pytest.mark.usefixtures("gpu_only")
@pytest.mark.usefixtures("pt201_and_above_only")
@pytest.mark.model("N/A")
@pytest.mark.integration("health_check")
def test_health_check_dcgm(pytorch_training):
    """
    Run local DCGM test on Pytorch DLC
    """
    try:
        image = pytorch_training
        ctx = Context()
        container_name = test_utils.get_container_name("health_check", image)
        test_utils.start_container(container_name, image, ctx)
        command_output = test_utils.run_cmd_on_container(container_name, ctx, DCGM_TEST_CMD, warn=True)
        command_stdout = command_output.stdout.strip()
        if command_output.return_code != 0:
            raise RuntimeError(
                f"Image {image} DCGM test failed: {command_stdout} "
            )
    finally:
        test_utils.stop_and_remove_container(container_name, ctx)

@pytest.mark.usefixtures("sagemaker_only")
@pytest.mark.usefixtures("gpu_only")
@pytest.mark.usefixtures("pt201_and_above_only")
@pytest.mark.model("N/A")
@pytest.mark.integration("health_check")
def test_health_check_local_nccl(pytorch_training):
    """
    Run local NCCL test on Pytorch DLC
    """
    try:
        image = pytorch_training
        ctx = Context()
        container_name = test_utils.get_container_name("health_check", image)
        test_utils.start_container(container_name, image, ctx)
        command_output = test_utils.run_cmd_on_container(container_name, ctx, NCCL_LOCAL_TEST_CMD, warn=True)
        command_stdout = command_output.stdout.strip()
        if command_output.return_code != 0:
            raise RuntimeError(
                f"Image {image} local NCCL test failed: {command_stdout} "
            )
    finally:
        test_utils.stop_and_remove_container(container_name, ctx)

@pytest.mark.usefixtures("sagemaker_only")
@pytest.mark.usefixtures("gpu_only")
@pytest.mark.usefixtures("pt201_and_above_only")
@pytest.mark.model("N/A")
@pytest.mark.integration("health_check")
def test_health_check_local_efa(pytorch_training):
    """
    Run local EFA test on Pytorch DLC
    """
    try:
        image = pytorch_training
        ctx = Context()
        container_name = test_utils.get_container_name("health_check", image)
        test_utils.start_container(container_name, image, ctx)
        command_output = test_utils.run_cmd_on_container(container_name, ctx, EFA_LOCAL_TEST_CMD, warn=True)
        command_stdout = command_output.stdout.strip()
        if command_output.return_code != 0:
            raise RuntimeError(
                f"Image {image} local EFA test failed: {command_stdout} "
            )
    finally:
        test_utils.stop_and_remove_container(container_name, ctx)