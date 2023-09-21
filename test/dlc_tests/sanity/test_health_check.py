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

@pytest.mark.usefixtures("sagemaker_only")
@pytest.mark.usefixtures("pt201_and_above_only")
@pytest.mark.processor("gpu")
@pytest.mark.parametrize("ec2_instance_type", ["g3.8xlarge"], indirect=True)
@pytest.mark.model("N/A")
@pytest.mark.timeout(1200)
@pytest.mark.integration("health_check")
def test_health_check_dcgm(gpu, ec2_connection):
    """
    Run local DCGM test on Pytorch DLC
    """
    docker_cmd = "nvidia-docker"
    account_id = test_utils.get_account_id_from_image_uri(gpu)
    image_region = test_utils.get_region_from_image_uri(gpu)
    dcgm_l1_timeout = 300
    LOGGER.info(f"test_health_check_dcgm pulling image: {gpu}")
    test_utils.login_to_ecr_registry(ec2_connection, account_id, image_region)
    ec2_connection.run(f"{docker_cmd} pull {gpu}", hide="out")

    image = gpu
    container_name = test_utils.get_container_name("health_check", image)
    container_test_local_dir = os.path.join("$HOME", "container_tests")
    bin_bash_cmd = "--entrypoint /bin/bash "
    LOGGER.info(f"test_health_check_dcgm starting docker image: {gpu}")
    ec2_connection.run(
        f"{docker_cmd} run --name {container_name} "
        f"-v {container_test_local_dir}:{os.path.join(os.sep, 'test')} -itd {bin_bash_cmd}{gpu}",
        hide=False,
    )

    LOGGER.info(f"test_health_check_dcgm run {DCGM_TEST_CMD} on container")
    executable = os.path.join(os.sep, "bin", "bash")
    execution_command = (
        f"{docker_cmd} exec --user root {container_name} {executable} -c '{DCGM_TEST_CMD}'"
    )

    run_output = ec2_connection.run(execution_command, hide=False, timeout=dcgm_l1_timeout)
    if not run_output.ok:
        raise RuntimeError(
            f"Image {image} DCGM test {DCGM_TEST_CMD} failed: {run_output}"
        )

@pytest.mark.usefixtures("sagemaker_only")
@pytest.mark.usefixtures("pt201_and_above_only")
@pytest.mark.processor("gpu")
@pytest.mark.parametrize("ec2_instance_type", ["g3.8xlarge"], indirect=True)
@pytest.mark.model("N/A")
@pytest.mark.timeout(600)
@pytest.mark.integration("health_check")
def test_health_check_local_nccl(gpu, ec2_connection):
    """
    Run local DCGM test on Pytorch DLC
    """
    docker_cmd = "nvidia-docker"
    account_id = test_utils.get_account_id_from_image_uri(gpu)
    image_region = test_utils.get_region_from_image_uri(gpu)
    local_nccl_timeout = 240
    LOGGER.info(f"test_health_check_local_nccl pulling image: {gpu}")
    test_utils.login_to_ecr_registry(ec2_connection, account_id, image_region)
    ec2_connection.run(f"{docker_cmd} pull {gpu}", hide="out")

    image = gpu
    container_name = test_utils.get_container_name("health_check", image)
    container_test_local_dir = os.path.join("$HOME", "container_tests")
    bin_bash_cmd = "--entrypoint /bin/bash "
    LOGGER.info(f"test_health_check_local_nccl starting docker image: {gpu}")
    ec2_connection.run(
        f"{docker_cmd} run --name {container_name} "
        f"-v {container_test_local_dir}:{os.path.join(os.sep, 'test')} -itd {bin_bash_cmd}{gpu}",
        hide=False,
    )

    LOGGER.info(f"test_health_check_local_nccl run {NCCL_LOCAL_TEST_CMD} on container")
    executable = os.path.join(os.sep, "bin", "bash")
    execution_command = (
        f"{docker_cmd} exec --user root {container_name} {executable} -c '{NCCL_LOCAL_TEST_CMD}'"
    )

    run_output = ec2_connection.run(execution_command, hide=False, timeout=local_nccl_timeout)
    if not run_output.ok:
        raise RuntimeError(
            f"Image {image} NCCL test {NCCL_LOCAL_TEST_CMD} failed: {run_output}"
        )

@pytest.mark.usefixtures("sagemaker_only")
@pytest.mark.usefixtures("pt201_and_above_only")
@pytest.mark.processor("gpu")
@pytest.mark.parametrize("ec2_instance_type", ["g3.8xlarge"], indirect=True)
@pytest.mark.model("N/A")
@pytest.mark.timeout(1200)
@pytest.mark.integration("health_check")
def test_health_check_local_efa(gpu, ec2_connection):
    """
    Run local EFA test on Pytorch DLC
    """
    docker_cmd = "nvidia-docker"
    account_id = test_utils.get_account_id_from_image_uri(gpu)
    image_region = test_utils.get_region_from_image_uri(gpu)
    local_efa_timeout = 120
    LOGGER.info(f"test_health_check_local_efa pulling image: {gpu}")
    test_utils.login_to_ecr_registry(ec2_connection, account_id, image_region)
    ec2_connection.run(f"{docker_cmd} pull {gpu}", hide="out")

    image = gpu
    container_name = test_utils.get_container_name("health_check", image)
    container_test_local_dir = os.path.join("$HOME", "container_tests")
    bin_bash_cmd = "--entrypoint /bin/bash "
    LOGGER.info(f"test_health_check_local_efa starting docker image: {gpu}")
    ec2_connection.run(
        f"{docker_cmd} run --name {container_name} "
        f"-v {container_test_local_dir}:{os.path.join(os.sep, 'test')} -itd {bin_bash_cmd}{gpu}",
        hide=False,
    )

    LOGGER.info(f"test_health_check_local_efa run {EFA_LOCAL_TEST_CMD} on container")
    executable = os.path.join(os.sep, "bin", "bash")
    execution_command = (
        f"{docker_cmd} exec --user root {container_name} {executable} -c '{EFA_LOCAL_TEST_CMD}'"
    )

    run_output = ec2_connection.run(execution_command, hide=False, timeout=local_efa_timeout)
    if not run_output.ok:
        raise RuntimeError(
            f"Image {image} EFA test {EFA_LOCAL_TEST_CMD} failed: {run_output} "
        )