import os

import pytest

from test.test_utils import CONTAINER_TESTS_PREFIX, LOGGER, is_tf2, is_tf1
from test.test_utils.ec2 import get_ec2_instance_type


SMDEBUG_SCRIPT = os.path.join(CONTAINER_TESTS_PREFIX, "testSmdebug")


SMDEBUG_EC2_GPU_INSTANCE_TYPE = get_ec2_instance_type(default="p3.8xlarge", processor="gpu")
SMDEBUG_EC2_CPU_INSTANCE_TYPE = get_ec2_instance_type(default="c5.9xlarge", processor="cpu")


@pytest.mark.integration("smdebug")
@pytest.mark.model("mnist")
@pytest.mark.parametrize("ec2_instance_type", SMDEBUG_EC2_GPU_INSTANCE_TYPE, indirect=True)
@pytest.mark.flaky(reruns=0)
def test_smdebug_gpu(training, ec2_connection, region, gpu_only, py3_only):
    # TODO: Remove this once test timeout has been debugged (failures especially on p2.8xlarge)
    if is_tf2(training) and "2.3.0" in training and "p2.8xlarge" in SMDEBUG_EC2_GPU_INSTANCE_TYPE:
        pytest.skip("Currently skipping for TF2.3.0 on p2.8xlarge until the issue is fixed")
    if is_tf1(training):
        pytest.skip("Currently skipping for TF1 until the issue is fixed")
    run_smdebug_test(training, ec2_connection, region, docker_executable="nvidia-docker", container_name="smdebug-gpu")


@pytest.mark.flaky(reruns=0)
@pytest.mark.integration("smdebug")
@pytest.mark.model("mnist")
@pytest.mark.parametrize("ec2_instance_type", SMDEBUG_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_smdebug_cpu(training, ec2_connection, region, cpu_only, py3_only):
    # TODO: Remove this once test timeout has been debugged (failures especially on m4.16xlarge)
    if is_tf2(training) and "m4.16xlarge" in SMDEBUG_EC2_CPU_INSTANCE_TYPE:
        pytest.skip("Currently skipping for TF2 on m4.16xlarge until the issue is fixed")
    if is_tf1(training):
        pytest.skip("Currently skipping for TF1 until the issue is fixed")
    run_smdebug_test(training, ec2_connection, region)


def run_smdebug_test(
    image_uri,
    ec2_connection,
    region,
    docker_executable="docker",
    container_name="smdebug",
    test_script=SMDEBUG_SCRIPT,
    logfile="output.log",
):
    framework = get_framework_from_image_uri(image_uri)
    container_test_local_dir = os.path.join("$HOME", "container_tests")
    ec2_connection.run(f"$(aws ecr get-login --no-include-email --region {region})", hide=True)

    ec2_connection.run(
        f"{docker_executable} run --name {container_name} -v "
        f"{container_test_local_dir}:{os.path.join(os.sep, 'test')} -itd {image_uri}",
        hide=True,
    )

    try:
        test_output = ec2_connection.run(
            f"{docker_executable} exec --user root {container_name} "
            f"/bin/bash -c '{test_script} {framework}' | tee {logfile}",
            hide=True,
            warn=True,
            timeout=3000,
        )
    except Exception:
        debug_output = ec2_connection.run(f"cat {logfile}")
        LOGGER.error(f"Caught exception while trying to run test via fabric. Output: {debug_output.stdout}")
        raise

    # LOGGER.info(test_output.stdout)  # Uncomment this line for a complete log dump

    assert test_output.ok, f"SMDebug tests failed. Output:\n{test_output.stdout}"


def get_framework_from_image_uri(image_uri):
    frameworks = ("tensorflow", "mxnet", "pytorch")
    for framework in frameworks:
        if framework in image_uri:
            if framework == "tensorflow" and is_tf2(image_uri):
                return "tensorflow2"
            return framework
    raise RuntimeError(f"Could not find any framework {frameworks} in {image_uri}")
