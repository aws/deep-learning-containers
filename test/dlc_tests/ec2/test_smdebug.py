import os

import pytest

from test.test_utils import CONTAINER_TESTS_PREFIX, LOGGER, is_tf_version
from test.test_utils.ec2 import get_ec2_instance_type


SMDEBUG_SCRIPT = os.path.join(CONTAINER_TESTS_PREFIX, "testSmdebug")


SMDEBUG_EC2_GPU_INSTANCE_TYPE = get_ec2_instance_type(default="p3.8xlarge", processor="gpu")
SMDEBUG_EC2_CPU_INSTANCE_TYPE = get_ec2_instance_type(default="c4.8xlarge", processor="cpu")


@pytest.mark.integration("smdebug")
@pytest.mark.model("mnist")
@pytest.mark.parametrize("ec2_instance_type", SMDEBUG_EC2_GPU_INSTANCE_TYPE, indirect=True)
@pytest.mark.flaky(reruns=0)
def test_smdebug_gpu(training, ec2_connection, region, ec2_instance_type, gpu_only, py3_only):
    run_smdebug_test(
        training,
        ec2_connection,
        region,
        ec2_instance_type,
        docker_executable="nvidia-docker",
        container_name="smdebug-gpu",
    )


@pytest.mark.flaky(reruns=0)
@pytest.mark.integration("smdebug")
@pytest.mark.model("mnist")
@pytest.mark.parametrize("ec2_instance_type", SMDEBUG_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_smdebug_cpu(training, ec2_connection, region, ec2_instance_type, cpu_only, py3_only):
    run_smdebug_test(training, ec2_connection, region, ec2_instance_type)


class SMDebugTestFailure(Exception):
    pass


def run_smdebug_test(
    image_uri,
    ec2_connection,
    region,
    ec2_instance_type,
    docker_executable="docker",
    container_name="smdebug",
    test_script=SMDEBUG_SCRIPT,
):
    large_shm_instance_types = ("p2.8xlarge", "m4.16xlarge")
    shm_setting = " --shm-size=1g " if ec2_instance_type in large_shm_instance_types else " "
    framework = get_framework_from_image_uri(image_uri)
    container_test_local_dir = os.path.join("$HOME", "container_tests")
    ec2_connection.run(f"$(aws ecr get-login --no-include-email --region {region})", hide=True)

    try:
        ec2_connection.run(
            f"{docker_executable} run --name {container_name} -v "
            f"{container_test_local_dir}:{os.path.join(os.sep, 'test')}{shm_setting}{image_uri} "
            f"./{test_script} {framework}",
            hide=True,
            timeout=2400,
        )
    except Exception as e:
        debug_output = ec2_connection.run(f"docker logs {container_name}")
        debug_stdout = debug_output.stdout
        if "All SMDebug tests succeeded!" in debug_stdout:
            LOGGER.warning(
                f"SMDebug tests succeeded, but there is an issue with fabric. Error:\n{e}\nTest output:\n{debug_stdout}"
            )
            return
        raise SMDebugTestFailure(
            f"SMDebug test failed on {image_uri} on {ec2_instance_type}. Full output:\n{debug_stdout}"
        ) from e


def get_framework_from_image_uri(image_uri):
    frameworks = ("tensorflow", "mxnet", "pytorch")
    for framework in frameworks:
        if framework in image_uri:
            if framework == "tensorflow" and is_tf_version("2", image_uri):
                return "tensorflow2"
            return framework
    raise RuntimeError(f"Could not find any framework {frameworks} in {image_uri}")
