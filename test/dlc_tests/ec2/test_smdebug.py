import os

import pytest

from test.test_utils import CONTAINER_TESTS_PREFIX, is_tf2
from test.test_utils.ec2 import get_ec2_instance_type


SMDEBUG_SCRIPT = os.path.join(CONTAINER_TESTS_PREFIX, "testSmdebug")


# TODO: Set enable_p3dn=True when releasing
SMDEBUG_EC2_GPU_INSTANCE_TYPE = get_ec2_instance_type(default="p2.8xlarge", processor="gpu")
SMDEBUG_EC2_CPU_INSTANCE_TYPE = get_ec2_instance_type(default="c5.9xlarge", processor="cpu")


@pytest.mark.parametrize("ec2_instance_type", SMDEBUG_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_smdebug_gpu(training, ec2_connection, region, gpu_only, py3_only):
    test_script = SMDEBUG_SCRIPT
    framework = get_framework_from_image_uri(training)
    container_test_local_dir = os.path.join("$HOME", "container_tests")
    ec2_connection.run(
        f"$(aws ecr get-login --no-include-email --region {region})", hide=True
    )

    ec2_connection.run(
        f"nvidia-docker run --name smdebug-gpu -v "
        f"{container_test_local_dir}:{os.path.join(os.sep, 'test')} -itd {training}",
        hide=True,
    )

    ec2_connection.run(
        f"nvidia-docker exec --user root smdebug-gpu /bin/bash -c '{test_script} {framework}'",
        hide=True,
    )


@pytest.mark.parametrize("ec2_instance_type", SMDEBUG_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_smdebug_cpu(training, ec2_connection, region, cpu_only, py3_only):
    test_script = SMDEBUG_SCRIPT
    framework = get_framework_from_image_uri(training)
    container_test_local_dir = os.path.join("$HOME", "container_tests")
    ec2_connection.run(
        f"$(aws ecr get-login --no-include-email --region {region})", hide=True
    )

    ec2_connection.run(
        f"docker run --name smdebug-cpu -v {container_test_local_dir}:{os.path.join(os.sep, 'test')} -itd {training}",
        hide=True,
    )

    ec2_connection.run(
        f"docker exec --user root smdebug-cpu /bin/bash -c '{test_script} {framework}'",
        hide=True,
    )


def get_framework_from_image_uri(image_uri):
    frameworks = ("tensorflow", "mxnet", "pytorch")
    for framework in frameworks:
        if framework in image_uri:
            if framework == "tensorflow" and is_tf2(image_uri):
                return "tensorflow2"
            return framework
    raise RuntimeError(f"Could not find any framework {frameworks} in {image_uri}")
