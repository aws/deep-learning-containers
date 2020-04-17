import os
import re

import pytest

from test.test_utils import CONTAINER_TESTS_PREFIX, is_tf2


SMDEBUG_SCRIPT = os.path.join(CONTAINER_TESTS_PREFIX, "testSmdebug")


@pytest.mark.parametrize("ec2_instance_type", ["p3.8xlarge"], indirect=True)
def test_smdebug_gpu(training, ec2_connection, region, gpu_only, py3_only):
    test_script = SMDEBUG_SCRIPT
    framework = get_framework_from_image_uri(training)
    container_test_local_dir = os.path.join("$HOME", "container_tests")
    ec2_connection.run(
        f"$(aws ecr get-login --no-include-email --region {region})", hide=True
    )

    ec2_connection.run(
        f"nvidia-docker run -v {container_test_local_dir}:{os.path.join(os.sep, 'test')} "
        f"--entrypoint \"{os.path.join(os.sep, 'bin', 'bash')} -c '{test_script} {framework}'\" {training} ",
        hide=True,
    )


@pytest.mark.parametrize("ec2_instance_type", ["c5.9xlarge"], indirect=True)
def test_smdebug_cpu(training, ec2_connection, region, cpu_only, py3_only):
    test_script = SMDEBUG_SCRIPT
    framework = get_framework_from_image_uri(training)
    container_test_local_dir = os.path.join("$HOME", "container_tests")
    ec2_connection.run(
        f"$(aws ecr get-login --no-include-email --region {region})", hide=True
    )

    ec2_connection.run(
        f"docker run -v {container_test_local_dir}:{os.path.join(os.sep, 'test')} "
        f"--entrypoint \"{os.path.join(os.sep, 'bin', 'bash')} -c '{test_script} {framework}'\" {training} ",
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
