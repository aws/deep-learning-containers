import os

import pytest

from test.test_utils import CONTAINER_TESTS_PREFIX


SMDEBUG_SCRIPT = os.path.join(CONTAINER_TESTS_PREFIX, "testSmdebug")


@pytest.mark.parametrize("ec2_instance_type", ["p3.8xlarge"], indirect=True)
def test_smdebug_gpu(image, ec2_connection, gpu_only, py3_only):
    if "inference" in image:
        pytest.skip("Skipping testSMDebug on inference images")

    framework = get_framework_from_image_uri(image)
    container_test_local_dir = os.path.join("$HOME", "container_tests")
    ec2_connection.run(
        f"$(aws ecr get-login --no-include-email --region {region})", hide=True
    )

    ec2_connection.run(
        f"""nvidia-docker run -v {container_test_local_dir}:{os.path.join(os.sep, 'test')} {image} 
            {os.path.join(os.sep, 'bin', 'bash')} -c "{SMDEBUG_SCRIPT} '{framework}'" """,
        hide=True,
    )


@pytest.mark.parametrize("ec2_instance_type", ["c5.9xlarge"], indirect=True)
def test_smdebug_cpu(image, ec2_connection, region, cpu_only, py3_only, framework=None):
    if "inference" in image:
        pytest.skip("Skipping testSMDebug on inference images")

    framework = get_framework_from_image_uri(image)
    container_test_local_dir = os.path.join("$HOME", "container_tests")
    ec2_connection.run(
        f"$(aws ecr get-login --no-include-email --region {region})", hide=True
    )

    ec2_connection.run(
        f"""docker run -v {container_test_local_dir}:{os.path.join(os.sep, 'test')} {image} 
            {os.path.join(os.sep, 'bin', 'bash')} -c "{SMDEBUG_SCRIPT} '{framework}'" """,
        hide=True,
    )


def get_framework_from_image_uri(image_uri):
    frameworks = ("tensorflow", "mxnet", "pytorch")
    for framework in frameworks:
        if framework in image_uri:
            return framework
    raise RuntimeError(f"Could not find any framework {frameworks} in {image_uri}")
