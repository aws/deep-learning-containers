import os
import test.test_utils as test_utils
from test.test_utils import (
    CONTAINER_TESTS_PREFIX,
    LOGGER,
    get_account_id_from_image_uri,
    get_framework_and_version_from_tag,
    is_nightly_context,
    is_tf_version,
    login_to_ecr_registry,
)
from test.test_utils.ec2 import get_ec2_instance_type

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version

SMDEBUG_SCRIPT = os.path.join(CONTAINER_TESTS_PREFIX, "testSmdebug")
SMPROFILER_SCRIPT = os.path.join(CONTAINER_TESTS_PREFIX, "testSmprofiler")


SMDEBUG_EC2_GPU_INSTANCE_TYPE = get_ec2_instance_type(default="g5.12xlarge", processor="gpu")
SMDEBUG_EC2_CPU_INSTANCE_TYPE = get_ec2_instance_type(default="c5.9xlarge", processor="cpu")


@pytest.mark.skip_smdebug_v1_test
@pytest.mark.usefixtures("feature_smdebug_present")
@pytest.mark.usefixtures("sagemaker_only")
@pytest.mark.integration("smdebug")
@pytest.mark.model("mnist")
@pytest.mark.parametrize("ec2_instance_type", SMDEBUG_EC2_GPU_INSTANCE_TYPE, indirect=True)
@pytest.mark.team("smdebug")
@pytest.mark.flaky(reruns=0)
def test_smdebug_gpu(
    training, ec2_connection, region, ec2_instance_type, gpu_only, py3_only, below_tf213_only
):
    if test_utils.is_image_incompatible_with_instance_type(training, ec2_instance_type):
        pytest.skip(f"Image {training} is incompatible with instance type {ec2_instance_type}")

    _, image_framework_version = get_framework_and_version_from_tag(training)
    if (
        "trcomp" in training
        and "pytorch" in training
        and Version(image_framework_version) in SpecifierSet("<2.0")
    ):
        pytest.skip(f"Image {training} doesn't support s3. Hence test is skipped.")
    smdebug_test_timeout = 2400
    if is_tf_version("1", training):
        if is_nightly_context():
            smdebug_test_timeout = 7200
        else:
            pytest.skip(
                "TF1 gpu smdebug tests can take up to 2 hours, thus we are only running in nightly context"
            )

    run_smdebug_test(
        training,
        ec2_connection,
        region,
        ec2_instance_type,
        docker_runtime="--runtime=nvidia --gpus all",
        container_name="smdebug-gpu",
        timeout=smdebug_test_timeout,
    )


@pytest.mark.skip_smdebug_v1_test
@pytest.mark.usefixtures("feature_smdebug_present")
@pytest.mark.usefixtures("sagemaker_only")
@pytest.mark.integration("smprofiler")
@pytest.mark.model("mnist")
@pytest.mark.team("smdebug")
@pytest.mark.parametrize("ec2_instance_type", SMDEBUG_EC2_GPU_INSTANCE_TYPE, indirect=True)
@pytest.mark.flaky(reruns=0)
def test_smprofiler_gpu(
    training,
    ec2_connection,
    region,
    ec2_instance_type,
    gpu_only,
    py3_only,
    tf23_and_above_only,
    pt16_and_above_only,
    below_tf213_only,
):
    # Running the profiler tests for pytorch and tensorflow2 frameworks only.
    # This code needs to be modified past reInvent 2020
    if test_utils.is_image_incompatible_with_instance_type(training, ec2_instance_type):
        pytest.skip(f"Image {training} is incompatible with instance type {ec2_instance_type}")
    _, image_framework_version = get_framework_and_version_from_tag(training)
    if (
        "trcomp" in training
        and "pytorch" in training
        and Version(image_framework_version) in SpecifierSet("<2.0")
    ):
        pytest.skip(f"Image {training} doesn't support s3. Hence test is skipped.")
    framework = get_framework_from_image_uri(training)
    if framework not in ["pytorch", "tensorflow2"]:
        return
    smdebug_test_timeout = 2400
    run_smprofiler_test(
        training,
        ec2_connection,
        region,
        ec2_instance_type,
        docker_runtime="--runtime=nvidia --gpus all",
        container_name="smdebug-gpu",
        timeout=smdebug_test_timeout,
    )


@pytest.mark.skip_smdebug_v1_test
@pytest.mark.usefixtures("feature_smdebug_present")
@pytest.mark.usefixtures("sagemaker_only")
@pytest.mark.flaky(reruns=0)
@pytest.mark.integration("smdebug")
@pytest.mark.model("mnist")
@pytest.mark.team("smdebug")
@pytest.mark.parametrize("ec2_instance_type", SMDEBUG_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_smdebug_cpu(
    training, ec2_connection, region, ec2_instance_type, cpu_only, py3_only, below_tf213_only
):
    run_smdebug_test(training, ec2_connection, region, ec2_instance_type)


@pytest.mark.skip_smdebug_v1_test
@pytest.mark.usefixtures("feature_smdebug_present")
@pytest.mark.usefixtures("sagemaker_only")
@pytest.mark.flaky(reruns=0)
@pytest.mark.integration("smdebug")
@pytest.mark.model("mnist")
@pytest.mark.team("smdebug")
@pytest.mark.parametrize("ec2_instance_type", SMDEBUG_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_smprofiler_cpu(
    training,
    ec2_connection,
    region,
    ec2_instance_type,
    cpu_only,
    py3_only,
    tf23_and_above_only,
    pt16_and_above_only,
    below_tf213_only,
):
    # Running the profiler tests for pytorch and tensorflow2 frameworks only.
    # This code needs to be modified past reInvent 2020
    framework = get_framework_from_image_uri(training)
    if framework not in ["pytorch", "tensorflow2"]:
        return
    run_smprofiler_test(training, ec2_connection, region, ec2_instance_type)


class SMDebugTestFailure(Exception):
    pass


def run_smdebug_test(
    image_uri,
    ec2_connection,
    region,
    ec2_instance_type,
    docker_runtime="",
    container_name="smdebug",
    test_script=SMDEBUG_SCRIPT,
    timeout=2400,
):
    large_shm_instance_types = ("g5.12xlarge", "m5.16xlarge")
    shm_setting = " --shm-size=1g " if ec2_instance_type in large_shm_instance_types else " "
    framework = get_framework_from_image_uri(image_uri)
    container_test_local_dir = os.path.join("$HOME", "container_tests")
    account_id = get_account_id_from_image_uri(image_uri)
    login_to_ecr_registry(ec2_connection, account_id, region)
    # Do not add -q to docker pull as it leads to a hang for huge images like trcomp
    ec2_connection.run(f"docker pull {image_uri}")

    try:
        ec2_connection.run(
            f"docker run {docker_runtime} --name {container_name} -v "
            f"{container_test_local_dir}:{os.path.join(os.sep, 'test')}{shm_setting}{image_uri} "
            f"./{test_script} {framework}",
            hide=True,
            timeout=timeout,
        )
    except Exception as e:
        debug_output = ec2_connection.run(f"docker logs {container_name}")
        debug_stdout = debug_output.stdout
        if "All SMDebug tests succeeded!" in debug_stdout:
            LOGGER.warning(
                f"SMDebug tests succeeded, but there is an issue with fabric:\n{e}:\nTest output:\n{debug_stdout}"
            )
            return
        raise SMDebugTestFailure(
            f"SMDebug test failed on {image_uri} on {ec2_instance_type}. Full output:\n{debug_stdout}"
        ) from e


def run_smprofiler_test(
    image_uri,
    ec2_connection,
    region,
    ec2_instance_type,
    docker_runtime="",
    container_name="smdebug",
    test_script=SMPROFILER_SCRIPT,
    timeout=2400,
):
    large_shm_instance_types = ("g5.12xlarge", "m5.16xlarge")
    shm_setting = " --shm-size=1g " if ec2_instance_type in large_shm_instance_types else " "
    framework = get_framework_from_image_uri(image_uri)
    container_test_local_dir = os.path.join("$HOME", "container_tests")
    account_id = get_account_id_from_image_uri(image_uri)
    login_to_ecr_registry(ec2_connection, account_id, region)
    # Do not add -q to docker pull as it leads to a hang for huge images like trcomp
    ec2_connection.run(f"docker pull {image_uri}")

    try:
        ec2_connection.run(
            f"docker run {docker_runtime} --name {container_name} -v "
            f"{container_test_local_dir}:{os.path.join(os.sep, 'test')}{shm_setting}{image_uri} "
            f"./{test_script} {framework}",
            hide=True,
            timeout=timeout,
        )
    except Exception as e:
        debug_output = ec2_connection.run(f"docker logs {container_name}")
        debug_stdout = debug_output.stdout
        if "All SMprofiler tests succeeded!" in debug_stdout:
            LOGGER.warning(
                f"SMProfiler tests succeeded, but there is an issue with fabric:\n{e}:\nTest output:\n{debug_stdout}"
            )
            return
        raise SMDebugTestFailure(
            f"SMProfiler test failed on {image_uri} on {ec2_instance_type}. Full output:\n{debug_stdout}"
        ) from e


def get_framework_from_image_uri(image_uri):
    frameworks = ("tensorflow", "mxnet", "pytorch")
    for framework in frameworks:
        if framework in image_uri:
            if framework == "tensorflow" and is_tf_version("2", image_uri):
                return "tensorflow2"
            return framework
    raise RuntimeError(f"Could not find any framework {frameworks} in {image_uri}")
