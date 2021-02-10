import os

import pytest

from test.test_utils import CONTAINER_TESTS_PREFIX, LOGGER
from test.test_utils import get_cuda_version_from_tag
from test.test_utils.ec2 import get_ec2_instance_type

SMCLARIFY_SCRIPT = os.path.join(CONTAINER_TESTS_PREFIX, "test_smclarify_bias_metrics.py")

SMCLARIFY_EC2_GPU_INSTANCE_TYPE = get_ec2_instance_type(default="p3.2xlarge", processor="gpu")
SMCLARIFY_EC2_CPU_INSTANCE_TYPE = get_ec2_instance_type(default="c4.2xlarge", processor="cpu")

# Adding seperate tests to run on cpu instance for cpu image and gpu instance for gpu image.
# But the test behavior doesn't change for cpu or gpu image type.
@pytest.mark.integration("smclarify_cpu")
@pytest.mark.model("N/A")
@pytest.mark.parametrize("ec2_instance_type", SMCLARIFY_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_smclarify_metrics_cpu(training, ec2_connection, region, ec2_instance_type, cpu_only, py3_only):
    run_smclarify_bias_metrics(training, ec2_connection, region, ec2_instance_type)


@pytest.mark.integration("smclarify_gpu")
@pytest.mark.model("N/A")
@pytest.mark.parametrize("ec2_instance_type", SMCLARIFY_EC2_GPU_INSTANCE_TYPE, indirect=True)
@pytest.mark.skip(reason="Skipping test because it is flaky on mainline pipeline.")
def test_smclarify_metrics_gpu(training, ec2_connection, region, ec2_instance_type, gpu_only, py3_only):
    image_cuda_version = get_cuda_version_from_tag(training)
    if image_cuda_version != "cu110":
        pytest.skip("SmClarify is currently installed in cuda 11 gpu images")
    run_smclarify_bias_metrics(training, ec2_connection, region, ec2_instance_type)


class SMClarifyTestFailure(Exception):
    pass


def run_smclarify_bias_metrics(
    image_uri,
    ec2_connection,
    region,
    ec2_instance_type,
    docker_executable="docker",
    container_name="smclarify",
    test_script=SMCLARIFY_SCRIPT,
):
    container_test_local_dir = os.path.join("$HOME", "container_tests")
    ec2_connection.run(f"$(aws ecr get-login --no-include-email --region {region})", hide=True)

    try:
        ec2_connection.run(
            f"{docker_executable} run --name {container_name} -v "
            f"{container_test_local_dir}:{os.path.join(os.sep, 'test')} {image_uri} "
            f"python {test_script}",
            hide=True,
            timeout=300,
        )
    except Exception as e:
        debug_output = ec2_connection.run(f"docker logs {container_name}")
        debug_stdout = debug_output.stdout
        if "Test SMClarify Bias Metrics succeeded!" in debug_stdout:
            LOGGER.warning(
                f"SMClarify test succeeded, but there is an issue with fabric. "
                f"Error:\n{e}\nTest output:\n{debug_stdout}"
            )
            return
        raise SMClarifyTestFailure(
            f"SMClarify test failed on {image_uri} on {ec2_instance_type}. Full output:\n{debug_stdout}"
        ) from e
