import os


from test.test_utils import CONTAINER_TESTS_PREFIX, LOGGER
from test.test_utils import (
    get_account_id_from_image_uri,
    get_region_from_image_uri,
    login_to_ecr_registry,
)

SMCLARIFY_SCRIPT = os.path.join(CONTAINER_TESTS_PREFIX, "test_smclarify_bias_metrics.py")


def smclarify_metrics_cpu(training, ec2_connection):
    run_smclarify_bias_metrics(training, ec2_connection)


def smclarify_metrics_gpu(training, ec2_connection):
    run_smclarify_bias_metrics(
        training, ec2_connection, docker_runtime="--runtime=nvidia --gpus all"
    )


class SMClarifyTestFailure(Exception):
    pass


def run_smclarify_bias_metrics(
    image_uri,
    ec2_connection,
    docker_runtime="",
    container_name="smclarify",
    test_script=SMCLARIFY_SCRIPT,
):
    container_test_local_dir = os.path.join("$HOME", "container_tests")
    account_id = get_account_id_from_image_uri(image_uri)
    region = get_region_from_image_uri(image_uri)

    login_to_ecr_registry(ec2_connection, account_id, region)
    # Do not add -q to docker pull as it leads to a hang for huge images like trcomp
    ec2_connection.run(f"docker pull {image_uri}")

    try:
        ec2_connection.run(
            f"docker run {docker_runtime} --name {container_name} -v "
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
            f"SMClarify test failed on {image_uri}. Full output:\n{debug_stdout}"
        ) from e
