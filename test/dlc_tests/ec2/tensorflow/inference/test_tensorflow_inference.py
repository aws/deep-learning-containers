import os

import pytest

from test import test_utils
from test.dlc_tests.conftest import LOGGER

TENSORFLOW1_VERSION = "1."
TENSORFLOW2_VERSION = "2."

@pytest.mark.parametrize("ec2_instance_type", ["c5.4xlarge"], indirect=True)
def test_placeholder_cpu(tensorflow_inference, ec2_connection, cpu_only):
    conn = ec2_connection

    # Assert that connection is successful
    output = conn.run(f"echo {tensorflow_inference}").stdout.strip("\n")
    assert output == tensorflow_inference, f"Fabric output did not match -- {output}"

    # Test that copy from s3 was successful by checking if file path below exists
    conn.run(f"[ -f $HOME/container_tests/bin/testPipInstall ]")

def get_tensorflow_framework_version(image_uri):
    return (image_uri.split('-')[-1])[:2]


def host_setup_for_tensorflow_inference(home_dir, framework_version, ec2_connection):
    run_out = ec2_connection.run(f"pip install -U tensorflow=={framework_version} tensorflow-serving-api=={framework_version}")
    LOGGER.info(f"Install pip package for tensorflow inference status : {run_out.return_code == 0}")
    if os.path.exists(f"{home_dir}/serving"):
        ec2_connection.run(f"rm -rf {home_dir}/serving")
    run_out = ec2_connection.run("git clone https://github.com/tensorflow/serving.git")
    ec2_connection.run(f"cd {home_dir}/serving && git checkout r1.13")
    LOGGER.info(f"Clone TF serving repository status {run_out.return_code == 0}")
    return run_out.return_code == 0

def run_ec2_tensorflow_inference():
    return

