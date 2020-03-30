import pytest

import test.test_utils.ec2 as ec2_utils


def test_placeholder_gpu(tensorflow_inference, gpu_only):
    print(tensorflow_inference)


@pytest.mark.parametrize("ec2_instance_type", ["c5.4xlarge"], indirect=True)
@pytest.mark.parametrize("ec2_instance_role_name", [ec2_utils.EC2_INSTANCE_ROLE_NAME], indirect=True)
def test_placeholder_cpu(tensorflow_inference, ec2_connection, cpu_only):
    conn = ec2_connection

    # Assert that connection is successful
    output = conn.run(f"echo {tensorflow_inference}").stdout.strip("\n")
    assert output == tensorflow_inference, f"Fabric output did not match -- {output}"

    # Assert that copy from s3 was successful
    conn.run(f"[ -f $HOME/container_tests/bin/testPipInstall ]")
