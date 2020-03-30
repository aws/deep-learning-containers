import pytest


def test_placeholder_gpu(tensorflow_inference, gpu_only):
    print(tensorflow_inference)


@pytest.mark.parametrize("ec2_instance_type", ["c5.4xlarge"], indirect=True)
def test_placeholder_cpu(tensorflow_inference, ec2_connection, cpu_only):
    conn = ec2_connection

    # Assert that connection is successful
    output = conn.run(f"echo {tensorflow_inference}").stdout.strip("\n")
    assert output == tensorflow_inference, f"Fabric output did not match -- {output}"

    # Test that copy from s3 was successful by checking if file path below exists
    conn.run(f"[ -f $HOME/container_tests/bin/testPipInstall ]")
