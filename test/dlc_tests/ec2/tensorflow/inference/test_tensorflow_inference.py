import pytest


def test_placeholder_gpu(tensorflow_inference, gpu_only):
    print(tensorflow_inference)


@pytest.mark.parametrize("ec2_instance_type", ["c5.4xlarge"], indirect=True)
@pytest.mark.parametrize("ec2_instance_role_name", ["ecsInstanceRole"], indirect=True)
def test_placeholder_cpu(tensorflow_inference, ec2_connection, cpu_only):
    output = ec2_connection.run(f"echo {tensorflow_inference}").stdout.strip("\n")
    assert output == tensorflow_inference, f"Fabric output did not match -- {output}"
