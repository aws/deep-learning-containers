import pytest

from test.test_utils import UBUNTU_16_BASE_DLAMI


def test_placeholder_gpu(tensorflow_inference, gpu_only):
    print(tensorflow_inference)


@pytest.mark.parametrize("ec2_instance_type", ["c5.4xlarge"], indirect=True)
@pytest.mark.parametrize("ec2_instance_ami", [UBUNTU_16_BASE_DLAMI], indirect=True)
def test_placeholder_cpu(tensorflow_inference, ec2_connection, cpu_only):
    print(tensorflow_inference)
    output = ec2_connection.run("ls -l")
    print(output.stdout)
