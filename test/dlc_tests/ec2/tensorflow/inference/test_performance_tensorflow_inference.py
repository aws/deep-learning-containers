import os
from test.dlc_tests.ec2.tensorflow.inference.test_tensorflow_inference import run_ec2_tensorflow_inference
import pytest

from test.test_utils import CONTAINER_TESTS_PREFIX

TF_PERFORMANCE_INFERENCE_GPU_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "performance_tests", "run_tensorflow_inference_performance_gpu")
TF_PERFORMANCE_INFERENCE_CPU_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "performance_tests", "run_tensorflow_inference_performance_cpu")


@pytest.mark.parametrize("ec2_instance_type", ["p3.16xlarge"], indirect=True)
def test_performance_ec2_tensorflow_inference_gpu(tensorflow_inference, ec2_connection, region, gpu_only):
    ec2_performance_tensorflow_inference(tensorflow_inference, "gpu", ec2_connection, region, TF_PERFORMANCE_INFERENCE_GPU_CMD)


@pytest.mark.parametrize("ec2_instance_type", ["c5.18xlarge"], indirect=True)
def test_performance_ec2_tensorflow_inference_cpu(tensorflow_inference, ec2_connection, region, cpu_only):
    ec2_performance_tensorflow_inference(tensorflow_inference, "cpu", ec2_connection, region, TF_PERFORMANCE_INFERENCE_CPU_CMD)


def ec2_performance_tensorflow_inference(image_uri, processor, ec2_connection, region, test_cmd):
    docker_cmd = "nvidia-docker" if processor == "gpu" else "docker"
    python_version = "py2" if "py2" in image_uri else "py3"
    container_test_local_dir = os.path.join("$HOME", "container_tests")
    repo_name, image_tag = image_uri.split("/")[-1].split(":")

    # Make sure we are logged into ECR so we can pull the image
    ec2_connection.run(f"$(aws ecr get-login --no-include-email --region {region})", hide=True)

    ec2_connection.run(f"{docker_cmd} pull -q {image_uri} ", hide=False)

    # Run performance inference command, display benchmark results to console
    container_name = f"{repo_name}-performance-{image_tag}-ec2"
    ec2_connection.run(
        f"{docker_cmd} run -d --name {container_name} "
        f"-v {container_test_local_dir}:{os.path.join(os.sep, 'test')} {image_uri} ",
        hide=False
    )
    ec2_connection.run(
        f"{docker_cmd} exec {container_name} "
        f"{os.path.join(os.sep, 'bin', 'bash')} -c {test_cmd}",
        hide=False
    )
    ec2_connection.run(
        f"docker rm -f {container_name}",
        hide=False
    )
