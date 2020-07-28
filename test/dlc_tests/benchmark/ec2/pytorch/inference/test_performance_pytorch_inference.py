import os
import time
import pytest

from test.test_utils import CONTAINER_TESTS_PREFIX

PT_PERFORMANCE_INFERENCE_SCRIPT = os.path.join(CONTAINER_TESTS_PREFIX, "benchmark", "run_pytorch_inference_performance.py")
PT_PERFORMANCE_INFERENCE_CPU_CMD = f"{PT_PERFORMANCE_INFERENCE_SCRIPT}  --iterations 500"
PT_PERFORMANCE_INFERENCE_GPU_CMD = f"{PT_PERFORMANCE_INFERENCE_SCRIPT}  --iterations 1000 --gpu"

@pytest.mark.skip()
@pytest.mark.model("resnet18, VGG13, MobileNetV2, GoogleNet, DenseNet121, InceptionV3")
@pytest.mark.parametrize("ec2_instance_type", ["p3.16xlarge"], indirect=True)
def test_performance_ec2_pytorch_inference_gpu(pytorch_inference, ec2_connection, region, gpu_only):
    ec2_performance_pytorch_inference(pytorch_inference, "gpu", ec2_connection, region, PT_PERFORMANCE_INFERENCE_GPU_CMD)

@pytest.mark.skip()
@pytest.mark.model("resnet18, VGG13, MobileNetV2, GoogleNet, DenseNet121, InceptionV3")
@pytest.mark.parametrize("ec2_instance_type", ["c5.18xlarge"], indirect=True)
def test_performance_ec2_pytorch_inference_cpu(pytorch_inference, ec2_connection, region, cpu_only):
    ec2_performance_pytorch_inference(pytorch_inference, "cpu", ec2_connection, region, PT_PERFORMANCE_INFERENCE_CPU_CMD)


def ec2_performance_pytorch_inference(image_uri, processor, ec2_connection, region, test_cmd):
    docker_cmd = "nvidia-docker" if processor == "gpu" else "docker"
    python_version = "py2" if "py2" in image_uri else "py3"
    container_test_local_dir = os.path.join("$HOME", "container_tests")
    repo_name, image_tag = image_uri.split("/")[-1].split(":")

    # Make sure we are logged into ECR so we can pull the image
    ec2_connection.run(f"$(aws ecr get-login --no-include-email --region {region})", hide=True)

    ec2_connection.run(f"{docker_cmd} pull -q {image_uri} ")

    time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
    commit_info = os.getenv("CODEBUILD_RESOLVED_SOURCE_VERSION")
    # Run performance inference command, display benchmark results to console
    container_name = f"{repo_name}-performance-{image_tag}-ec2"
    log_file = f"inference_benchmark_results_{commit_info}_{time_str}.log"
    ec2_connection.run(
        f"{docker_cmd} run -d --name {container_name}  -e OMP_NUM_THREADS=1 "
        f"-v {container_test_local_dir}:{os.path.join(os.sep, 'test')} {image_uri} "
    )
    ec2_connection.run(
        f"{docker_cmd} exec {container_name} "
        f"python {test_cmd} "
        f"2>&1 | tee {log_file}"
    )
    ec2_connection.run(
        f"docker rm -f {container_name}"
    )
    ec2_connection.run(
        f"echo Benchmark Results: >&2;"
        f"echo PyTorch Inference {processor} {python_version} >&2"
    )
    if python_version == "py3":
        ec2_connection.run(f"tail -28 {log_file} >&2")
    else:
        ec2_connection.run(f"cat {log_file} >&2")
    ec2_connection.run(
        f"aws s3 cp {log_file} s3://dlinfra-dlc-cicd-performance/pytorch/ec2/inference/{processor}/{python_version}/{log_file}"
    )
    ec2_connection.run(
        f"echo To retrieve complete benchmark log, check s3://dlinfra-dlc-cicd-performance/pytorch/ec2/inference/{processor}/{python_version}/{log_file} >&2"
    )
