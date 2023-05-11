import os
import time
import pytest

from packaging.version import Version

from src.benchmark_metrics import (
    TENSORFLOW_INFERENCE_GPU_THRESHOLD,
    TENSORFLOW_INFERENCE_CPU_THRESHOLD,
    get_threshold_for_image,
)
from test.test_utils import (
    get_framework_and_version_from_tag,
    is_pr_context,
    is_tf_version,
    UL20_BENCHMARK_CPU_ARM64_US_WEST_2,
)
from test.test_utils.ec2 import (
    ec2_performance_upload_result_to_s3_and_validate,
    post_process_inference,
)


@pytest.mark.model("inception, RCNN-Resnet101-kitti, resnet50_v2, mnist, SSDResnet50Coco")
@pytest.mark.parametrize("ec2_instance_type", ["p3.16xlarge"], indirect=True)
def test_performance_ec2_tensorflow_inference_gpu(
    tensorflow_inference, ec2_connection, ec2_instance_ami, region, gpu_only
):
    _, framework_version = get_framework_and_version_from_tag(tensorflow_inference)
    threshold = get_threshold_for_image(framework_version, TENSORFLOW_INFERENCE_GPU_THRESHOLD)
    ec2_performance_tensorflow_inference(
        tensorflow_inference, "gpu", ec2_connection, ec2_instance_ami, region, threshold
    )


@pytest.mark.model("inception, RCNN-Resnet101-kitti, resnet50_v2, mnist, SSDResnet50Coco")
@pytest.mark.parametrize("ec2_instance_type", ["c5.18xlarge"], indirect=True)
# TODO: Unskip this test for TF 2.4.x Inference CPU images
def test_performance_ec2_tensorflow_inference_cpu(
    tensorflow_inference, ec2_connection, ec2_instance_ami, region, cpu_only
):
    _, framework_version = get_framework_and_version_from_tag(tensorflow_inference)
    if Version(framework_version) == Version("2.4.1"):
        pytest.skip("This test times out, and needs to be run manually.")
    threshold = get_threshold_for_image(framework_version, TENSORFLOW_INFERENCE_CPU_THRESHOLD)
    ec2_performance_tensorflow_inference(
        tensorflow_inference, "cpu", ec2_connection, ec2_instance_ami, region, threshold
    )


@pytest.mark.model("inception, RCNN-Resnet101-kitti, resnet50_v2, mnist, SSDResnet50Coco")
@pytest.mark.parametrize("ec2_instance_type", ["c6g.4xlarge"], indirect=True)
@pytest.mark.parametrize("ec2_instance_ami", [UL20_BENCHMARK_CPU_ARM64_US_WEST_2], indirect=True)
def test_performance_ec2_tensorflow_inference_graviton_cpu(
    tensorflow_inference_graviton, ec2_connection, ec2_instance_ami, region, cpu_only
):
    _, framework_version = get_framework_and_version_from_tag(tensorflow_inference_graviton)
    threshold = get_threshold_for_image(framework_version, TENSORFLOW_INFERENCE_CPU_THRESHOLD)
    ec2_performance_tensorflow_inference(
        tensorflow_inference_graviton, "cpu", ec2_connection, ec2_instance_ami, region, threshold
    )


def ec2_performance_tensorflow_inference(
    image_uri, processor, ec2_connection, ec2_instance_ami, region, threshold
):
    docker_cmd = "nvidia-docker" if processor == "gpu" else "docker"
    is_graviton = "graviton" in image_uri

    # active python env location used with graviton AMI is different than x86_64 AMI
    # using only "python3" on graviton will not yeiled the python env where tensorflow is installed
    python_cmd = "/usr/bin/python3" if "graviton" in image_uri else "python3"

    container_test_local_dir = os.path.join("$HOME", "container_tests")
    tf_version = "1" if is_tf_version("1", image_uri) else "2"
    _, tf_api_version = get_framework_and_version_from_tag(image_uri)

    # setting 1000 iterations on graviton causes the test to timeout thru codebuild, this does not
    # happen when the test is ran manually. Setting graviton to same iteration as PR.
    # TODO: revamp test supporting same configurations on x86_64 and graviton
    num_iterations = 500 if is_pr_context() or is_graviton else 1000

    # Make sure we are logged into ECR so we can pull the image
    ec2_connection.run(f"$(aws ecr get-login --no-include-email --region {region})", hide=True)
    ec2_connection.run(f"{docker_cmd} pull -q {image_uri} ")
    if is_graviton:
        # TF training binary is used that is compatible for graviton instance type
        ec2_connection.run(
            (
                # Protobuf is being pinned upgraded to <3.21 for compatibility with TFS 2.9 and TFS 2.12 the
                # protobuf on the host is not compatible TFS 2.9 and failed benchmarks.
                # Numpy is being updated to <1.24 to support both TFS 2.9 and TFS 2.12 as higher versions
                # will fail with TFS 2.9.
                f"/usr/bin/pip3 install --user --upgrade awscli boto3 grpcio 'protobuf<3.21' 'numpy<1.24'"
            ),
            hide=True,
        )
        ec2_connection.run(
            (
                f"/usr/bin/pip3 install --user --no-dependencies tensorflow-serving-api=={tf_api_version}"
            ),
            hide=True,
        )
    else:
        ec2_connection.run(f"pip3 install -U pip")
        ec2_connection.run(
            f"pip3 install --user boto3 grpcio 'tensorflow-serving-api<={tf_api_version}'"
        )
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
    commit_info = os.getenv("CODEBUILD_RESOLVED_SOURCE_VERSION")
    log_file = f"synthetic_{commit_info}_{time_str}.log"

    # Run the test
    assert ec2_connection.run(
        f"{python_cmd} {container_test_local_dir}/bin/benchmark/tf{tf_version}_serving_perf.py "
        f"--processor {processor} --docker_image_name {image_uri} "
        f"--run_all_s3 --binary /usr/bin/tensorflow_model_server --get_perf --iterations {num_iterations} "
        f"2>&1 | tee {log_file}"
    )

    # Check is benchmark is within limits
    ec2_performance_upload_result_to_s3_and_validate(
        ec2_connection,
        image_uri,
        log_file,
        "synthetic",
        threshold,
        post_process_inference,
        log_file,
    )
