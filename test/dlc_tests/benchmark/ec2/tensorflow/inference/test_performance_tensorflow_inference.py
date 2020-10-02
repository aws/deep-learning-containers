import os
import time
import re
import pytest
from src.benchmark_metrics import (
    TENSORFLOW2_INFERENCE_GPU_THRESHOLD,
    TENSORFLOW2_INFERENCE_CPU_THRESHOLD,
    TENSORFLOW1_INFERENCE_GPU_THRESHOLD,
    TENSORFLOW1_INFERENCE_CPU_THRESHOLD,
)
from test.test_utils import BENCHMARK_RESULTS_S3_BUCKET, is_tf1
from test.test_utils.ec2 import (
    ec2_performance_upload_result_to_s3_and_validate,
    post_process_inference,
)


@pytest.mark.model("inception, RCNN-Resnet101-kitti, resnet50_v2, mnist, SSDResnet50Coco")
@pytest.mark.parametrize("ec2_instance_type", ["p3.16xlarge"], indirect=True)
def test_performance_ec2_tensorflow_inference_gpu(tensorflow_inference, ec2_connection, region, gpu_only):
    threshold = (
        TENSORFLOW1_INFERENCE_GPU_THRESHOLD if is_tf1(tensorflow_inference) else TENSORFLOW2_INFERENCE_GPU_THRESHOLD
    )
    ec2_performance_tensorflow_inference(tensorflow_inference, "gpu", ec2_connection, region, threshold)


@pytest.mark.model("inception, RCNN-Resnet101-kitti, resnet50_v2, mnist, SSDResnet50Coco")
@pytest.mark.parametrize("ec2_instance_type", ["c5.18xlarge"], indirect=True)
def test_performance_ec2_tensorflow_inference_cpu(tensorflow_inference, ec2_connection, region, cpu_only):
    threshold = (
        TENSORFLOW1_INFERENCE_CPU_THRESHOLD if is_tf1(tensorflow_inference) else TENSORFLOW2_INFERENCE_CPU_THRESHOLD
    )
    ec2_performance_tensorflow_inference(tensorflow_inference, "cpu", ec2_connection, region, threshold)


def ec2_performance_tensorflow_inference(image_uri, processor, ec2_connection, region, threshold):
    docker_cmd = "nvidia-docker" if processor == "gpu" else "docker"
    container_test_local_dir = os.path.join("$HOME", "container_tests")
    tf_version = "1" if is_tf1(image_uri) else "2"
    tf_api_version = "1.15" if tf_version == "1" else "2.3.0"

    # Make sure we are logged into ECR so we can pull the image
    ec2_connection.run(f"$(aws ecr get-login --no-include-email --region {region})", hide=True)

    ec2_connection.run(f"{docker_cmd} pull -q {image_uri} ")

    # Run performance inference command, display benchmark results to console
    ec2_connection.run(f"pip3 install -U pip")
    ec2_connection.run(
        f"pip3 install boto3 grpcio tensorflow-serving-api=={tf_api_version} --user --no-warn-script-location"
    )
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
    commit_info = os.getenv("CODEBUILD_RESOLVED_SOURCE_VERSION")
    log_file = f"synthetic_{commit_info}_{time_str}.log"
    ec2_connection.run(
        f"python3 {container_test_local_dir}/bin/benchmark/tf{tf_version}_serving_perf.py "
        f"--processor {processor} --docker_image_name {image_uri} --run_all_s3 --binary /usr/bin/tensorflow_model_server --get_perf --iterations 1000 "
        f"2>&1 | tee {log_file}"
    )
    ec2_performance_upload_result_to_s3_and_validate(
        ec2_connection, image_uri, log_file, "synthetic", threshold, post_process_inference, log_file,
    )
