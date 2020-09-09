import os
import re
import pytest

from test.test_utils import CONTAINER_TESTS_PREFIX, is_tf2
from test.test_utils.ec2 import execute_ec2_training_performance_test
from src.benchmark_metrics import (
    TENSORFLOW2_TRAINING_CPU_SYNTHETIC_THRESHOLD,
    TENSORFLOW2_TRAINING_GPU_SYNTHETIC_THRESHOLD,
    TENSORFLOW2_TRAINING_GPU_IMAGENET_THRESHOLD,
    TENSORFLOW1_TRAINING_CPU_SYNTHETIC_THRESHOLD,
    TENSORFLOW1_TRAINING_GPU_SYNTHETIC_THRESHOLD,
    TENSORFLOW1_TRAINING_GPU_IMAGENET_THRESHOLD,
)

TF_PERFORMANCE_TRAINING_CPU_SYNTHETIC_CMD = os.path.join(
    CONTAINER_TESTS_PREFIX, "benchmark", "run_tensorflow_training_performance_cpu"
)
TF_PERFORMANCE_TRAINING_GPU_SYNTHETIC_CMD = os.path.join(
    CONTAINER_TESTS_PREFIX, "benchmark", "run_tensorflow_training_performance_gpu_synthetic",
)
TF_PERFORMANCE_TRAINING_GPU_IMAGENET_CMD = os.path.join(
    CONTAINER_TESTS_PREFIX, "benchmark", "run_tensorflow_training_performance_gpu_imagenet",
)

TF_EC2_GPU_INSTANCE_TYPE = "p3.16xlarge"
TF_EC2_CPU_INSTANCE_TYPE = "c5.18xlarge"


@pytest.mark.integration("synthetic dataset")
@pytest.mark.model("resnet50")
@pytest.mark.parametrize("ec2_instance_type", [TF_EC2_CPU_INSTANCE_TYPE], indirect=True)
def test_performance_tensorflow_cpu(tensorflow_training, ec2_connection, cpu_only):
    threshold = (
        TENSORFLOW2_TRAINING_CPU_SYNTHETIC_THRESHOLD
        if is_tf2(tensorflow_training)
        else TENSORFLOW1_TRAINING_CPU_SYNTHETIC_THRESHOLD
    )
    execute_ec2_training_performance_test(
        ec2_connection,
        tensorflow_training,
        TF_PERFORMANCE_TRAINING_CPU_SYNTHETIC_CMD,
        post_process=post_process_tensorflow_training_performance,
        data_source="synthetic",
        threshold={"Throughput": threshold},
    )


@pytest.mark.integration("synthetic dataset")
@pytest.mark.model("resnet50")
@pytest.mark.parametrize("ec2_instance_type", [TF_EC2_GPU_INSTANCE_TYPE], indirect=True)
def test_performance_tensorflow_gpu_synthetic(tensorflow_training, ec2_connection, gpu_only):
    threshold = (
        TENSORFLOW2_TRAINING_GPU_SYNTHETIC_THRESHOLD
        if is_tf2(tensorflow_training)
        else TENSORFLOW1_TRAINING_GPU_SYNTHETIC_THRESHOLD
    )
    execute_ec2_training_performance_test(
        ec2_connection,
        tensorflow_training,
        TF_PERFORMANCE_TRAINING_GPU_SYNTHETIC_CMD,
        post_process=post_process_tensorflow_training_performance,
        data_source="synthetic",
        threshold={"Throughput": threshold},
    )


@pytest.mark.integration("imagenet dataset")
@pytest.mark.model("resnet50")
@pytest.mark.parametrize("ec2_instance_type", [TF_EC2_GPU_INSTANCE_TYPE], indirect=True)
def test_performance_tensorflow_gpu_imagenet(tensorflow_training, ec2_connection, gpu_only):
    threshold = (
        TENSORFLOW2_TRAINING_GPU_IMAGENET_THRESHOLD
        if is_tf2(tensorflow_training)
        else TENSORFLOW1_TRAINING_GPU_IMAGENET_THRESHOLD
    )
    execute_ec2_training_performance_test(
        ec2_connection,
        tensorflow_training,
        TF_PERFORMANCE_TRAINING_GPU_IMAGENET_CMD,
        post_process=post_process_tensorflow_training_performance,
        data_source="imagenet",
        threshold={"Throughput": threshold},
    )


def post_process_tensorflow_training_performance(connection, log_location):
    last_lines = connection.run(f"tail {log_location}").stdout.split("\n")
    throughput = 0
    for line in reversed(last_lines):
        if "images/sec:" in line:
            throughput = float(re.search(r"(images/sec:[ ]*)(?P<throughput>[0-9]+\.?[0-9]+)", line).group("throughput"))
            break
    return {"Throughput": throughput}
