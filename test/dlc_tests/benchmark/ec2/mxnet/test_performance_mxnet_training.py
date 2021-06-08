import os
import pytest

from test.test_utils import CONTAINER_TESTS_PREFIX, get_framework_and_version_from_tag
from test.test_utils.ec2 import (
    execute_ec2_training_performance_test,
    post_process_mxnet_ec2_performance,
)
from src.benchmark_metrics import (
    MXNET_TRAINING_CPU_CIFAR_THRESHOLD,
    MXNET_TRAINING_GPU_IMAGENET_THRESHOLD,
    get_threshold_for_image,
)


MX_PERFORMANCE_TRAINING_GPU_CMD = os.path.join(
    CONTAINER_TESTS_PREFIX, "benchmark", "run_mxnet_training_performance_gpu"
)
MX_PERFORMANCE_TRAINING_CPU_CMD = os.path.join(
    CONTAINER_TESTS_PREFIX, "benchmark", "run_mxnet_training_performance_cpu"
)

MX_EC2_GPU_INSTANCE_TYPE = "p3.16xlarge"
MX_EC2_CPU_INSTANCE_TYPE = "c5.18xlarge"


@pytest.mark.integration("imagenet dataset")
@pytest.mark.model("resnet50_v2")
@pytest.mark.parametrize("ec2_instance_type", [MX_EC2_GPU_INSTANCE_TYPE], indirect=True)
def test_performance_ec2_mxnet_training_gpu(mxnet_training, ec2_connection, gpu_only, py3_only):
    _, framework_version = get_framework_and_version_from_tag(mxnet_training)
    threshold = get_threshold_for_image(framework_version, MXNET_TRAINING_GPU_IMAGENET_THRESHOLD)
    execute_ec2_training_performance_test(
        ec2_connection,
        mxnet_training,
        MX_PERFORMANCE_TRAINING_GPU_CMD,
        post_process=post_process_mxnet_ec2_performance,
        data_source="imagenet",
        threshold={"Throughput": threshold},
    )


@pytest.mark.integration("cifar10 dataset")
@pytest.mark.model("resnet18_v2")
@pytest.mark.parametrize("ec2_instance_type", [MX_EC2_CPU_INSTANCE_TYPE], indirect=True)
def test_performance_ec2_mxnet_training_cpu(mxnet_training, ec2_connection, cpu_only):
    _, framework_version = get_framework_and_version_from_tag(mxnet_training)
    threshold = get_threshold_for_image(framework_version, MXNET_TRAINING_CPU_CIFAR_THRESHOLD)
    execute_ec2_training_performance_test(
        ec2_connection,
        mxnet_training,
        MX_PERFORMANCE_TRAINING_CPU_CMD,
        post_process=post_process_mxnet_ec2_performance,
        data_source="cifar10",
        threshold={"Throughput": threshold},
    )
