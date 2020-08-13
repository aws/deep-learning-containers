import os

import pytest

from test.test_utils import CONTAINER_TESTS_PREFIX
from test.test_utils.ec2 import execute_ec2_training_performance_test
from src.benchmark_metrics import MXNET_TRAINING_CPU_CIFAR_THRESHOLD


MX_PERFORMANCE_TRAINING_GPU_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "benchmark", "run_mxnet_training_performance_gpu")
MX_PERFORMANCE_TRAINING_CPU_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "benchmark", "run_mxnet_training_performance_cpu")

MX_EC2_GPU_INSTANCE_TYPE = "p3.16xlarge"
MX_EC2_CPU_INSTANCE_TYPE = "c5.18xlarge"

@pytest.mark.skip()
@pytest.mark.integration("imagenet dataset")
@pytest.mark.model("resnet50_v2")
@pytest.mark.parametrize("ec2_instance_type", [MX_EC2_GPU_INSTANCE_TYPE], indirect=True)
def test_performance_ec2_mxnet_training_gpu(mxnet_training, ec2_connection, gpu_only):
    execute_ec2_training_performance_test(ec2_connection, mxnet_training, MX_PERFORMANCE_TRAINING_GPU_CMD)


@pytest.mark.integration("cifar10 dataset")
@pytest.mark.model("resnet18_v2")
@pytest.mark.parametrize("ec2_instance_type", [MX_EC2_CPU_INSTANCE_TYPE], indirect=True)
def test_performance_ec2_mxnet_training_cpu(mxnet_training, ec2_connection, cpu_only):
    execute_ec2_training_performance_test(ec2_connection, mxnet_training, MX_PERFORMANCE_TRAINING_CPU_CMD,
                                          post_process=post_process_mxnet_ec2_performance,
                                          data_source="cifar10", threshold=MXNET_TRAINING_CPU_CIFAR_THRESHOLD)


def post_process_mxnet_ec2_performance(connection, log_location):
    index = 4
    if "cpu" in log_location and "inference" in log_location:
        index = 1
    log_content = connection.run(f"cat {log_location}").stdout.split("\n")
    total = 0.0
    n = 0
    for line in log_content:
        if "Speed" in line:
            try:
                total += float(line.split()[index])
            except ValueError as e:
                raise RuntimeError("LINE: {} split {} ERROR: {}".format(line, line.split()[index], e))
            n += 1
    if total and n:
        return total / n
    else:
        raise ValueError("total: {}; n: {} -- something went wrong".format(total, n))
