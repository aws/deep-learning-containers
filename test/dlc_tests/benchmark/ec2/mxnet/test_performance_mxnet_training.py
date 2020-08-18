import os
import re
import pytest

from test.test_utils import CONTAINER_TESTS_PREFIX
from test.test_utils.ec2 import execute_ec2_training_performance_test
from src.benchmark_metrics import MXNET_TRAINING_CPU_CIFAR_THRESHOLD, MXNET_TRAINING_GPU_IMAGENET_THRESHOLD


MX_PERFORMANCE_TRAINING_GPU_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "benchmark", "run_mxnet_training_performance_gpu")
MX_PERFORMANCE_TRAINING_CPU_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "benchmark", "run_mxnet_training_performance_cpu")

MX_EC2_GPU_INSTANCE_TYPE = "p3.16xlarge"
MX_EC2_CPU_INSTANCE_TYPE = "c5.18xlarge"


@pytest.mark.integration("imagenet dataset")
@pytest.mark.model("resnet50_v2")
@pytest.mark.parametrize("ec2_instance_type", [MX_EC2_GPU_INSTANCE_TYPE], indirect=True)
def test_performance_ec2_mxnet_training_gpu(mxnet_training, ec2_connection, gpu_only, py3_only):
    execute_ec2_training_performance_test(ec2_connection, mxnet_training, MX_PERFORMANCE_TRAINING_GPU_CMD,
                                          post_process=post_process_mxnet_ec2_performance,
                                          data_source="imagenet",
                                          threshold={"Throughput": MXNET_TRAINING_GPU_IMAGENET_THRESHOLD})


@pytest.mark.integration("cifar10 dataset")
@pytest.mark.model("resnet18_v2")
@pytest.mark.parametrize("ec2_instance_type", [MX_EC2_CPU_INSTANCE_TYPE], indirect=True)
def test_performance_ec2_mxnet_training_cpu(mxnet_training, ec2_connection, cpu_only):
    execute_ec2_training_performance_test(ec2_connection, mxnet_training, MX_PERFORMANCE_TRAINING_CPU_CMD,
                                          post_process=post_process_mxnet_ec2_performance,
                                          data_source="cifar10",
                                          threshold={"Throughput": MXNET_TRAINING_CPU_CIFAR_THRESHOLD})


def post_process_mxnet_ec2_performance(connection, log_location):
    log_content = connection.run(f"cat {log_location}").stdout.split("\n")
    total = 0.0
    n = 0
    for line in log_content:
        if "samples/sec" in line:
            throughput = re.search(r'((?P<throughput>[0-9]+\.?[0-9]+)[ ]+samples/sec)', line).group("throughput")
            total += float(throughput)
            n += 1
    if total and n:
        return {"Throughput": total / n}
    else:
        raise ValueError("total: {}; n: {} -- something went wrong".format(total, n))
