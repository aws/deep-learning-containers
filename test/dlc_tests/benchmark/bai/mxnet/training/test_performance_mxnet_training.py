import re
import pytest

from invoke.context import Context
from test.test_utils.benchmark import execute_single_node_benchmark

@pytest.mark.model('resnet18_v2')
@pytest.mark.integration('cifar10 dataset')
def test_performance_mxnet_cpu(mxnet_training, cpu_only):
    ctx = Context()
    python_version = re.search(r"py\s*([\d])", mxnet_training).group()
    task = f"mx_train_single_node_cpu_{python_version}_resnet18v2_cifar10"
    script_url = " https://github.com/awslabs/deeplearning-benchmark.git"
    execute_single_node_benchmark(ctx, mxnet_training, "mxnet", task, python_version, script_url)
