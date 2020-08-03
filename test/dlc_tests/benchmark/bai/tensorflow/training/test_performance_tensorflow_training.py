import re
import pytest

from invoke.context import Context
from test.test_utils.benchmark import  execute_single_node_benchmark

@pytest.mark.model('resnet50')
@pytest.mark.integration('synthetic dataset')
def test_performance_tensorflow_cpu(tensorflow_training, cpu_only):
    ctx = Context()
    python_version = re.search(r"py\s*([\d])", tensorflow_training).group()
    version = re.search(r":\s*([\d][.][\d]+)", tensorflow_training).group(1)
    task = f"tf_train_single_node_{version}_cpu_{python_version}_resnet50_synthetic"
    script_url = "https://github.com/tensorflow/benchmarks.git"
    execute_single_node_benchmark(ctx, tensorflow_training, "tensorflow", task, python_version,script_url)
