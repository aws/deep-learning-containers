import re
import pytest

from invoke.context import Context
from test.test_utils.benchmark import execute_single_node_benchmark, get_framework_version, get_py_version


@pytest.mark.model("resnet50")
@pytest.mark.integration("synthetic dataset")
def test_performance_tensorflow_cpu(tensorflow_training, cpu_only):
    ctx = Context()
    python_version = get_py_version(tensorflow_training)
    framework_version = get_framework_version(tensorflow_training)
    task_name = f"tf_train_single_node_{framework_version}_cpu_{python_version}_resnet50_synthetic"
    script_url = "https://github.com/tensorflow/benchmarks.git"
    execute_single_node_benchmark(ctx, tensorflow_training, "tensorflow", task_name, python_version, script_url)
