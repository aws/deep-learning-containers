import os
import time

import pytest

from invoke.context import Context
from src.benchmark_metrics import (
    MXNET_TRAINING_GPU_IMAGENET_ACCURACY_THRESHOLD,
    MXNET_TRAINING_GPU_IMAGENET_LATENCY_THRESHOLD,
    get_threshold_for_image,
)
from test.test_utils import (
    BENCHMARK_RESULTS_S3_BUCKET, LOGGER, get_framework_and_version_from_tag, get_cuda_version_from_tag,
)


# This test can also be performed for 1 node, but it takes a very long time, and CodeBuild job may expire before the
# test ends.
@pytest.mark.model('resnet50')
@pytest.mark.parametrize("num_nodes", [4], indirect=True)
def test_mxnet_sagemaker_training_performance(mxnet_training, num_nodes, region, gpu_only, py3_only):
    """
    Run MX sagemaker training performance test

    Additional context: Setup for this function is performed by 'setup_sm_benchmark_mx_train_env' -- this installs
    some prerequisite packages, pulls required script, and creates a virtualenv called sm_benchmark_venv.

    The training script mxnet_imagenet_resnet50.py is invoked via a shell script smtrain-resnet50-imagenet.sh
    The shell script sets num-epochs to 40. This parameter is configurable.

    TODO: Refactor the above setup function to be more obviously connected to this function,
    TODO: and install requirements via a requirements.txt file
    TODO: Change latency [time/epoch] metric to Throughput metric

    :param mxnet_training: ECR image URI
    :param num_nodes: Number of nodes to run on
    :param region: AWS region
    """
    _, framework_version = get_framework_and_version_from_tag(mxnet_training)
    device_cuda_str = f"gpu-{get_cuda_version_from_tag(mxnet_training)}"
    py_version = "py37" if "py37" in mxnet_training else "py2" if "py2" in mxnet_training else "py3"
    ec2_instance_type = "p3.16xlarge"

    time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
    commit_info = os.getenv("CODEBUILD_RESOLVED_SOURCE_VERSION", "manual")
    target_upload_location = os.path.join(
        BENCHMARK_RESULTS_S3_BUCKET, "mxnet", framework_version, "sagemaker", "training", device_cuda_str, py_version
    )
    training_job_name = f"mx-tr-bench-{device_cuda_str}-{num_nodes}-node-{py_version}-{commit_info[:7]}-{time_str}"

    test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources")
    venv_dir = os.path.join(test_dir, "sm_benchmark_venv")

    ctx = Context()

    with ctx.cd(test_dir), ctx.prefix(f"source {venv_dir}/bin/activate"):
        log_file = f"results-{commit_info}-{time_str}-{num_nodes}-node.txt"
        run_out = ctx.run(f"timeout 90m python mx_sm_benchmark.py "
                          f"--framework-version {framework_version} "
                          f"--image-uri {mxnet_training} "
                          f"--instance-type ml.{ec2_instance_type} "
                          f"--node-count {num_nodes} "
                          f"--python {py_version} "
                          f"--region {region} "
                          f"--job-name {training_job_name} "
                          f"2>&1 | tee {log_file}",
                          warn=True, echo=True)

        if not run_out.ok:
            target_upload_location = os.path.join(target_upload_location, "failure_log")

    ctx.run(f"aws s3 cp {os.path.join(test_dir, log_file)} {os.path.join(target_upload_location, log_file)}",
            warn=True,
            echo=True)

    LOGGER.info(f"Test results can be found at {os.path.join(target_upload_location, log_file)}")

    assert run_out.ok, (
        f"Benchmark Test failed with return code {run_out.return_code}. "
        f"Test results can be found at {os.path.join(target_upload_location, log_file)}"
    )

    result_statement, time_val, accuracy = _print_results_of_test(os.path.join(test_dir, log_file))

    accuracy_threshold = get_threshold_for_image(framework_version, MXNET_TRAINING_GPU_IMAGENET_ACCURACY_THRESHOLD)
    assert accuracy > accuracy_threshold, (
        f"mxnet {framework_version} sagemaker training {py_version} imagenet {num_nodes} nodes "
        f"Benchmark Result {accuracy} does not reach the threshold accuracy {accuracy_threshold}"
    )

    time_threshold = get_threshold_for_image(framework_version, MXNET_TRAINING_GPU_IMAGENET_LATENCY_THRESHOLD)
    assert time_val < time_threshold, (
        f"mxnet {framework_version} sagemaker training {py_version} imagenet {num_nodes} nodes "
        f"Benchmark Result {time_val} does not reach the threshold latency {time_threshold}"
    )


def _print_results_of_test(file_path):
    last_n_lines = Context().run(f"tail -500 {file_path}").stdout.split("\n")
    result_dict = dict()
    accuracy = 0
    time_cost = 0
    accuracy_key = "Train-accuracy"
    time_cost_key = "Time cost"
    reversed_log = reversed(last_n_lines)
    for line in reversed_log:
        if all(key in result_dict for key in ("Train-accuracy", "Time cost")):
            break
        if accuracy_key in line:
            if accuracy_key in result_dict:
                continue
            accuracy_str = line.split("=")[1]
            result_dict[accuracy_key] = accuracy_str
            accuracy = float(accuracy_str)
        if time_cost_key in line:
            if time_cost_key in result_dict:
                continue
            time_str = line.split("=")[1]
            result_dict[time_cost_key] = time_str
            time_cost = float(time_str)
    result = "\n".join(result_dict.values()) + "\n"
    LOGGER.info(f'Result is {result}')
    LOGGER.info(f'{accuracy_key} is {accuracy}')
    LOGGER.info(f'{time_cost_key} is {time_cost}')
    return result, time_cost, accuracy
