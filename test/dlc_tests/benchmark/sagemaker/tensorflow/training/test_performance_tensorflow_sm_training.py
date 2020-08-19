import os
import re
import time

from random import Random

import pytest

from invoke.context import Context
from src.benchmark_metrics import (
    TENSORFLOW2_SM_TRAINING_CPU_1NODE_THRESHOLD,
    TENSORFLOW2_SM_TRAINING_CPU_4NODE_THRESHOLD,
    TENSORFLOW2_SM_TRAINING_GPU_1NODE_THRESHOLD,
    TENSORFLOW2_SM_TRAINING_GPU_4NODE_THRESHOLD,
)
from test.test_utils import BENCHMARK_RESULTS_S3_BUCKET, LOGGER


@pytest.mark.integration("imagenet dataset")
@pytest.mark.multinode(4)
@pytest.mark.model("resnet50")
def test_tensorflow_sagemaker_training_performance_multinode(tensorflow_training, region):
    run_sm_perf_test(tensorflow_training, 4, region)


@pytest.mark.integration("imagenet dataset")
@pytest.mark.model("resnet50")
def test_tensorflow_sagemaker_training_performance_singlenode(tensorflow_training, region):
    run_sm_perf_test(tensorflow_training, 1, region)


def run_sm_perf_test(image_uri, num_nodes, region):
    """
    Run TF sagemaker training performance tests

    Additonal context: Setup for this function is performed by 'setup_sm_benchmark_tf_train_env' -- this installs
    some prerequisite packages, clones some repos, and creates a virtualenv called sm_benchmark_venv.

    TODO: Refactor the above setup function to be more obviously connected to this function,
    TODO: and install requirements via a requirements.txt file

    :param image_uri: ECR image URI
    :param num_nodes: Number of nodes to run on
    :param region: AWS region
    """
    framework_version = re.search(r"[1,2](\.\d+){2}", image_uri).group()
    if framework_version.startswith("1."):
        pytest.skip("Skipping benchmark test on TF 1.x images.")

    processor = "gpu" if "gpu" in image_uri else "cpu"

    ec2_instance_type = "p3.16xlarge" if processor == "gpu" else "c5.18xlarge"

    py_version = "py2" if "py2" in image_uri else "py37" if "py37" in image_uri else "py3"

    time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
    commit_info = os.getenv("CODEBUILD_RESOLVED_SOURCE_VERSION")
    target_upload_location = os.path.join(
        BENCHMARK_RESULTS_S3_BUCKET, "tensorflow", framework_version, "sagemaker", "training", processor, py_version
    )
    training_job_name = (
        f"tf{framework_version[0]}-tr-bench-{processor}-{num_nodes}-node-{py_version}" f"-{commit_info[:7]}-{time_str}"
    )

    # Inserting random sleep because this test starts multiple training jobs around the same time, resulting in
    # a throttling error for SageMaker APIs.
    time.sleep(Random(x=training_job_name).random() * 60)

    test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources")
    venv_dir = os.path.join(test_dir, "sm_benchmark_venv")

    ctx = Context()

    with ctx.cd(test_dir), ctx.prefix(f"source {venv_dir}/bin/activate"):
        log_file = f"results-{commit_info}-{time_str}-{framework_version}-{processor}-{py_version}-{num_nodes}-node.txt"
        run_out = ctx.run(
            f"timeout 45m python tf_sm_benchmark.py "
            f"--framework-version {framework_version} "
            f"--image-uri {image_uri} "
            f"--instance-type ml.{ec2_instance_type} "
            f"--node-count {num_nodes} "
            f"--python {py_version} "
            f"--region {region} "
            f"--job-name {training_job_name}"
            f"2>&1 | tee {log_file}",
            warn=True,
            echo=True,
        )

        if not (run_out.ok or run_out.return_code == 124):
            target_upload_location = os.path.join(target_upload_location, "failure_log")

    ctx.run(f"aws s3 cp {os.path.join(test_dir, log_file)} {os.path.join(target_upload_location, log_file)}")

    LOGGER.info(f"Test results can be found at {os.path.join(target_upload_location, log_file)}")

    result_statement, throughput = _print_results_of_test(os.path.join(test_dir, log_file), processor)
    throughput /= num_nodes

    assert run_out.ok, (
        f"Benchmark Test failed with return code {run_out.return_code}. "
        f"Test results can be found at {os.path.join(target_upload_location, log_file)}"
    )

    threshold = (
        (TENSORFLOW2_SM_TRAINING_CPU_1NODE_THRESHOLD if num_nodes == 1 else TENSORFLOW2_SM_TRAINING_CPU_4NODE_THRESHOLD)
        if processor == "cpu"
        else TENSORFLOW2_SM_TRAINING_GPU_1NODE_THRESHOLD
        if num_nodes == 1
        else TENSORFLOW2_SM_TRAINING_GPU_4NODE_THRESHOLD
    )
    LOGGER.info(
        f"tensorflow {framework_version} sagemaker training {processor} {py_version} "
        f"imagenet {num_nodes} nodes Throughput: {throughput} images/sec, threshold: {threshold} images/sec"
    )
    assert throughput > threshold, (
        f"tensorflow {framework_version} sagemaker training {processor} {py_version} imagenet {num_nodes} nodes "
        f"Benchmark Result {throughput} does not reach the threshold {threshold}"
    )


def _print_results_of_test(file_path, processor):
    last_100_lines = Context().run(f"tail -100 {file_path}").stdout.split("\n")
    result = ""
    throughput = 0
    if processor == "cpu":
        for line in last_100_lines:
            if "Total img/sec on " in line:
                result = line + "\n"
                throughput = float(
                    re.search(r"(CPU\(s\):[ ]*)(?P<throughput>[0-9]+\.?[0-9]+)", line).group("throughput")
                )
                break
    elif processor == "gpu":
        result_dict = dict()
        for line in last_100_lines:
            if "images/sec: " in line:
                key = line.split("<stdout>")[0]
                result_dict[key] = line.strip("\n")
                if throughput == 0:
                    throughput = float(
                        re.search(r"(images/sec:[ ]*)(?P<throughput>[0-9]+\.?[0-9]+)", line).group("throughput")
                    )
        result = "\n".join(result_dict.values()) + "\n"
    LOGGER.info(result)
    return result, throughput
