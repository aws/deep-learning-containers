'''
This testplan will currently testing XLA effectiveness for DLC TF containers.
'''
import os
import re
import time

from random import Random

import pytest

from invoke.context import Context

from test.test_utils import (
    BENCHMARK_RESULTS_S3_BUCKET, LOGGER, get_framework_and_version_from_tag, get_cuda_version_from_tag,
)


@pytest.mark.sagemaker_only
@pytest.mark.flaky(reruns=3)
@pytest.mark.integration("imagenet dataset")
@pytest.mark.multinode(4)
@pytest.mark.model("resnet50")
def test_optimized_tensorflow_sagemaker_training_performance_multinode(tensorflow_training, region, gpu_only, tf25_and_above_only):
    throughput_without_xla = run_sm_perf_test(
                                image_uri=tensorflow_training,
                                xla=False,
                                num_nodes=4,
                                region=region,
                                threshold=None
                                )
    '''
    TODO: Add an absolute THRESHOLD for the XLA accelerated benchmark to guard against regression.
    '''
    throughput_with_xla = run_sm_perf_test(
                                image_uri=tensorflow_training,
                                xla=True,
                                num_nodes=4,
                                region=region,
                                threshold=None
                                )
   
    _, framework_version = get_framework_and_version_from_tag(tensorflow_training)
    py_version = "py2" if "py2" in tensorflow_training else "py37" if "py37" in tensorflow_training else "py3"
    '''
    This is a data collection effort for now.
    TODO: Add a THRESHOLD in terms of performance ratio
    assert throughput_with_xla > throughput_without_xla, (
        f"tensorflow {framework_version} sagemaker training gpu {py_version} imagenet 4 nodes "
        f"XLA Benchmark Result {throughput_with_xla} does not reach the threshold {throughput_without_xla}"
    )
    '''


@pytest.mark.sagemaker_only
@pytest.mark.integration("imagenet dataset")
@pytest.mark.model("resnet50")
def test_optimized_tensorflow_sagemaker_training_performance_singlenode(tensorflow_training, region, gpu_only, tf25_and_above_only):
    throughput_without_xla = run_sm_perf_test(
                                image_uri=tensorflow_training,
                                xla=False,
                                num_nodes=1,
                                region=region,
                                threshold=None
                                )
    '''
    TODO: Add an absolute THRESHOLD for the XLA accelerated benchmark to guard against regression.
    '''
    throughput_with_xla = run_sm_perf_test(
                                image_uri=tensorflow_training,
                                xla=True,
                                num_nodes=1,
                                region=region,
                                threshold=None
                                )
   
    _, framework_version = get_framework_and_version_from_tag(tensorflow_training)
    py_version = "py2" if "py2" in tensorflow_training else "py37" if "py37" in tensorflow_training else "py3"
    '''
    This is a data collection effort for now.
    TODO: Add a THRESHOLD in terms of performance ratio
    assert throughput_with_xla > throughput_without_xla, (
        f"tensorflow {framework_version} sagemaker training gpu {py_version} imagenet 1 nodes "
        f"XLA Benchmark Result {throughput_with_xla} does not reach the threshold {throughput_without_xla}"
    )
    '''


def run_sm_perf_test(image_uri, xla, num_nodes, region, threshold=None):
    """
    Run TF sagemaker training performance tests

    Additional context: Setup for this function is performed by 'setup_sm_benchmark_tf_train_env' -- this installs
    some prerequisite packages, clones some repos, and creates a virtualenv called sm_benchmark_venv.

    TODO: Refactor the above setup function to be more obviously connected to this function,
    TODO: and install requirements via a requirements.txt file

    :param image_uri: ECR image URI
    :param xla: [ True | False ] Enable XLA acceleration
    :param num_nodes: Number of nodes to run on
    :param region: AWS region

    This function was inspired by deep-learning-containers/test/dlc_tests/benchmark/sagemaker/tensorflow/training/test_performance_tensorflow_sm_training.py

    """
    _, framework_version = get_framework_and_version_from_tag(image_uri)

    processor = "xla" if xla else "gpu"
    device_cuda_str = f"{processor}-{get_cuda_version_from_tag(image_uri)}"
    '''
    TODO: Switch to p3.16xlarge when EC2 availability issues are resolved
    '''
    ec2_instance_type = "p3.8xlarge"
    py_version = "py2" if "py2" in image_uri else "py37" if "py37" in image_uri else "py3"

    time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
    commit_info = os.getenv("CODEBUILD_RESOLVED_SOURCE_VERSION")
    target_upload_location = os.path.join(
        BENCHMARK_RESULTS_S3_BUCKET, "xla", "tensorflow", framework_version, "sagemaker", "training", device_cuda_str, py_version
    )
    training_job_name = (
        f"opt-tf{framework_version[0]}-bench-{device_cuda_str}-{num_nodes}-node-{py_version}-{commit_info[:7]}-{time_str}"
    )

    # Inserting random sleep because this test starts multiple training jobs around the same time, resulting in
    # a throttling error for SageMaker APIs.
    time.sleep(Random(x=training_job_name).random() * 60)

    test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources")
    venv_dir = os.path.join(test_dir, "sm_benchmark_venv")

    ctx = Context()

    with ctx.cd(test_dir), ctx.prefix(f"source {venv_dir}/bin/activate"):
        log_file = (
            f"results-{commit_info}-{time_str}-optimized-tf{framework_version}-{device_cuda_str}-{py_version}-{num_nodes}-node.txt"
        )
        run_out = ctx.run(
            f"timeout 45m python tf_sm_benchmark.py "
            f"--framework-version {framework_version} "
            f"--image-uri {image_uri} "
            f"--instance-type ml.{ec2_instance_type} "
            f"--node-count {num_nodes} "
            f"--python {py_version} "
            f"--region {region} "
            f"--job-name {training_job_name} "
            f"--xla-{'on' if xla else 'off'} "
            f"2>&1 | tee {log_file}",
            warn=True,
            echo=True,
        )

        if not (run_out.ok or run_out.return_code == 124):
            target_upload_location = os.path.join(target_upload_location, "failure_log")

    ctx.run(f"aws s3 cp {os.path.join(test_dir, log_file)} {os.path.join(target_upload_location, log_file)}")

    LOGGER.info(f"Test results can be found at {os.path.join(target_upload_location, log_file)}")

    result_statement, throughput = _print_results_of_test(os.path.join(test_dir, log_file))
    throughput /= num_nodes

    assert run_out.ok, (
        f"Benchmark Test failed with return code {run_out.return_code}. "
        f"Test results can be found at {os.path.join(target_upload_location, log_file)}"
    )

    LOGGER.info(
        f"optimized-tensorflow-{framework_version} sagemaker training {ec2_instance_type} {device_cuda_str} {py_version} "
        f"imagenet {num_nodes} nodes Throughput: {throughput} images/sec, threshold: {threshold} images/sec"
    )
    if threshold:
        assert throughput > threshold, (
            f"optimized-tensorflow-{framework_version} sagemaker training {ec2_instance_type} {device_cuda_str} {py_version} imagenet {num_nodes} nodes "
            f"Regression Benchmark Result {throughput} does not reach the threshold {threshold}"
        )
    return throughput


def _print_results_of_test(file_path):
    result = ""
    throughput = 0
    """calculate average throughput"""
    result_list, throughput_list = [], []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "images/sec: " in line:
                result_list.append(line.strip("\n"))
                throughput = float(
                    re.search(r"(images/sec:[ ]*)(?P<throughput>[0-9]+\.?[0-9]+)", line).group("throughput")
                )
                throughput_list.append(throughput)
    result = "\n".join(result_list[-100:])+ "\n"
    if len(throughput_list) == 0:
        raise Exception(
            "Cannot find throughput lines. Looks like SageMaker job was not run successfully. Please check"
        )
    # Take average of last 100 throughput lines
    throughput = sum(throughput_list[-100:])/len(throughput_list[-100:])
    LOGGER.info(result)
    return result, throughput
