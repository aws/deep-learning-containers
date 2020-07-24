import os
import re
import time

from random import Random

import pytest

from invoke.context import Context

from test.test_utils import BENCHMARK_RESULTS_S3_BUCKET, LOGGER


@pytest.mark.integration("imagenet dataset")
@pytest.mark.multinode("multinode")
@pytest.mark.model("resnet50")
@pytest.mark.parametrize("num_nodes", [1, 4], indirect=True)
def test_tensorflow_sagemaker_training_performance(tensorflow_training, num_nodes, region):

    framework_version = re.search(r"[1,2](\.\d+){2}", tensorflow_training).group()
    if framework_version.startswith("1."):
        pytest.skip("Skipping benchmark test on TF 1.x images.")

    processor = "gpu" if "gpu" in tensorflow_training else "cpu"

    ec2_instance_type = "p3.16xlarge" if processor == "gpu" else "c5.18xlarge"

    py_version = "py2" if "py2" in tensorflow_training else "py37" if "py37" in tensorflow_training else "py3"

    time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
    commit_info = os.getenv("CODEBUILD_RESOLVED_SOURCE_VERSION")
    target_upload_location = os.path.join(
        BENCHMARK_RESULTS_S3_BUCKET, "tensorflow", framework_version, "sagemaker", "training", processor, py_version
    )
    training_job_name = (f"tf{framework_version[0]}-tr-bench-{processor}-{num_nodes}-node-{py_version}"
                         f"-{commit_info[:7]}-{time_str}")

    # Inserting random sleep because this test starts multiple training jobs around the same time, resulting in
    # a throttling error for SageMaker APIs.
    time.sleep(Random(x=training_job_name).random() * 60)

    test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources")
    venv_dir = os.path.join(test_dir, "sm_benchmark_venv")

    ctx = Context()

    with ctx.cd(test_dir), ctx.prefix(f"source {venv_dir}/bin/activate"):
        log_file = f"results-{commit_info}-{time_str}-{framework_version}-{processor}-{py_version}-{num_nodes}-node.txt"
        run_out = ctx.run(f"timeout 45m python tf_sm_benchmark.py "
                          f"--framework-version {framework_version} "
                          f"--image-uri {tensorflow_training} "
                          f"--instance-type ml.{ec2_instance_type} "
                          f"--node-count {num_nodes} "
                          f"--python {py_version} "
                          f"--region {region} "
                          f"--job-name {training_job_name}"
                          f"2>&1 > {log_file}",
                          warn=True, echo=True)

        if not (run_out.ok or run_out.return_code == 124):
            target_upload_location = os.path.join(target_upload_location, "failure_log")

    ctx.run(f"aws s3 cp {os.path.join(test_dir, log_file)} {os.path.join(target_upload_location, log_file)}")

    LOGGER.info(f"Test results can be found at {os.path.join(target_upload_location, log_file)}")

    _print_results_of_test(os.path.join(test_dir, log_file), processor)

    assert run_out.ok, (f"Benchmark Test failed with return code {run_out.return_code}. "
                        f"Test results can be found at {os.path.join(target_upload_location, log_file)}")


def _print_results_of_test(file_path, processor):
    last_100_lines = Context().run(f"tail -100 {file_path}").stdout.split("\n")
    result = ""
    if processor == "cpu":
        for line in last_100_lines:
            if "Total img/sec on " in line:
                result = line + "\n"
    elif processor == "gpu":
        result_dict = dict()
        for line in last_100_lines:
            if "images/sec: " in line:
                key = line.split("<stdout>")[0]
                result_dict[key] = line.strip("\n")
        result = "\n".join(result_dict.values()) + "\n"
    LOGGER.info(result)
    return result
