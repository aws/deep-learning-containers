import random
import os
import re
import time

import pytest

from invoke.context import Context

from test.test_utils import BENCHMARK_RESULTS_S3_BUCKET


# This test can also be performed for 1 node, but it takes a very long time, and CodeBuild job may expire before the
# test ends.
@pytest.mark.parametrize("num_nodes", [12], indirect=True)
def test_mxnet_sagemaker_training_performance(mxnet_training, num_nodes, region, gpu_only):

    framework_version = re.search(r"1(\.\d+){2}", mxnet_training).group()
    py_version = "py37" if "py37" in mxnet_training else "py2" if "py2" in mxnet_training else "py3"
    ec2_instance_type = "p3.16xlarge"

    time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
    commit_info = os.getenv("CODEBUILD_RESOLVED_SOURCE_VERSION", "manual")
    target_upload_location = os.path.join(
        BENCHMARK_RESULTS_S3_BUCKET, "mxnet", framework_version, "sagemaker", "training", "gpu", py_version
    )
    training_job_name = f"mx-tr-bench-gpu-{num_nodes}-node-{py_version}-{commit_info[:7]}-{time_str}"

    test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources")
    venv_dir = os.path.join(test_dir, "sm_benchmark_venv")

    ctx = Context()

    with ctx.cd(test_dir), ctx.prefix(f"source {venv_dir}/bin/activate"):
        log_file = f"results-{commit_info}-{time_str}-{num_nodes}-node.txt"
        run_out = ctx.run(f"timeout 150m python mx_sm_benchmark.py "
                          f"--framework-version {framework_version} "
                          f"--image-uri {mxnet_training} "
                          f"--instance-type ml.{ec2_instance_type} "
                          f"--node-count {num_nodes} "
                          f"--python {py_version} "
                          f"--region {region} "
                          f"--job-name {training_job_name} "
                          f"2>&1 > {log_file}",
                          warn=True, echo=True)

        if not run_out.ok:
            target_upload_location = os.path.join(target_upload_location, "failure_log")

        ctx.run(f"aws s3 cp {log_file} {os.path.join(target_upload_location, log_file)}")
        ctx.run(f"cat {log_file}", echo=True)

    assert run_out.ok, f"Benchmark Test failed with return code {run_out.return_code}. "\
                       f"Error logs in {os.path.join(target_upload_location, log_file)}"
