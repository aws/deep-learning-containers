import os
import random
import re
import time

import pytest

from invoke.context import Context

from test.test_utils import BENCHMARK_RESULTS_S3_BUCKET, LOGGER


@pytest.mark.skip(reason="Temp disable it to make pipeline green")
@pytest.mark.parametrize("num_nodes", [1, 4], indirect=True)
def test_tensorflow_sagemaker_training_performance(tensorflow_training, num_nodes, region):

    # This sleep has been inserted because all the parametrized training jobs are automatically created
    # by SageMaker with the same name, due to being started around the same time, and with the same image uri.
    time.sleep(random.Random(x=f"{tensorflow_training}{num_nodes}").random() * 60)

    framework_version = re.search(r"[1,2](\.\d+){2}", tensorflow_training).group()
    processor = "gpu" if "gpu" in tensorflow_training else "cpu"

    ec2_instance_type = "p3.16xlarge" if processor == "gpu" else "c5.18xlarge"

    py_version = "py2" if "py2" in tensorflow_training else "py37" if "py37" in tensorflow_training else "py3"

    time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
    commit_info = os.getenv("CODEBUILD_RESOLVED_SOURCE_VERSION")
    target_upload_location = os.path.join(
        BENCHMARK_RESULTS_S3_BUCKET, "tensorflow", framework_version, "sagemaker", "training", processor, py_version
    )

    test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources")
    venv_dir = os.path.join(test_dir, "sm_benchmark_venv")

    ctx = Context()

    with ctx.cd(test_dir), ctx.prefix(f"source {venv_dir}/bin/activate"):
        log_file = f"results-{commit_info}-{time_str}-{num_nodes}-node.txt"
        run_out = ctx.run(f"timeout 45m python tf_sm_benchmark.py "
                          f"--framework-version {framework_version} "
                          f"--image-uri {tensorflow_training} "
                          f"--instance-type ml.{ec2_instance_type} "
                          f"--node-count {num_nodes} "
                          f"--python {py_version} "
                          f"--region {region} "
                          f"> {log_file}",
                          warn=True, echo=True)

        if not (run_out.ok or run_out.return_code == 124):
            target_upload_location = os.path.join(target_upload_location, "failure_log")

        ctx.run(f"aws s3 cp {log_file} {os.path.join(target_upload_location, log_file)}")

    LOGGER.info(f"Test results can be found at {os.path.join(target_upload_location, log_file)}")

    assert run_out.ok, (f"Benchmark Test failed with return code {run_out.return_code}. "
                        f"Test results can be found at {os.path.join(target_upload_location, log_file)}")
