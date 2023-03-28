# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import os
import time
import pytest
from test.test_utils import (
    CONTAINER_TESTS_PREFIX,
    get_framework_and_version_from_tag,
    LOGGER,
    BENCHMARK_RESULTS_S3_BUCKET
)

SETUP_CMD = "cd $HOME &&
             git clone --branch v2.0.0-rc3 --recursive https://github.com/pytorch/pytorch.git &&
             git clone --recursive https://github.com/pytorch/benchmark.git &&
             git checkout $(cat pytorch/.github/ci_commit_pins/benchmark.txt &&
             cd $HOME/benchmark &&
             python install.py;"

@pytest.mark.parametrize("ec2_instance_type", ["p3.16xlarge"], indirect=True)
@pytest.mark.parametrize("suite", ["huggingface", "timm", "torchbench"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_performance_ec2_pytorch_inference(pytorch_inference, ec2_connection, region, gpu_only):
    ec2_performance_pytorch_inference(
        pytorch_inference, "gpu", ec2_connection, region, 
        f"python benchmarks/dynamo/runner.py --suites={suite} --inference --dtypes=float32 --compilers=inductor --output-dir $HOME/pytorch/benchmark_logs --device {device}",
    )

def ec2_performance_pytorch_inference(image_uri, processor, ec2_connection, region, test_cmd, threshold):
    docker_cmd = "nvidia-docker" if processor == "gpu" else "docker"
    container_test_local_dir = os.path.join("$HOME", "container_tests")
    repo_name, image_tag = image_uri.split("/")[-1].split(":")

    # Make sure we are logged into ECR so we can pull the image
    ec2_connection.run(f"$(aws ecr get-login --no-include-email --region {region})", hide=True)

    ec2_connection.run(f"{docker_cmd} pull -q {image_uri} ")

    time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
    commit_info = os.getenv("CODEBUILD_RESOLVED_SOURCE_VERSION")
    # Run performance inference command, display benchmark results to console
    container_name = f"{repo_name}-performance-{image_tag}-ec2"
    log_file = f"inductor_benchmarks_{time_str}.tar.gz"
    ec2_connection.run(
        f"{docker_cmd} run -d --name {container_name}  -e OMP_NUM_THREADS=1 "
        f"-v {container_test_local_dir}:{os.path.join(os.sep, 'test')} {image_uri} "
    )
    ec2_connection.run(f"{docker_cmd} exec {container_name} " f"/bin/bash {SETUP_CMD}")
    ec2_connection.run(f"{docker_cmd} exec {container_name} " f"/bin/bash {test_cmd} " f"2>&1 | tee {log_file}")
    ec2_connection.run(f"{docker_cmd} exec {container_name} " f"/bin/bash tar -cvzf /test/{log_file} $HOME/pytorch/benchmark_logs")
    ec2_connection.run(f"docker rm -f {container_name}")
    framework_version = re.search(r"\d+(\.\d+){2}", image_uri).group()
    s3_location = os.path.join(
        BENCHMARK_RESULTS_S3_BUCKET, "pytorch", framework_version, "ec2", "inference", processor, "py3", log_name,
    )
    ec2_connection.run(f"aws s3 cp {log_file} {s3_location}")
    LOGGER.info(f"To retrieve complete benchmark log, check {s3_location}")
