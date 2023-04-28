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

import datetime
import logging
import os
import re
import sys
from test.test_utils import (BENCHMARK_RESULTS_S3_BUCKET,
                             CONTAINER_TESTS_PREFIX, LOGGER,
                             UL20_CPU_ARM64_US_WEST_2,
                             get_framework_and_version_from_tag, is_pr_context)

import boto3
import pandas as pd
import pytest
from botocore.exceptions import ClientError
from packaging.specifiers import SpecifierSet
from packaging.version import Version

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))
LOGGER.setLevel(logging.INFO)


PT_PERFORMANCE_INFERENCE_GPU_INDUCTOR_HUGGINGFACE_CMD = os.path.join(
    CONTAINER_TESTS_PREFIX,
    "benchmark",
    "run_pytorch_inductor_inference_benchmark_gpu_huggingface",
)
PT_PERFORMANCE_INFERENCE_GPU_INDUCTOR_TIMM_CMD = os.path.join(
    CONTAINER_TESTS_PREFIX,
    "benchmark",
    "run_pytorch_inductor_inference_benchmark_gpu_timm",
)
PT_PERFORMANCE_INFERENCE_GPU_INDUCTOR_TORCHBENCH_CMD = os.path.join(
    CONTAINER_TESTS_PREFIX,
    "benchmark",
    "run_pytorch_inductor_inference_benchmark_gpu_torchbench",
)

PT_PERFORMANCE_INFERENCE_CPU_INDUCTOR_HUGGINGFACE_CMD = os.path.join(
    CONTAINER_TESTS_PREFIX,
    "benchmark",
    "run_pytorch_inductor_inference_benchmark_cpu_huggingface",
)
PT_PERFORMANCE_INFERENCE_CPU_INDUCTOR_TIMM_CMD = os.path.join(
    CONTAINER_TESTS_PREFIX,
    "benchmark",
    "run_pytorch_inductor_inference_benchmark_cpu_timm",
)
PT_PERFORMANCE_INFERENCE_CPU_INDUCTOR_TORCHBENCH_CMD = os.path.join(
    CONTAINER_TESTS_PREFIX,
    "benchmark",
    "run_pytorch_inductor_inference_benchmark_cpu_torchbench",
)


PT_EC2_GPU_INDUCTOR_INSTANCE_TYPES = ["p3.2xlarge", "g5.4xlarge", "g4dn.4xlarge"]
# PT_EC2_GPU_INDUCTOR_INSTANCE_TYPE_P3 = ["p3.2xlarge"]
# PT_EC2_GPU_INDUCTOR_INSTANCE_TYPE_G5 = ["g5.4xlarge"]
# PT_EC2_GPU_INDUCTOR_INSTANCE_TYPE_G4DN = ["g4dn.4xlarge"]
#
PT_EC2_CPU_INDUCTOR_INSTANCE_TYPES = ["c5.4xlarge", "m5.4xlarge"]
# PT_EC2_CPU_INDUCTOR_INSTANCE_TYPE_C5 = ["c5.4xlarge"]
# PT_EC2_CPU_INDUCTOR_INSTANCE_TYPE_M5 = ["m5.4xlarge"]
#
PT_EC2_GRAVITON_INDUCTOR_INSTANCE_TYPES = ["c6g.4xlarge", "c7g.4xlarge", "m7g.4xlarge"]
# PT_EC2_GRAVITON_INDUCTOR_INSTANCE_TYPE_C6G = ["c6g.4xlarge"]
# PT_EC2_GRAVITON_INDUCTOR_INSTANCE_TYPE_C7G = ["c7g.4xlarge"]
# PT_EC2_GRAVITON_INDUCTOR_INSTANCE_TYPE_M7G = ["m7g.4xlarge"]


# SETUP_CMD = "cd /root && \
#             git clone --branch v2.0.0 --recursive --single-branch --depth 1 https://github.com/pytorch/pytorch.git && \
#             git clone --recursive https://github.com/pytorch/benchmark.git && \
#             git checkout $(cat pytorch/.github/ci_commit_pins/benchmark.txt && \
#             cd /root/benchmark && \
#             python install.py;"


def unique_metric_dims(instance_type, precision, model_suite):
    dimensions = [
        {"Name": "InstanceType", "Value": instance_type},
        {"Name": "ModelSuite", "Value": model_suite},
        {"Name": "Precision", "Value": precision},
        {"Name": "WorkLoad", "Value": "Inference"},
    ]
    return dimensions


def get_boto3_session(region="us-west-2"):
    """Get boto3 session with us-east-1 as default region used to connect to AWS services."""
    return boto3.session.Session(region_name=region)


def get_cloudwatch_client(region="us-west-2"):
    """Get AWS CloudWatch client object. Currently assume region is IAD (us-east-1)"""
    return get_boto3_session(region=region).client("cloudwatch")


def put_metric_data(region, metric_name, namespace, unit, value, dimensions):
    """Puts data points to cloudwatch metrics"""
    cloudwatch_client = get_cloudwatch_client(region)
    current_timestamp = datetime.datetime.utcnow()
    try:
        response = cloudwatch_client.put_metric_data(
            Namespace=namespace,
            MetricData=[
                {
                    "MetricName": metric_name,
                    "Dimensions": dimensions,
                    "Value": value,
                    "Unit": unit,
                    "Timestamp": current_timestamp,
                }
            ],
        )
    except ClientError as e:
        LOGGER.error(
            "Error: Cannot put data to cloudwatch metric: {}".format(metric_name)
        )
        LOGGER.error("Exception: {}".format(e))
        raise e


def read_metric(csv_file):
    # csv = os.path.join("./", csv_file)
    df = pd.read_csv(csv_file)
    value = df[df.columns[-1]].iloc[0]
    if isinstance(value, str):
        for i in range(len(value)):
            if value[i].isdecimal() or value[i] == ".":
                continue
            else:
                return float(value[:i])
    return float(value)


def upload_metric(region, instance_type, precision, suite, metric_name, value, unit):
    put_metric_data(
        region=region,
        metric_name=metric_name,
        namespace=f"PyTorch/EC2/Benchmarks/TorchDynamo/Inductor",
        unit=unit,
        value=value,
        dimensions=unique_metric_dims(instance_type, precision, suite),
    )


@pytest.mark.skip(reason="for testing")
@pytest.mark.parametrize(
    "ec2_instance_type", ["c5.4xlarge", "m5.4xlarge"], indirect=True
)
@pytest.mark.parametrize("suite", ["huggingface", "timm_models", "torchbench"])
# @pytest.mark.parametrize("ec2_instance_type", ["c5.4xlarge"], indirect=True)
# @pytest.mark.parametrize("suite", ["torchbench"])
@pytest.mark.parametrize("precision", ["float32"])
def test_performance_ec2_pytorch_inference_cpu(
    ec2_instance_type,
    suite,
    precision,
    pytorch_inference,
    ec2_connection,
    region,
    cpu_only,
):
    _, image_framework_version = get_framework_and_version_from_tag(pytorch_inference)
    if Version(image_framework_version) in SpecifierSet("<2.0"):
        pytest.skip("skip the test as torch.compile only supported after 2.0")
    ec2_performance_pytorch_inference(
        pytorch_inference,
        ec2_instance_type,
        ec2_connection,
        region,
        suite,
        precision,
    )


@pytest.mark.xfail
@pytest.mark.parametrize(
    "ec2_instance_type", ["c6g.4xlarge", "c7g.4xlarge", "m7g.4xlarge"], indirect=True
)
@pytest.mark.parametrize("suite", ["huggingface", "timm_models", "torchbench"])
@pytest.mark.parametrize("precision", ["float32"])
@pytest.mark.parametrize("ec2_instance_ami", [UL20_CPU_ARM64_US_WEST_2], indirect=True)
def test_performance_ec2_pytorch_inference_graviton(
    ec2_instance_type,
    suite,
    precision,
    pytorch_inference_graviton,
    ec2_connection,
    region,
    cpu_only,
):
    _, image_framework_version = get_framework_and_version_from_tag(
        pytorch_inference_graviton
    )
    if Version(image_framework_version) in SpecifierSet("<2.0"):
        pytest.skip("skip the test as torch.compile only supported after 2.0")
    if "graviton" not in pytorch_inference_graviton:
        pytest.skip("skip EC2 tests for inductor")
    ec2_performance_pytorch_inference(
        pytorch_inference_graviton,
        ec2_instance_type,
        ec2_connection,
        region,
        suite,
        precision,
    )


# @pytest.mark.skip(reason="for testing")
# @pytest.mark.parametrize("ec2_instance_type", ["p3.2xlarge", "g5.4xlarge", "g4dn.4xlarge"], indirect=True)
# @pytest.mark.parametrize("suite", ["huggingface", "timm_models", "torchbench"])
@pytest.mark.parametrize("ec2_instance_type", ["g4dn.4xlarge"], indirect=True)
@pytest.mark.parametrize("suite", ["torchbench"])
@pytest.mark.parametrize("precision", ["float32"])
def test_performance_ec2_pytorch_inference_gpu(
    ec2_instance_type,
    suite,
    precision,
    pytorch_inference,
    ec2_connection,
    region,
    gpu_only,
):
    _, image_framework_version = get_framework_and_version_from_tag(pytorch_inference)
    if Version(image_framework_version) in SpecifierSet("<2.0"):
        pytest.skip("skip the test as torch.compile only supported after 2.0")
    ec2_performance_pytorch_inference(
        pytorch_inference,
        ec2_instance_type,
        ec2_connection,
        region,
        suite,
        precision,
    )


def ec2_performance_pytorch_inference(
    image_uri, instance_type, ec2_connection, region, suite, precision
):
    import subprocess as sp
    import time

    is_gpu = re.search(r"(p3|g4|g5)", instance_type)
    is_graviton = re.search(r"(c6g|c7g|m7g)", instance_type)
    device = "cuda" if is_gpu else "cpu"
    docker_cmd = "nvidia-docker" if is_gpu else "docker"
    ec2_local_dir = os.path.join("/home/ubuntu", "results")
    repo_name, image_tag = image_uri.split("/")[-1].split(":")
    container_test_local_dir = os.path.join("$HOME", "container_tests")

    LOGGER.info(
        f"ec2_performance_pytorch_inference params "
        f"image_uri:{image_uri}"
        f"instance_type:{instance_type}"
        f"ec2_connection:{ec2_connection}"
        f"region:{region}"
        f"suite{suite}"
        f"precision:{precision}"
    )

    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    log_name = f"{suite}_results_{os.getenv('CODEBUILD_RESOLVED_SOURCE_VERSION')}_{timestamp}.txt"
    log_location = os.path.join(container_test_local_dir, "benchmark", "logs", log_name)

    # Make sure we are logged into ECR so we can pull the image
    ec2_connection.run(
        f"$(aws ecr get-login --no-include-email --region {region})", hide=True
    )
    ec2_connection.run(f"{docker_cmd} pull {image_uri}", hide="out")

    if is_gpu:
        if suite == "torchbench":
            test_cmd = PT_PERFORMANCE_INFERENCE_GPU_INDUCTOR_TORCHBENCH_CMD
        if suite == "timm_models":
            test_cmd = PT_PERFORMANCE_INFERENCE_GPU_INDUCTOR_TIMM_CMD
        if suite == "huggingface":
            test_cmd = PT_PERFORMANCE_INFERENCE_GPU_INDUCTOR_HUGGINGFACE_CMD
    else:
        if suite == "torchbench":
            test_cmd = PT_PERFORMANCE_INFERENCE_CPU_INDUCTOR_TORCHBENCH_CMD
        if suite == "timm_models":
            test_cmd = PT_PERFORMANCE_INFERENCE_CPU_INDUCTOR_TIMM_CMD
        if suite == "huggingface":
            test_cmd = PT_PERFORMANCE_INFERENCE_CPU_INDUCTOR_HUGGINGFACE_CMD

    test_run_output = ec2_connection.run(
        f"{docker_cmd} run --user root "
        f"-e LOG_FILE={os.path.join(os.sep, 'test', 'benchmark', 'logs', log_name)} "
        f"-e PR_CONTEXT={1 if is_pr_context() else 0} "
        f"-v {container_test_local_dir}:{os.path.join(os.sep, 'test')} {image_uri} "
        f"{os.path.join(os.sep, 'bin', 'bash')} -cex {test_cmd}"
    ).stdout.split("\n")
    LOGGER.info("Output test run ======================= \n{test_run_output}")
    ec2_connection.run(f"docker rm -f {container_name}")
    LOGGER.info(f"To retrieve complete benchmark log, check {s3_location}")
