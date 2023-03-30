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
import datetime
import re
from test.test_utils import (
    get_framework_and_version_from_tag,
    UL20_CPU_ARM64_US_WEST_2,
    LOGGER,
    BENCHMARK_RESULTS_S3_BUCKET
)
import pytest
import pandas as pd
import boto3
from botocore.exceptions import ClientError
from packaging.version import Version
from packaging.specifiers import SpecifierSet

SETUP_CMD = "cd $HOME && \
             git clone --branch v2.0.0 --recursive --single-branch -depth 1 https://github.com/pytorch/pytorch.git && \
             git clone --recursive https://github.com/pytorch/benchmark.git && \
             git checkout $(cat pytorch/.github/ci_commit_pins/benchmark.txt && \
             cd $HOME/benchmark && \
             python install.py;"


def unique_metric_dims(instance_type, precision, model_suite):
    dimensions = [
        {"Name": "InstanceType", "Value": instance_type},
        {"Name": "ModelSuite", "Value": model_suite},
        {"Name": "Precision", "Value": precision},
        {"Name": "WorkLoad", "Value": "Inference"},
    ]
    return dimensions


def get_boto3_session(region="us-east-1"):
    """Get boto3 session with us-east-1 as default region used to connect to AWS services."""
    return boto3.session.Session(region_name=region)


def get_cloudwatch_client(region="us-east-1"):
    """Get AWS CloudWatch client object. Currently assume region is IAD (us-east-1)"""
    return get_boto3_session(region=region).client("cloudwatch")


def put_metric_data(metric_name, namespace, unit, value, dimensions):
    """Puts data points to cloudwatch metrics"""
    cloudwatch_client = get_cloudwatch_client()
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
            "Error: Cannot put data to cloudwatch metric: {}".format(metric_name))
        LOGGER.error("Exception: {}".format(e))
        raise e


def read_metric(model_suite, csv_file):
    csv = os.path.join("./", f"logs_{model_suite}", csv_file)
    df = pd.read_csv(csv)
    value = df[df.columns[-1]].iloc[0]
    if isinstance(value, str):
        for i in range(len(value)):
            if value[i].isdecimal() or value[i] == ".":
                continue
            else:
                return float(value[:i])
    return float(value)


def upload_metric(instance_type, precision, suite, metric_name, value, unit):
    put_metric_data(
        metric_name=metric_name,
        namespace=f"PyTorch/EC2/Benchmarks/TorchDynamo/Inductor",
        unit=unit,
        value=value,
        dimensions=unique_metric_dims(instance_type, precision, suite),
    )


@pytest.mark.parametrize("ec2_instance_type", ["c5.4xlarge", "m5.4xlarge"], indirect=True)
@pytest.mark.parametrize("suite", ["huggingface", "timm", "torchbench"])
@pytest.mark.parametrize("precision", ["float32"])
def test_performance_ec2_pytorch_inference_cpu(suite, precision, pytorch_inference, ec2_connection, region):
    _, image_framework_version = get_framework_and_version_from_tag(
        pytorch_inference)
    if Version(image_framework_version) in SpecifierSet("<2.0"):
        ec2_performance_pytorch_inference(
            pytorch_inference,
            ec2_instance_type,
            ec2_connection,
            region,
            suite,
            precision,
        )


@pytest.mark.parametrize("ec2_instance_type", ["c6g.4xlarge", "c7g.4xlarge", "m7g.4xlarge"], indirect=True)
@pytest.mark.parametrize("suite", ["huggingface", "timm", "torchbench"])
@pytest.mark.parametrize("precision", ["float32"])
@pytest.mark.parametrize("ec2_instance_ami", [UL20_CPU_ARM64_US_WEST_2], indirect=True)
def test_performance_ec2_pytorch_inference_graviton(suite, precision, pytorch_inference, ec2_connection, region):
    _, image_framework_version = get_framework_and_version_from_tag(
        pytorch_inference)
    if Version(image_framework_version) in SpecifierSet("<2.0"):
        ec2_performance_pytorch_inference(
            pytorch_inference,
            ec2_instance_type,
            ec2_connection,
            region,
            suite,
            precision,
        )

@pytest.mark.parametrize("ec2_instance_type", ["p3.2xlarge", "g5.4xlarge", "g4dn.4xlarge"], indirect=True)
@pytest.mark.parametrize("suite", ["huggingface", "timm", "torchbench"])
@pytest.mark.parametrize("precision", ["float32"])
def test_performance_ec2_pytorch_inference_gpu(suite, precision, pytorch_inference, ec2_connection, region):
    _, image_framework_version = get_framework_and_version_from_tag(
        pytorch_inference)
    if Version(image_framework_version) in SpecifierSet("<2.0"):
        ec2_performance_pytorch_inference(
            pytorch_inference,
            ec2_instance_type,
            ec2_connection,
            region,
            suite,
            precision,
        )


def ec2_performance_pytorch_inference(image_uri, instance_type, ec2_connection, region, suite, precision):
    is_gpu = re.search(r"(p3|g4|g5)", instance_type)
    device = "cuda" if is_gpu else "cpu"
    docker_cmd = "nvidia-docker" if is_gpu == "gpu" else "docker"
    container_test_local_dir = os.path.join("$HOME", "container_tests")
    repo_name, image_tag = image_uri.split("/")[-1].split(":")

    # Make sure we are logged into ECR so we can pull the image
    ec2_connection.run(
        f"$(aws ecr get-login --no-include-email --region {region})", hide=True)

    ec2_connection.run(f"{docker_cmd} pull -q {image_uri} ")

    test_cmd = f"python benchmarks/dynamo/runner.py"
    f" --suites = {suite}"
    f" --inference"
    f" --dtypes = {precision}"
    f" --compilers=inductor"
    f" --output-dir=logs_{suite}"
    f" --extra-args='--output-directory=./'"
    f" --device {device}"
    f" --no-update-archive"
    f" --no-gh-comment"

    # Run performance inference command, display benchmark results to console
    container_name = f"{repo_name}-performance-{image_tag}-ec2"
    log_file = f"inductor_benchmarks_{instance_type}_{suite}.log"
    ec2_connection.run(
        f"{docker_cmd} run -d --name {container_name}  -e OMP_NUM_THREADS=1 "
        f"-v {container_test_local_dir}:{os.path.join(os.sep, 'test')} {image_uri} "
    )
    ec2_connection.run(
        f"{docker_cmd} exec {container_name} " f"/bin/bash {SETUP_CMD}")
    ec2_connection.run(
        f"{docker_cmd} exec {container_name} " f"/bin/bash {test_cmd} " f"2>&1 | tee {log_file}")

    speedup = read_metric(suite, "geomean.csv")
    comp_time = read_metric(suite, "comp_time.csv")
    memory = read_metric(suite, "memory.csv")
    passrate = read_metric(suite, "passrate.csv")
    upload_metric(instance_type, precision, suite, "Speedup", speedup, "None")
    upload_metric(instance_type, precision, suite,
                  "CompilationTime", comp_time, "Seconds")
    upload_metric(instance_type, precision, suite,
                  "PeakMemoryFootprintCompressionRatio", memory, "None")
    upload_metric(instance_type, precision, suite,
                  "PassRate", passrate, "Percent")

    ec2_connection.run(f"docker rm -f {container_name}")
    framework_version = re.search(r"\d+(\.\d+){2}", image_uri).group()
    s3_location = os.path.join(
        BENCHMARK_RESULTS_S3_BUCKET, "pytorch", framework_version, "ec2", "inference", instance_type, "py3", log_file,
    )
    ec2_connection.run(f"aws s3 cp {log_file} {s3_location}")
    LOGGER.info(f"To retrieve complete benchmark log, check {s3_location}")
