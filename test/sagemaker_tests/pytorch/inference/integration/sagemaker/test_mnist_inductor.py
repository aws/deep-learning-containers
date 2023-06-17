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
from __future__ import absolute_import

import os
import sys
from packaging.version import Version
from packaging.specifiers import SpecifierSet

import numpy as np
import pytest
import sagemaker
from sagemaker.pytorch import PyTorchModel

import boto3
from datetime import datetime, timedelta
import time
import json
import logging

from ...integration import model_cpu_dir
from ...integration.sagemaker.timeout import timeout_and_delete_endpoint
from .... import invoke_pytorch_helper_function


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))


SM_SINGLE_GPU_INSTANCE_TYPES = ["ml.p3.2xlarge", "ml.g4dn.4xlarge", "ml.g5.4xlarge"]
SM_GRAVITON_C7G = ["ml.c7g.4xlarge"]


@pytest.mark.model("mnist")
@pytest.mark.processor("cpu")
@pytest.mark.cpu_test
def test_mnist_distributed_cpu_inductor(
    framework_version, ecr_image, instance_type, sagemaker_regions
):
    instance_type = instance_type or "ml.c5.9xlarge"
    if Version(framework_version) in SpecifierSet("<2.0"):
        pytest.skip("skip the test as torch.compile only supported after 2.0")
    model_dir = os.path.join(model_cpu_dir, "model_mnist_inductor.tar.gz")
    function_args = {
        "framework_version": framework_version,
        "instance_type": instance_type,
        "model_dir": model_dir,
    }
    invoke_pytorch_helper_function(
        ecr_image, sagemaker_regions, _test_mnist_distributed, function_args
    )


@pytest.mark.model("mnist")
@pytest.mark.processor("cpu")
@pytest.mark.cpu_test
@pytest.mark.parametrize("instance_type", SM_GRAVITON_C7G)
def test_mnist_distributed_graviton_inductor(
    framework_version, ecr_image, instance_type, sagemaker_regions
):
    if Version(framework_version) in SpecifierSet("<2.0"):
        pytest.skip("skip the test as torch.compile only supported after 2.0")
    if "graviton" not in ecr_image:
        pytest.skip("skip SM tests for inductor on c7g")
    model_dir = os.path.join(model_cpu_dir, "model_mnist_inductor.tar.gz")
    function_args = {
        "framework_version": framework_version,
        "instance_type": instance_type,
        "model_dir": model_dir,
    }
    invoke_pytorch_helper_function(
        ecr_image, sagemaker_regions, _test_mnist_distributed, function_args
    )


@pytest.mark.model("mnist")
@pytest.mark.processor("gpu")
@pytest.mark.gpu_test
@pytest.mark.parametrize("instance_type", SM_SINGLE_GPU_INSTANCE_TYPES)
def test_mnist_distributed_gpu_inductor(
    framework_version, ecr_image, instance_type, sagemaker_regions
):
    if Version(framework_version) in SpecifierSet("<2.0"):
        pytest.skip("skip the test as torch.compile only supported after 2.0")
    if "graviton" in ecr_image:
        pytest.skip("skip the graviton test for GPU instance types")
    model_dir = os.path.join(model_cpu_dir, "model_mnist_inductor.tar.gz")
    function_args = {
        "framework_version": framework_version,
        "instance_type": instance_type,
        "model_dir": model_dir,
    }
    invoke_pytorch_helper_function(
        ecr_image, sagemaker_regions, _test_mnist_distributed, function_args
    )


def _test_mnist_distributed(
    ecr_image,
    sagemaker_session,
    framework_version,
    instance_type,
    model_dir,
    accelerator_type=None,
    verify_logs=True,
):
    endpoint_name = sagemaker.utils.unique_name_from_base("sagemaker-pytorch-serving")

    model_data = sagemaker_session.upload_data(
        path=model_dir,
        key_prefix="sagemaker-pytorch-serving/models",
    )

    pytorch = PyTorchModel(
        model_data=model_data,
        role="SageMakerRole",
        entry_point="mnist.py",
        framework_version=framework_version,
        image_uri=ecr_image,
        sagemaker_session=sagemaker_session,
    )

    with timeout_and_delete_endpoint(endpoint_name, sagemaker_session, minutes=30):
        # Use accelerator type to differentiate EI vs. CPU and GPU. Don't use processor value
        if accelerator_type is not None:
            predictor = pytorch.deploy(
                initial_instance_count=1,
                instance_type=instance_type,
                accelerator_type=accelerator_type,
                endpoint_name=endpoint_name,
            )
        else:
            predictor = pytorch.deploy(
                initial_instance_count=1,
                instance_type=instance_type,
                endpoint_name=endpoint_name,
            )

        batch_size = 100
        data = np.random.rand(batch_size, 1, 28, 28).astype(np.float32)
        output = predictor.predict(data)
        assert output.shape == (batch_size, 10)

        #  Check for Cloudwatch logs
        if verify_logs:
            _check_for_cloudwatch_logs(endpoint_name, sagemaker_session)


def _check_for_cloudwatch_logs(endpoint_name, sagemaker_session):
    client = sagemaker_session.boto_session.client("logs")
    log_group_name = f"/aws/sagemaker/Endpoints/{endpoint_name}"
    time.sleep(30)
    identify_log_stream = client.describe_log_streams(
        logGroupName=log_group_name, orderBy="LastEventTime", descending=True, limit=5
    )

    try:
        log_stream_name = identify_log_stream["logStreams"][0]["logStreamName"]
    except IndexError as e:
        raise RuntimeError(
            f"Unable to look up log streams for the log group {log_group_name}"
        ) from e

    log_events_response = client.get_log_events(
        logGroupName=log_group_name, logStreamName=log_stream_name, limit=50, startFromHead=True
    )

    records_available = bool(log_events_response["events"])

    if not records_available:
        raise RuntimeError(
            f"records_available variable is false... No cloudwatch events getting logged for the group {log_group_name}"
        )
    else:
        LOGGER.info(
            f"Most recently logged events were found for the given log group {log_group_name} & log stream {log_stream_name}... Now verifying that TorchServe endpoint is logging on cloudwatch"
        )
        check_for_torchserve_response = client.filter_log_events(
            logGroupName=log_group_name,
            logStreamNames=[log_stream_name],
            filterPattern="Torch worker started.",
            limit=10,
            interleaved=False,
        )
        assert bool(check_for_torchserve_response["events"])
