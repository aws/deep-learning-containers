# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import re
import json

import boto3

RESOURCE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources"))
DEFAULT_TIMEOUT = 15


class NoLogStreamFoundError(Exception):
    pass


class SageMakerEndpointFailure(Exception):
    pass


def dump_logs_from_cloudwatch(e, region="us-west-2"):
    """
    Function to dump logs from cloudwatch during error handling
    """
    endpoint_regex = re.compile(r"Error hosting endpoint ((\w|-)+):")
    endpoint_match = endpoint_regex.search(str(e))
    if endpoint_match:
        logs_client = boto3.client("logs", region_name=region)
        endpoint = endpoint_match.group(1)
        log_group_name = f"/aws/sagemaker/Endpoints/{endpoint}"
        log_stream_resp = logs_client.describe_log_streams(logGroupName=log_group_name)
        all_traffic_log_stream = ""
        for log_stream in log_stream_resp.get("logStreams", []):
            log_stream_name = log_stream.get("logStreamName")
            # Format of AllTraffic log stream should be AllTraffic/<instance_id>
            if log_stream_name.startswith("AllTraffic"):
                all_traffic_log_stream = log_stream_name
                break
        if not all_traffic_log_stream:
            raise NoLogStreamFoundError(
                f"Cannot find all traffic log streams for endpoint {endpoint}"
            ) from e
        events = logs_client.get_log_events(
            logGroupName=log_group_name, logStreamName=all_traffic_log_stream
        )
        raise SageMakerEndpointFailure(
            f"Error from endpoint {endpoint}:\n{json.dumps(events, indent=4)}"
        ) from e


def get_cuda_version_from_tag(image_uri):
    """
    Return the cuda version from the image tag as cuXXX
    :param image_uri: ECR image URI
    :return: cuda version as cuXXX
    """
    cuda_framework_version = None
    cuda_str = ["cu", "gpu"]
    image_region = get_region_from_image_uri(image_uri)
    ecr_client = boto3.Session(region_name=image_region).client("ecr")
    all_image_tags = get_all_the_tags_of_an_image_from_ecr(ecr_client, image_uri)

    for image_tag in all_image_tags:
        if all(keyword in image_tag for keyword in cuda_str):
            cuda_framework_version = re.search(r"(cu\d+)-", image_tag).groups()[0]
            return cuda_framework_version

    if "gpu" in image_uri:
        raise CudaVersionTagNotFoundException()
    else:
        return None


def get_framework_and_version_from_tag(image_uri):
    """
    Return the framework and version from the image tag.

    :param image_uri: ECR image URI
    :return: framework name, framework version
    """
    tested_framework = get_framework_from_image_uri(image_uri)
    allowed_frameworks = (
        "huggingface_tensorflow_trcomp",
        "huggingface_pytorch_trcomp",
        "huggingface_tensorflow",
        "huggingface_pytorch",
        "tensorflow",
        "mxnet",
        "pytorch",
    )

    if not tested_framework:
        raise RuntimeError(
            f"Cannot find framework in image uri {image_uri} "
            f"from allowed frameworks {allowed_frameworks}"
        )

    tag_framework_version = re.search(r"(\d+(\.\d+){1,2})", image_uri).groups()[0]

    return tested_framework, tag_framework_version


def get_processor_from_image_uri(image_uri):
    """
    Return processor from the image URI

    Assumes image uri includes -<processor> in it's tag, where <processor> is one of cpu, gpu or eia.

    :param image_uri: ECR image URI
    :return: cpu, gpu, eia, neuron or hpu
    """
    allowed_processors = ["eia", "neuronx", "neuron", "cpu", "gpu", "hpu"]

    for processor in allowed_processors:
        match = re.search(rf"-({processor})", image_uri)
        if match:
            return match.group(1)
    raise RuntimeError("Cannot find processor")
