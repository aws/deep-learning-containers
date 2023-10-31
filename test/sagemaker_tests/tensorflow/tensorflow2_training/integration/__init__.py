#  Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  A copy of the License is located at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  or in the "license" file accompanying this file. This file is distributed
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#  express or implied. See the License for the specific language governing
#  permissions and limitations under the License.
from __future__ import absolute_import

import logging
import os
import subprocess
import boto3
from base64 import b64decode
import sys
import re
import boto3
import botocore
import subprocess


logging.getLogger("boto3").setLevel(logging.INFO)
logging.getLogger("botocore").setLevel(logging.INFO)

RESOURCE_PATH = os.path.join(os.path.dirname(__file__), "..", "resources")
DEFAULT_TIMEOUT = 120

# these regions have some p2 and p3 instances, but not enough for automated testing
NO_P2_REGIONS = [
    "ap-northeast-3",
    "ca-central-1",
    "eu-central-1",
    "eu-west-2",
    "us-west-1",
    "eu-west-3",
    "eu-north-1",
    "sa-east-1",
    "ap-east-1",
    "me-south-1",
    "cn-northwest-1",
    "eu-south-1",
    "af-south-1",
    "il-central-1",
]
NO_P3_REGIONS = [
    "ap-northeast-1",
    "ap-northeast-2",
    "ap-northeast-3",
    "ap-southeast-1",
    "ap-southeast-2",
    "ap-south-1",
    "ca-central-1",
    "eu-central-1",
    "eu-west-2",
    "us-west-1",
    "eu-west-3",
    "eu-north-1",
    "sa-east-1",
    "ap-east-1",
    "me-south-1",
    "cn-northwest-1",
    "eu-south-1",
    "af-south-1",
    "il-central-1",
]
NO_P4_REGIONS = [
    "ap-northeast-3",
    "ap-southeast-1",
    "ap-southeast-2",
    "ap-south-1",
    "ca-central-1",
    "eu-central-1",
    "eu-west-2",
    "us-west-1",
    "eu-west-3",
    "eu-north-1",
    "sa-east-1",
    "ap-east-1",
    "me-south-1",
    "cn-northwest-1",
    "eu-south-1",
    "af-south-1",
    "il-central-1",
]


def _botocore_resolver():
    """
    Get the DNS suffix for the given region.
    :return: endpoint object
    """
    loader = botocore.loaders.create_loader()
    return botocore.regions.EndpointResolver(loader.load_data("endpoints"))


def get_ecr_registry(account, region):
    """
    Get prefix of ECR image URI
    :param account: Account ID
    :param region: region where ECR repo exists
    :return: AWS ECR registry
    """
    endpoint_data = _botocore_resolver().construct_endpoint("ecr", region)
    return "{}.dkr.{}".format(account, endpoint_data["hostname"])


def get_framework_from_image_uri(image_uri):
    return (
        "huggingface_tensorflow_trcomp"
        if "huggingface-tensorflow-trcomp" in image_uri
        else "huggingface_tensorflow"
        if "huggingface-tensorflow" in image_uri
        else "huggingface_pytorch_trcomp"
        if "huggingface-pytorch-trcomp" in image_uri
        else "huggingface_pytorch"
        if "huggingface-pytorch" in image_uri
        else "mxnet"
        if "mxnet" in image_uri
        else "pytorch"
        if "pytorch" in image_uri
        else "tensorflow"
        if "tensorflow" in image_uri
        else None
    )


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
