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
import re
import boto3
import sagemaker
import botocore
from sagemaker import utils


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
    return f'{account}.dkr.{endpoint_data["hostname"]}'


def get_sagemaker_session(region, default_bucket=None) -> sagemaker.Session:
    """
    Initiates a SageMaker session in specified SageMaker region
    optonally setting the default bucket
    :param region: SageMaker region
    :param default_bucket: default bucket
    :return: returns a SageMaker session
    """
    return sagemaker.Session(
        boto_session=boto3.Session(region_name=region),
        default_bucket=default_bucket
    )

def create_endpoint_name(prefix) -> str:
    """
    Returns unique name for the endpoint
    :param prefix: A prefix for the unique name
    :return: str unique endpoint name
    """
    return utils.unique_name_from_base(prefix)

def get_account_id_from_image_uri(image_uri):
    """
    Find the account ID where the image is located

    :param image_uri: <str> ECR image URI
    :return: <str> AWS Account ID
    """
    return image_uri.split(".")[0]


def get_region_from_image_uri(image_uri):
    """
    Find the region where the image is located

    :param image_uri: <str> ECR image URI
    :return: <str> AWS Region Name
    """
    region_pattern = r"(us(-gov)?|ap|ca|cn|eu|sa)-(central|(north|south)?(east|west)?)-\d+"
    region_search = re.search(region_pattern, image_uri)
    assert region_search, f"{image_uri} must have region that matches {region_pattern}"
    return region_search.group()


def get_framework_name(image_uri):
    """
    Get the framework name from image uri
    :param image_uri: <str> image uri containing framework name
    :return: Returns str representing framework
    """
    return (
        "mxnet" if "mxnet" in image_uri
        else "pytorch" if "pytorch" in image_uri
        else "tensorflow" if "tensorflow" in image_uri
        else None
    )

def get_framework_version(image_uri):
    """
    Gets framework version from image_uri
    :param image_uri: Where framework_version will be regexed from
    :return: Returns str representing framework_version
    """
    framework_version = re.search(r":\s*([\d][.][\d]+)", image_uri).group(1)
    return framework_version


def get_py_version(image_uri):
    """
    Gets python version from image_uri
    :param image_uri: Where python_version will be regexed from
    :return: Returns str representing python_version
    """
    python_version = re.search(r"py\s*([\d])", image_uri).group()
    return python_version


def get_repository_and_tag_from_image_uri(image_uri):
    """
    Return the name of the repository holding the image

    :param image_uri: URI of the image
    :return: <str> repository name
    """
    repository_uri, tag = image_uri.split(":")
    _, repository_name = repository_uri.split("/")
    return repository_name, tag