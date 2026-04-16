"""Common utility functions for all tests under module test/
For test utility functions, please appropriately declare function argument types
and their output types for readability and reusability.
When necessary, use docstrings to explain the functions' mechanisms.
"""

import logging
import os
import random
import re
import string
import time
from collections.abc import Callable
from typing import Any

import boto3

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def random_suffix_name(resource_name: str, max_length: int, delimiter: str = "-") -> str:
    rand_length = max_length - len(resource_name) - len(delimiter)
    rand = "".join(
        random.choice(string.ascii_lowercase + string.digits) for _ in range(rand_length)
    )
    return f"{resource_name}{delimiter}{rand}"


def clean_string(text: str, symbols_to_remove: str, replacement: str = "-") -> str:
    for symbol in symbols_to_remove:
        text = text.replace(symbol, replacement)
    return text


def wait_for_status(
    expected_status: str,
    wait_periods: int,
    period_length: int,
    get_status_method: Callable[[Any], str],
    *method_args: Any,
) -> bool:
    actual_status = None
    for i in range(wait_periods):
        time.sleep(period_length)
        LOGGER.debug(f"Time passed while waiting: {period_length * (i + 1)}s.")
        actual_status = get_status_method(*method_args)
        if actual_status == expected_status:
            return True

    LOGGER.error(f"Wait for status: {expected_status} timed out. Actual status: {actual_status}")
    return False


def get_repository_and_tag_from_image_uri(image_uri):
    """
    Return the name of the repository holding the image

    :param image_uri: URI of the image
    :return: <str> repository name, <str> tag
    """
    repository_uri, tag = image_uri.split(":")
    _, repository_name = repository_uri.split("/")
    return repository_name, tag


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
    region_pattern = r"(us(-gov)?|af|ap|ca|cn|eu|il|me|sa)-(central|(north|south)?(east|west)?)-\d+"
    region_search = re.search(region_pattern, image_uri)
    assert region_search, f"{image_uri} must have region that matches {region_pattern}"
    return region_search.group()


def get_unique_name_from_tag(image_uri):
    """
    Return the unique from the image tag.
    :param image_uri: ECR image URI
    :return: unique name
    """
    return re.sub("[^A-Za-z0-9]+", "", image_uri)


def get_repository_local_path():
    git_repo_path = os.getcwd().split("/test/")[0]
    return git_repo_path


class CudaVersionTagNotFoundException(Exception):
    """
    When none of the tags of a GPU image have a Cuda version in them
    """

    pass


def get_framework_from_image_uri(image_uri):
    framework_map = {
        "huggingface-pytorch": "huggingface_pytorch",
        "huggingface-tensorflow": "huggingface_tensorflow",
        "huggingface-vllm": "huggingface_vllm",
        "huggingface-sglang": "huggingface_sglang",
        "stabilityai-pytorch": "stabilityai_pytorch",
        "pytorch": "pytorch",
        "tensorflow": "tensorflow",
        "autogluon": "autogluon",
        "base": "base",
        "vllm": "vllm",
        "sglang": "sglang",
    }
    for image_pattern, framework in framework_map.items():
        if image_pattern in image_uri:
            return framework
    return None


def get_framework_and_version_from_tag(image_uri):
    tested_framework = get_framework_from_image_uri(image_uri)
    if not tested_framework:
        raise RuntimeError(f"Cannot find framework in image uri {image_uri}")
    tag_framework_version = re.search(r"(\d+(\.\d+){1,2})", image_uri).groups()[0]
    return tested_framework, tag_framework_version


def get_all_the_tags_of_an_image_from_ecr(ecr_client, image_uri):
    account_id = get_account_id_from_image_uri(image_uri)
    image_repo_name, image_tag = get_repository_and_tag_from_image_uri(image_uri)
    response = ecr_client.describe_images(
        registryId=account_id,
        repositoryName=image_repo_name,
        imageIds=[{"imageTag": image_tag}],
    )
    return response["imageDetails"][0]["imageTags"]


def get_cuda_version_from_tag(image_uri):
    cuda_str = ["cu", "gpu"]
    image_region = get_region_from_image_uri(image_uri)
    ecr_client = boto3.Session(region_name=image_region).client("ecr")
    _, local_image_tag = get_repository_and_tag_from_image_uri(image_uri)
    all_image_tags = [local_image_tag]
    try:
        all_image_tags = get_all_the_tags_of_an_image_from_ecr(ecr_client, image_uri)
    except ecr_client.exceptions.ImageNotFoundException:
        LOGGER.info(f"Image {image_uri} not found in ECR — using local tag only")
    for image_tag in all_image_tags:
        if all(keyword in image_tag for keyword in cuda_str):
            cuda_framework_version = re.search(r"(cu\d+)-", image_tag).groups()[0]
            return cuda_framework_version
    if "gpu" in image_uri:
        raise CudaVersionTagNotFoundException()
    return None
