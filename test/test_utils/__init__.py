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
