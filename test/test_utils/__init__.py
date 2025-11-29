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
"""Common utility functions for all tests under module test/
For test utility functions, please appropriately declare function argument types
and their output types for readability and reusability.
When necessary, use docstrings to explain the functions' mechanisms.
"""

import json
import logging
import random
import string
import time
from collections.abc import Callable
from typing import Any

from aws import AWSSessionManager
from botocore.exceptions import ClientError

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


def get_hf_token(aws_session: AWSSessionManager) -> str:
    LOGGER.info("Retrieving HuggingFace token from AWS Secrets Manager...")
    token_path = "test/hf_token"

    try:
        get_secret_value_response = aws_session.secretsmanager.get_secret_value(SecretId=token_path)
        LOGGER.info("Successfully retrieved HuggingFace token")
    except ClientError as e:
        LOGGER.error(f"Failed to retrieve HuggingFace token: {e}")
        raise e

    # Do not print secrets token in logs
    response = json.loads(get_secret_value_response["SecretString"])
    token = response.get("HF_TOKEN")
    return token
