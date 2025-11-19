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
"""Common utility functions for all tests under module test/"""

import json
import logging
import random
import string

from botocore.exceptions import ClientError

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def random_suffix_name(resource_name, max_length, delimiter="-") -> str:
    rand_length = max_length - len(resource_name) - len(delimiter)
    rand = "".join(
        random.choice(string.ascii_lowercase + string.digits) for _ in range(rand_length)
    )
    return f"{resource_name}{delimiter}{rand}"


def get_hf_token(aws_session):
    LOGGER.info("Retrieving HuggingFace token from AWS Secrets Manager...")
    token_path = "test/hf_token"

    try:
        get_secret_value_response = aws_session.secretsmanager.get_secret_value(SecretId=token_path)
        LOGGER.info("Successfully retrieved HuggingFace token")
    except ClientError as e:
        LOGGER.error(f"Failed to retrieve HuggingFace token: {e}")
        raise e

    response = json.loads(get_secret_value_response["SecretString"])
    token = response.get("HF_TOKEN")
    return token
