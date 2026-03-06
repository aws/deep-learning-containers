"""HuggingFace helper utilities."""

import json
import logging

from botocore.exceptions import ClientError

from .aws import AWSSessionManager

LOGGER = logging.getLogger(__name__)


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
