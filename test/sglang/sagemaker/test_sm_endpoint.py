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
"""Integration test for serving endpoint with SGLang DLC"""

import json
import logging
from pprint import pformat

import pytest
from botocore.exceptions import ClientError
from sagemaker import serializers
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from test_utils import clean_string, random_suffix_name, wait_for_status
from test_utils.aws import AWSSessionManager

# To enable debugging, change logging.INFO to logging.DEBUG
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

ENDPOINT_WAIT_PERIOD = 60
ENDPOINT_WAIT_LENGTH = 30
ENDPOINT_INSERVICE = "InService"


def get_endpoint_status(sagemaker_client, endpoint_name):
    response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
    LOGGER.debug(f"Describe endpoint response: {pformat(response)}")
    return response["EndpointStatus"]


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


@pytest.fixture(scope="module")
def aws_session():
    return AWSSessionManager()


@pytest.fixture(scope="function")
def model_id(request):
    # Return the model_id given by the test parameter
    return request.param


@pytest.fixture(scope="function")
def instance_type(request):
    # Return the model_id given by the test parameter
    return request.param


@pytest.fixture(scope="function")
def model_package(aws_session, image_uri, model_id):
    sagemaker_client = aws_session.sagemaker
    sagemaker_role = aws_session.iam_resource.Role("SageMakerRole").arn
    cleaned_id = clean_string(model_id.split("/")[1], "_./")
    model_name = random_suffix_name(f"sglang-{cleaned_id}-model-package", 50)

    LOGGER.debug(f"Using image: {image_uri}")
    LOGGER.debug(f"Model ID: {model_id}")

    LOGGER.info(f"Creating SageMaker model: {model_name}...")
    hf_token = get_hf_token(aws_session)
    model = Model(
        name=model_name,
        image_uri=image_uri,
        role=sagemaker_role,
        predictor_cls=Predictor,
        env={
            "SM_SGLANG_MODEL_PATH": model_id,
            "HF_TOKEN": hf_token,
        },
    )
    LOGGER.info("Model created successfully")

    yield model

    LOGGER.info(f"Deleting model: {model_name}")
    sagemaker_client.delete_model(ModelName=model_name)


@pytest.fixture(scope="function")
def model_endpoint(aws_session, model_package, instance_type):
    sagemaker_client = aws_session.sagemaker
    model = model_package
    cleaned_instance = clean_string(instance_type, "_./")
    endpoint_name = random_suffix_name(f"sglang-{cleaned_instance}-endpoint", 50)

    LOGGER.debug(f"Using instance type: {instance_type}")

    LOGGER.info("Starting endpoint deployment (this may take 10-15 minutes)...")
    predictor = model.deploy(
        instance_type=instance_type,
        initial_instance_count=1,
        endpoint_name=endpoint_name,
        inference_ami_version="al2-ami-sagemaker-inference-gpu-3-1",
        serializer=serializers.JSONSerializer(),
        wait=True,
    )
    LOGGER.info("Endpoint deployment completed successfully")

    LOGGER.info(f"Waiting for endpoint {ENDPOINT_INSERVICE} status ...")
    assert wait_for_status(
        ENDPOINT_INSERVICE,
        ENDPOINT_WAIT_PERIOD,
        ENDPOINT_WAIT_LENGTH,
        get_endpoint_status,
        sagemaker_client,
        endpoint_name,
    )

    yield predictor

    LOGGER.info(f"Deleting endpoint: {endpoint_name}")
    sagemaker_client.delete_endpoint(EndpointName=endpoint_name)

    LOGGER.info(f"Deleting endpoint configuration: {endpoint_name}")
    sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)


@pytest.mark.parametrize("instance_type", ["ml.g5.12xlarge"], indirect=True)
@pytest.mark.parametrize("model_id", ["Qwen/Qwen3-0.6B"], indirect=True)
def test_sglang_sagemaker_endpoint(model_endpoint, model_id):
    predictor = model_endpoint

    prompt = "Write a python script to calculate square of n"
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2400,
        "temperature": 0.01,
        "top_p": 0.9,
        "top_k": 50,
    }
    LOGGER.debug(f"Sending inference request with payload: {pformat(payload)}")

    response = predictor.predict(payload)
    LOGGER.info("Inference request invoked successfully")

    if isinstance(response, bytes):
        response = response.decode("utf-8")

    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            LOGGER.warning("Response is not valid JSON. Returning as string.")

    assert response, "Model response is empty, failing endpoint test!"

    LOGGER.info(f"Model response: {pformat(response)}")
    LOGGER.info("Inference test successful!")
