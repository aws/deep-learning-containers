# Copyright 2018-2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import json
import os
import time

import boto3
import pytest
import sagemaker
from sagemaker import serializers
from sagemaker.config.config_schema import MODEL
from sagemaker.model import Model
from sagemaker.predictor import Predictor

# Fixed parameters
MODEL_ID = "Qwen/Qwen3-0.6B"
AWS_REGION = "us-west-2"
INSTANCE_TYPE = "ml.g5.12xlarge"
ROLE = "SageMakerRole"


def get_secret_hf_token():
    print("Retrieving HuggingFace token from AWS Secrets Manager...")
    secret_name = "test/hf_token"
    region_name = "us-west-2"

    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)
    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        print("Successfully retrieved HuggingFace token")
    except ClientError as e:
        print(f"Failed to retrieve HuggingFace token: {e}")
        raise e

    response = json.loads(get_secret_value_response["SecretString"])
    return response


def deploy_endpoint(name, image_uri, role, instance_type):
    try:
        print(f"Starting deployment of endpoint: {name}")
        print(f"Using image: {image_uri}")
        print(f"Instance type: {instance_type}")

        response = get_secret_hf_token()
        hf_token = response.get("HF_TOKEN")
        print("Creating SageMaker model...")

        model = Model(
            name=name,
            image_uri=image_uri,
            role=role,
            env={
                "SM_SGLANG_MODEL_PATH": MODEL_ID,
                "SM_SGLANG_REASONING_PARSER": "qwen3",
                "HF_TOKEN": hf_token,
            },
        )
        print("Model created successfully")
        print("Starting endpoint deployment (this may take 10-15 minutes)...")

        endpoint_config = model.deploy(
            instance_type=instance_type,
            initial_instance_count=1,
            endpoint_name=name,
            inference_ami_version="al2-ami-sagemaker-inference-gpu-3-1",
            wait=True,
        )
        print("Endpoint deployment completed successfully")
        return True
    except Exception as e:
        print(f"Deployment failed: {str(e)}")
        return False


def invoke_endpoint(endpoint_name, prompt, max_tokens=2400, temperature=0.01):
    try:
        print(f"Creating predictor for endpoint: {endpoint_name}")
        predictor = Predictor(
            endpoint_name=endpoint_name,
            serializer=serializers.JSONSerializer(),
        )

        payload = {
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": 50,
        }
        print(f"Sending inference request with prompt: '{prompt[:50]}...'")
        print(f"Request parameters: max_tokens={max_tokens}, temperature={temperature}")

        response = predictor.predict(payload)
        print("Inference request completed successfully")

        if isinstance(response, bytes):
            response = response.decode("utf-8")

        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError:
                print("Warning: Response is not valid JSON. Returning as string.")

        return response
    except Exception as e:
        print(f"Inference failed: {str(e)}")
        return None


def delete_endpoint(endpoint_name):
    try:
        sagemaker_client = boto3.client("sagemaker", region_name=AWS_REGION)

        print(f"Deleting endpoint: {endpoint_name}")
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)

        print(f"Deleting endpoint configuration: {endpoint_name}")
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)

        print(f"Deleting model: {endpoint_name}")
        sagemaker_client.delete_model(ModelName=endpoint_name)

        print("Successfully deleted all resources")
        return True
    except Exception as e:
        print(f"Error during deletion: {str(e)}")
        return False


def wait_for_endpoint(endpoint_name, timeout=1800):
    sagemaker_client = boto3.client("sagemaker", region_name=AWS_REGION)
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            status = response["EndpointStatus"]

            if status == "InService":
                return True
            elif status in ["Failed", "OutOfService"]:
                print(f"Endpoint creation failed with status: {status}")
                return False

            print(f"Endpoint status: {status}. Waiting...")
            time.sleep(30)
        except Exception as e:
            print(f"Error checking endpoint status: {str(e)}")
            return False

    print("Timeout waiting for endpoint to be ready")
    return False


@pytest.mark.processor("gpu")
@pytest.mark.team("conda")
def test_sglang_on_sagemaker(ecr_image):
    endpoint_name = f"test-{ecr_image}-{MODEL_ID.replace('/', '-')}-sglang-{INSTANCE_TYPE.replace('.', '-')}"

    print("\n" + "=" * 80)
    print("STARTING SGLang SAGEMAKER ENDPOINT TEST".center(80))
    print("=" * 80)
    print(f"Test Configuration:")
    print(f"     Image URI: {ecr_image}")
    print(f"     Endpoint name: {endpoint_name}")
    print(f"     Region: {AWS_REGION}")
    print(f"     Instance type: {INSTANCE_TYPE}")
    print("\n" + "-" * 80)
    print("PHASE 1: ENDPOINT DEPLOYMENT".center(80))
    print("-" * 80)

    if not deploy_endpoint(endpoint_name, ecr_image, ROLE, INSTANCE_TYPE):
        print("\n" + "=" * 80)
        print("DEPLOYMENT FAILED - CLEANING UP".center(80))
        print("=" * 80)
        # Cleanup any partially created resources
        delete_endpoint(endpoint_name)
        raise Exception("SageMaker endpoint deployment failed")

    print("\n" + "-" * 80)
    print("PHASE 2: WAITING FOR ENDPOINT READINESS".center(80))
    print("-" * 80)
    if not wait_for_endpoint(endpoint_name):
        print("\nEndpoint failed to become ready. Initiating cleanup...")
        delete_endpoint(endpoint_name)
        print("\n" + "=" * 80)
        print("ENDPOINT READINESS FAILED".center(80))
        print("=" * 80)
        raise Exception("SageMaker endpoint failed to become ready")

    print("\nEndpoint is ready for inference!")
    print("\n" + "-" * 80)
    print("PHASE 3: TESTING INFERENCE".center(80))
    print("-" * 80)
    test_prompt = "Write a python script to calculate square of n"

    response = invoke_endpoint(
        endpoint_name=endpoint_name, prompt=test_prompt, max_tokens=2400, temperature=0.01
    )

    if response:
        print("\n Inference test successful!")
        print("\n Response from endpoint:")
        print("-" * 40)
        if isinstance(response, (dict, list)):
            print(json.dumps(response, indent=2))
        else:
            print(response)
        print("-" * 40)

        print("\n" + "-" * 80)
        print(" PHASE 4: CLEANUP".center(80))
        print("-" * 80)
        if delete_endpoint(endpoint_name):
            print("\n" + "=" * 80)
            print(" TEST COMPLETED SUCCESSFULLY! ".center(80))
            print("=" * 80)
        else:
            print("\n Cleanup failed")
    else:
        print("\n No response received from the endpoint.")
        print("\n" + "-" * 80)
        print(" PHASE 4: CLEANUP (FAILED INFERENCE)".center(80))
        print("-" * 80)
        delete_endpoint(endpoint_name)
        print("\n" + "=" * 80)
        print(" TEST FAILED ".center(80))
        print("=" * 80)
        raise Exception("SageMaker endpoint inference test failed")
