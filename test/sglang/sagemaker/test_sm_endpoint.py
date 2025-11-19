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

import logging
import sys

import pytest
from test_utils import random_suffix_name

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))

# Test configuration
MODEL_ID = "Qwen/Qwen3-0.6B"
AWS_REGION = "us-west-2"
INSTANCE_TYPE = "ml.g5.12xlarge"
ROLE = "SageMakerRole"


def deploy_endpoint(endpoint_name, image_uri):
    try:
        LOGGER.debug(f"Starting deployment of endpoint: {endpoint_name}")
        LOGGER.debug(f"Using image: {image_uri}")
        LOGGER.debug(f"Instance type: {INSTANCE_TYPE}")

        response = get_secret_hf_token()
        hf_token = response.get("HF_TOKEN")
        LOGGER.info("Creating SageMaker model...")

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
        LOGGER.info("Model created successfully")
        LOGGER.info("Starting endpoint deployment (this may take 10-15 minutes)...")

        _ = model.deploy(
            instance_type=instance_type,
            initial_instance_count=1,
            endpoint_name=name,
            inference_ami_version="al2-ami-sagemaker-inference-gpu-3-1",
            wait=True,
        )
        LOGGER.info("Endpoint deployment completed successfully")
        return True
    except Exception as e:
        LOGGER.error(f"Deployment failed: {str(e)}")
        delete_endpoint(name)
        raise


@pytest.mark.parametrize("region", ["us-west-2"])
@pytest.mark.parametrize("instance_type", ["ml.g5.12xlarge"])
def test_sglang_on_sagemaker(image_uri):
    _ = random_suffix_name(
        f"test-sglang-{MODEL_ID.replace('/', '-').replace('.', '')}-{INSTANCE_TYPE.replace('.', '-')}",
        50,
    )
