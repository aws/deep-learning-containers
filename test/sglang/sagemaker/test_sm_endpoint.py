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
import time
from pprint import pformat

import pytest
from sagemaker import serializers
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from test_utils import get_hf_token, random_suffix_name
from test_utils.aws import AWSSessionManager

# To enable debugging, change logging.INFO to logging.DEBUG
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class TestEndpoint:
    def deploy_endpoint(self, aws_session, endpoint_name, role, image_uri, instance_type, model_id):
        try:
            LOGGER.debug(f"Starting deployment of endpoint: {endpoint_name}")
            LOGGER.debug(f"Using image: {image_uri}")
            LOGGER.debug(f"Instance type: {instance_type}")

            LOGGER.info("Creating SageMaker model...")
            hf_token = get_hf_token(aws_session)
            model = Model(
                name=endpoint_name,
                image_uri=image_uri,
                role=role,
                env={
                    "SM_SGLANG_MODEL_PATH": model_id,
                    "HF_TOKEN": hf_token,
                },
            )
            LOGGER.info("Model created successfully")

            LOGGER.info("Starting endpoint deployment (this may take 10-15 minutes)...")
            _ = model.deploy(
                instance_type=instance_type,
                initial_instance_count=1,
                endpoint_name=endpoint_name,
                inference_ami_version="al2-ami-sagemaker-inference-gpu-3-1",
                wait=True,
            )
            LOGGER.info("Endpoint deployment completed successfully")

        except Exception as e:
            LOGGER.error(f"Endpoint deployment failed: {str(e)}")
            raise

    def wait_for_endpoint(self, aws_session, endpoint_name, timeout=1800):
        sagemaker_client = aws_session.sagemaker
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
                status = response["EndpointStatus"]
                LOGGER.debug(f"Describe endpoint response: {pformat(response)}")

                if status == "InService":
                    return
                elif status in ["Failed", "OutOfService"]:
                    LOGGER.error(f"Endpoint creation failed with status: {status}")
                    raise

                LOGGER.info(
                    f"Endpoint status: {status}. Time passed: {time.time() - start_time} ..."
                )
                time.sleep(30)

            except Exception as e:
                LOGGER.error(f"Error checking endpoint status: {str(e)}")
                raise

    def invoke_endpoint(
        self, aws_session, endpoint_name, model_id, prompt, max_tokens=2400, temperature=0.01
    ):
        try:
            LOGGER.info(f"Creating predictor for endpoint: {endpoint_name}")
            predictor = Predictor(
                endpoint_name=endpoint_name,
                serializer=serializers.JSONSerializer(),
            )

            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
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

            if not response:
                LOGGER.error("No response received from the endpoint.")
                raise

            return response

        except Exception as e:
            LOGGER.error(f"Inference failed: {str(e)}")
            raise

    def delete_endpoint(self, aws_session, endpoint_name):
        try:
            sagemaker_client = aws_session.sagemaker

            LOGGER.info(f"Deleting endpoint: {endpoint_name}")
            sagemaker_client.delete_endpoint(EndpointName=endpoint_name)

            LOGGER.info(f"Deleting endpoint configuration: {endpoint_name}")
            sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)

            LOGGER.info(f"Deleting model: {endpoint_name}")
            sagemaker_client.delete_model(ModelName=endpoint_name)

            LOGGER.info("Successfully deleted all resources")
        except Exception as e:
            LOGGER.error(f"Error during deletion: {str(e)}")
            raise

    @pytest.mark.parametrize("region", ["us-west-2"])
    @pytest.mark.parametrize("instance_type", ["ml.g5.12xlarge"])
    @pytest.mark.parametrize("model_id", ["Qwen/Qwen3-0.6B"])
    def test_sglang_sagemaker_endpoint(self, image_uri, region, instance_type, model_id):
        aws_session = AWSSessionManager(region)
        sagemaker_role = aws_session.iam_resource.Role("SageMakerRole").arn

        endpoint_name = random_suffix_name(
            f"test-sglang-{model_id.replace('/', '-').replace('.', '')}-{instance_type.replace('.', '-')}",
            50,
        )

        try:
            # Phase 1: Deployment
            self.deploy_endpoint(
                aws_session=aws_session,
                endpoint_name=endpoint_name,
                role=sagemaker_role,
                image_uri=image_uri,
                instance_type=instance_type,
                model_id=model_id,
            )

            # Phase 2: Endpoint Readiness
            self.wait_for_endpoint(aws_session=aws_session, endpoint_name=endpoint_name)

            # Phase 3: Testing Inference
            test_prompt = "Write a python script to calculate square of n"
            response = self.invoke_endpoint(
                aws_session=aws_session,
                endpoint_name=endpoint_name,
                model_id=model_id,
                prompt=test_prompt,
            )
            LOGGER.info(f"Model response: {pformat(response)}")
            LOGGER.info("Inference test successful!")

        except Exception as e:
            LOGGER.error(e)

        finally:
            # Phase 4: Cleanup
            self.delete_endpoint(aws_session=aws_session, endpoint_name=endpoint_name)
