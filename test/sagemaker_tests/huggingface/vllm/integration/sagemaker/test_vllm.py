# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import logging

import pytest
import sagemaker
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

from ...integration import dump_logs_from_cloudwatch
from ...integration.sagemaker.timeout import timeout_and_delete_endpoint
from ..... import invoke_sm_endpoint_helper_function

LOGGER = logging.getLogger(__name__)


@pytest.mark.model("bloom-560m")
@pytest.mark.processor("gpu")
@pytest.mark.gpu_test
@pytest.mark.team("sagemaker-1p-algorithms")
def test_vllm_bloom(framework_version, ecr_image, instance_type, sagemaker_regions):
    invoke_sm_endpoint_helper_function(
        ecr_image=ecr_image,
        sagemaker_regions=sagemaker_regions,
        test_function=_test_vllm_model,
        dump_logs_from_cloudwatch=dump_logs_from_cloudwatch,
        framework_version=framework_version,
        instance_type=instance_type,
        model_id="bigscience/bloom-560m",
    )


@pytest.mark.model("qwen3-8b")
@pytest.mark.processor("gpu")
@pytest.mark.gpu_test
@pytest.mark.team("sagemaker-1p-algorithms")
def test_vllm_qwen(framework_version, ecr_image, instance_type, sagemaker_regions):
    invoke_sm_endpoint_helper_function(
        ecr_image=ecr_image,
        sagemaker_regions=sagemaker_regions,
        test_function=_test_vllm_model,
        dump_logs_from_cloudwatch=dump_logs_from_cloudwatch,
        framework_version=framework_version,
        instance_type=instance_type,
        model_id="Qwen/Qwen3-8B",
    )


def _test_vllm_model(
    image_uri,
    sagemaker_session,
    instance_type,
    model_id,
    framework_version=None,
    **kwargs,
):
    """Test vLLM model deployment and inference using OpenAI-compatible API format
    
    Uses sagemaker.model.Model for SDK v3 compatibility instead of HuggingFaceModel.
    
    Args:
        image_uri: ECR image URI
        sagemaker_session: SageMaker session
        instance_type: ML instance type
        model_id: HuggingFace model ID
        framework_version: Optional version info
        **kwargs: Additional args from helper (boto_session, sagemaker_client, etc.)
    """
    endpoint_name = sagemaker.utils.unique_name_from_base("sagemaker-hf-vllm-serving")

    env = {
        "SM_VLLM_MODEL": model_id,
        "SM_VLLM_MAX_MODEL_LEN": "512",
        "SM_VLLM_HOST": "0.0.0.0",
    }

    model = Model(
        name=endpoint_name,
        image_uri=image_uri,
        role="SageMakerRole",
        env=env,
        sagemaker_session=sagemaker_session,
        predictor_cls=Predictor,
    )

    with timeout_and_delete_endpoint(endpoint_name, sagemaker_session, minutes=45):
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            container_startup_health_check_timeout=1800,
            inference_ami_version="al2-ami-sagemaker-inference-gpu-3-1",
        )

        predictor.serializer = JSONSerializer()
        predictor.deserializer = JSONDeserializer()

        # vLLM uses OpenAI-compatible API format
        data = {
            "prompt": "What is Deep Learning?",
            "max_tokens": 50,
            "temperature": 0.7,
        }

        LOGGER.info(f"Running inference with data: {data}")
        output = predictor.predict(data)
        LOGGER.info(f"Output: {json.dumps(output)}")

        assert output is not None