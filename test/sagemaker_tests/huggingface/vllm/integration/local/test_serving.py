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
from contextlib import contextmanager

import pytest
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

from ...integration import ROLE

LOGGER = logging.getLogger(__name__)


@contextmanager
def _predictor(image, sagemaker_local_session, instance_type, model_id):
    """Context manager for vLLM model deployment and cleanup."""
    env = {
        "SM_VLLM_MODEL": model_id,
        "SM_VLLM_MAX_MODEL_LEN": "512",
        "SM_VLLM_HOST": "0.0.0.0",
    }

    model = Model(
        role=ROLE,
        image_uri=image,
        env=env,
        sagemaker_session=sagemaker_local_session,
        predictor_cls=Predictor,
    )

    predictor = None
    try:
        predictor = model.deploy(1, instance_type)
        yield predictor
    finally:
        if predictor is not None:
            predictor.delete_endpoint()


def _assert_vllm_prediction(predictor):
    """Test vLLM inference using OpenAI-compatible API format."""
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
    # vLLM returns OpenAI-compatible response with 'choices' field
    assert "choices" in output or "text" in output


def _assert_vllm_chat_prediction(predictor):
    """Test vLLM inference using chat completions format."""
    predictor.serializer = JSONSerializer()
    predictor.deserializer = JSONDeserializer()

    # vLLM chat completions format
    data = {
        "messages": [
            {"role": "user", "content": "What is Deep Learning?"}
        ],
        "max_tokens": 50,
        "temperature": 0.7,
    }

    LOGGER.info(f"Running chat inference with data: {data}")
    output = predictor.predict(data)
    LOGGER.info(f"Output: {json.dumps(output)}")

    assert output is not None
    assert "choices" in output


@pytest.mark.model("qwen3-0.6b")
@pytest.mark.processor("gpu")
@pytest.mark.gpu_test
@pytest.mark.team("sagemaker-1p-algorithms")
def test_vllm_local_completions(ecr_image, sagemaker_local_session, instance_type):
    """Test vLLM local deployment with completions API."""
    instance_type = instance_type if instance_type != "local" else "local_gpu"
    with _predictor(
        ecr_image, sagemaker_local_session, instance_type, "Qwen/Qwen3-0.6B"
    ) as predictor:
        _assert_vllm_prediction(predictor)


@pytest.mark.model("qwen3-0.6b")
@pytest.mark.processor("gpu")
@pytest.mark.gpu_test
@pytest.mark.team("sagemaker-1p-algorithms")
def test_vllm_local_chat(ecr_image, sagemaker_local_session, instance_type):
    """Test vLLM local deployment with chat completions API."""
    instance_type = instance_type if instance_type != "local" else "local_gpu"
    with _predictor(
        ecr_image, sagemaker_local_session, instance_type, "Qwen/Qwen3-0.6B"
    ) as predictor:
        _assert_vllm_chat_prediction(predictor)
