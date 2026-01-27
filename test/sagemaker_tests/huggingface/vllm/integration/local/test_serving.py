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

from contextlib import contextmanager

import pytest
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

from ...integration import ROLE, ensure_model_downloaded
from ...utils import local_mode_utils


@contextmanager
def _predictor(image, sagemaker_local_session, instance_type):
    """Context manager for vLLM model deployment and cleanup.
    
    Model is extracted to /opt/ml/model by SageMaker from model_data tar.gz.
    vLLM loads the model from this local path.
    """
    # Download model from HuggingFace Hub if not already present
    model_data_path = ensure_model_downloaded()
    
    env = {
        "SM_VLLM_MODEL": "/opt/ml/model",
        "SM_VLLM_MAX_MODEL_LEN": "512",
        "SM_VLLM_HOST": "0.0.0.0",
    }

    model = Model(
        model_data=f"file://{model_data_path}",
        role=ROLE,
        image_uri=image,
        env=env,
        sagemaker_session=sagemaker_local_session,
        predictor_cls=Predictor,
    )
    with local_mode_utils.lock():
        predictor = None
        try:
            predictor = model.deploy(1, instance_type)
            yield predictor
        finally:
            if predictor is not None:
                predictor.delete_endpoint()


def _assert_vllm_prediction(predictor):
    """Test vLLM inference using OpenAI-compatible completions API."""
    predictor.serializer = JSONSerializer()
    predictor.deserializer = JSONDeserializer()

    data = {
        "prompt": "What is Deep Learning?",
        "max_tokens": 50,
        "temperature": 0.7,
    }
    output = predictor.predict(data)

    assert output is not None
    assert "choices" in output


def _assert_vllm_chat_prediction(predictor):
    """Test vLLM inference using OpenAI-compatible chat completions API."""
    predictor.serializer = JSONSerializer()
    predictor.deserializer = JSONDeserializer()

    data = {
        "messages": [{"role": "user", "content": "What is Deep Learning?"}],
        "max_tokens": 50,
        "temperature": 0.7,
    }
    output = predictor.predict(data)

    assert output is not None
    assert "choices" in output


@pytest.mark.model("qwen2.5-0.5b")
@pytest.mark.team("sagemaker-1p-algorithms")
def test_vllm_local_completions(docker_image, sagemaker_local_session, instance_type):
    """Test vLLM local deployment with completions API."""
    with _predictor(docker_image, sagemaker_local_session, instance_type) as predictor:
        _assert_vllm_prediction(predictor)


@pytest.mark.model("qwen2.5-0.5b")
@pytest.mark.team("sagemaker-1p-algorithms")
def test_vllm_local_chat(docker_image, sagemaker_local_session, instance_type):
    """Test vLLM local deployment with chat completions API."""
    with _predictor(docker_image, sagemaker_local_session, instance_type) as predictor:
        _assert_vllm_chat_prediction(predictor)
