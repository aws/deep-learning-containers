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
import requests
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

from ...integration import ROLE, ensure_model_downloaded
from ...utils import local_mode_utils


@contextmanager
def _predictor(image, sagemaker_local_session, instance_type):
    """Context manager for Llama.cpp model deployment and cleanup.

    Model is extracted to /opt/ml/model by SageMaker from model_data tar.gz.
    The container entrypoint runs llama-server behind a SageMaker-compatible
    proxy on port 8080 (/ping, /invocations -> OpenAI routes on llama-server).
    """
    # Download model from HuggingFace Hub if not already present
    model_data_path = ensure_model_downloaded()

    env = {
        "SM_LLAMACPP_MODEL": "/opt/ml/model/Qwen3.5-0.8B-UD-IQ2_XXS.gguf",
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


def _assert_sagemaker_ping_local():
    """SageMaker contract: GET /ping on the container HTTP port (local mode: 8080)."""
    response = requests.get("http://127.0.0.1:8080/ping", timeout=60)
    assert response.status_code == 200


def _assert_llamacpp_chat_prediction(predictor):
    """Test Llama.cpp inference using OpenAI-compatible chat completions API."""
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


def _assert_llamacpp_chat_prediction_explicit_route(predictor):
    """Same as chat test but forces target path via SageMaker CustomAttributes (proxy route=)."""
    predictor.serializer = JSONSerializer()
    predictor.deserializer = JSONDeserializer()

    data = {
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 16,
        "temperature": 0.3,
    }
    output = predictor.predict(
        data,
        custom_attributes="route=/v1/chat/completions",
    )

    assert output is not None
    assert "choices" in output


@pytest.mark.model("qwen3.5-0.8b")
@pytest.mark.team("sagemaker-1p-algorithms")
def test_llamacpp_local_chat(docker_image, sagemaker_local_session, instance_type):
    """Test Llama.cpp local deployment: /ping shim, /invocations chat, and explicit route=."""
    with _predictor(docker_image, sagemaker_local_session, instance_type) as predictor:
        _assert_sagemaker_ping_local()
        _assert_llamacpp_chat_prediction(predictor)
        _assert_llamacpp_chat_prediction_explicit_route(predictor)
