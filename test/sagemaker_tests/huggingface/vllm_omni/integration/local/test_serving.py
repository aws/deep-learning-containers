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
from sagemaker.deserializers import BytesDeserializer
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer

from ...integration import ROLE, ensure_model_downloaded
from ...utils import local_mode_utils


@contextmanager
def _predictor(monkeypatch, image, sagemaker_local_session, instance_type):
    """Context manager for vLLM model deployment and cleanup.

    Model is extracted to /opt/ml/model by SageMaker from model_data tar.gz.
    vLLM loads the model from this local path.
    """
    # Download model from HuggingFace Hub if not already present
    model_data_path = ensure_model_downloaded()

    env = {
        "SM_VLLM_MODEL": "/opt/ml/model",
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
            monkeypatch.setattr(
                "sagemaker.local.entities.HEALTH_CHECK_TIMEOUT_LIMIT",
                600,
            )
            predictor = model.deploy(1, instance_type)
            yield predictor
        finally:
            if predictor:
                predictor.delete_endpoint()


def _assert_vllm_omni_speech_generation(predictor):
    """Test vLLM-Omni inference through the bundled SageMaker middleware."""
    predictor.serializer = JSONSerializer()
    predictor.deserializer = BytesDeserializer()

    data = {
        "input": "Hello world from SageMaker tests!",
        "voice": "ryan",
        "language": "English",
    }

    output = predictor.predict(
        data=data,
        initial_args={
            "CustomAttributes": "route=/v1/audio/speech",
        },
    )

    assert output is not None
    assert isinstance(output, bytes)
    assert len(output) > 0


@pytest.mark.model("qwen3-tts-12hz-1-7b-customvoice")
@pytest.mark.team("sagemaker-1p-algorithms")
def test_vllm_local_speech_generation(
    monkeypatch, docker_image, sagemaker_local_session, instance_type
):
    """Test vLLM-Omni local deployment with speech generation API."""
    with _predictor(monkeypatch, docker_image, sagemaker_local_session, instance_type) as predictor:
        _assert_vllm_omni_speech_generation(predictor)
