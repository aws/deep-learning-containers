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
from pathlib import Path
import sys

import pytest
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

VLLM_OMNI_TEST_DIR = Path(__file__).resolve().parents[2]
if str(VLLM_OMNI_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(VLLM_OMNI_TEST_DIR))

from integration import ROLE, ensure_model_downloaded
from utils import local_mode_utils


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


def _assert_vllm_omni_image_generation(predictor):
    """Test vLLM-Omni inference through the bundled SageMaker middleware."""
    predictor.serializer = JSONSerializer()
    predictor.deserializer = JSONDeserializer()

    data = {
        "task": "text-to-image",
        "prompt": "A cat sitting on a mat.",
    }
    output = predictor.predict(data)

    assert output is not None
    assert "data" in output
    assert "b64_json" in output["data"][0]


@pytest.mark.model("z-image-turbo")
@pytest.mark.team("sagemaker-1p-algorithms")
def test_vllm_local_image_generation(
    docker_image, sagemaker_local_session, instance_type
):
    """Test vLLM-Omni local deployment with image generation API."""
    with _predictor(docker_image, sagemaker_local_session, instance_type) as predictor:
        _assert_vllm_omni_image_generation(predictor)
