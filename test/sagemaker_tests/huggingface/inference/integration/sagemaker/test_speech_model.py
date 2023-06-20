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

import pytest
import sagemaker

from packaging.version import Version
from packaging.specifiers import SpecifierSet

from sagemaker.huggingface import HuggingFaceModel
from sagemaker.predictor import Predictor
from sagemaker.serializers import DataSerializer
from sagemaker.deserializers import JSONDeserializer

from test.test_utils import get_framework_and_version_from_tag
from ...integration import model_dir, dump_logs_from_cloudwatch, audio_sample_file_path
from ...integration.sagemaker.timeout import timeout_and_delete_endpoint
from ..... import invoke_sm_endpoint_helper_function


@pytest.mark.model("wav2vec2")
@pytest.mark.processor("cpu")
@pytest.mark.cpu_test
def test_speech_model_cpu(framework_version, ecr_image, instance_type, sagemaker_regions):
    framework, _ = get_framework_and_version_from_tag(ecr_image)
    if "pytorch" in framework and Version(framework_version) in SpecifierSet("==1.9.*"):
        pytest.skip("Skipping speech tests for PT1.9")
    if "tensorflow" in framework:
        pytest.skip(
            f"Zero-code deployments are currently not supported for TensorFlow Images - {ecr_image}"
        )
    instance_type = instance_type or "ml.m5.xlarge"
    invoke_sm_endpoint_helper_function(
        ecr_image=ecr_image,
        sagemaker_regions=sagemaker_regions,
        test_function=_test_speech_model,
        framework_version=framework_version,
        instance_type=instance_type,
        model_dir=model_dir,
        dump_logs_from_cloudwatch=dump_logs_from_cloudwatch,
    )


@pytest.mark.model("wav2vec2")
@pytest.mark.processor("gpu")
@pytest.mark.gpu_test
def test_speech_model_gpu(framework_version, ecr_image, instance_type, sagemaker_regions):
    framework, _ = get_framework_and_version_from_tag(ecr_image)
    if "pytorch" in framework and Version(framework_version) in SpecifierSet("==1.9.*"):
        pytest.skip("Skipping speech tests for PT1.9")
    if "tensorflow" in framework:
        pytest.skip(
            f"Zero-code deployments are currently not supported for TensorFlow Images - {ecr_image}"
        )
    instance_type = instance_type or "ml.p3.2xlarge"
    invoke_sm_endpoint_helper_function(
        ecr_image=ecr_image,
        sagemaker_regions=sagemaker_regions,
        test_function=_test_speech_model,
        framework_version=framework_version,
        instance_type=instance_type,
        model_dir=model_dir,
        dump_logs_from_cloudwatch=dump_logs_from_cloudwatch,
    )


def _test_speech_model(
    sagemaker_session, framework_version, image_uri, instance_type, model_dir, accelerator_type=None
):
    endpoint_name = sagemaker.utils.unique_name_from_base(
        "sagemaker-huggingface-serving-speech-model"
    )

    env = {
        "HF_MODEL_ID": "hf-internal-testing/tiny-random-wav2vec2",
        "HF_TASK": "automatic-speech-recognition",
    }

    hf_model = HuggingFaceModel(
        env=env,
        role="SageMakerRole",
        image_uri=image_uri,
        sagemaker_session=sagemaker_session,
        predictor_cls=Predictor,
    )

    with timeout_and_delete_endpoint(endpoint_name, sagemaker_session, minutes=30):
        predictor = hf_model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
        )

        predictor.serializer = DataSerializer(content_type="audio/wave")
        predictor.deserializer = JSONDeserializer()

        output = predictor.predict(audio_sample_file_path)

        assert "text" in output
