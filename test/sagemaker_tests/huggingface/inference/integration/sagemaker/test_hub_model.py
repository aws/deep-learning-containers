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

import os

import pytest
import sagemaker
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.exceptions import UnexpectedStatusException

from ...integration import model_dir, get_logs_from_cloudwatch
from ...integration.sagemaker.timeout import timeout_and_delete_endpoint


@pytest.mark.model("tiny-distilbert")
@pytest.mark.processor("cpu")
@pytest.mark.cpu_test
def test_sm_trained_model_cpu(sagemaker_session, framework_version, ecr_image, instance_type):
    instance_type = instance_type or "ml.m5.xlarge"
    try:
        _test_hub_model(sagemaker_session, framework_version, ecr_image, instance_type, model_dir)
    except UnexpectedStatusException as e:
        get_logs_from_cloudwatch(e)
        raise


@pytest.mark.model("tiny-distilbert")
@pytest.mark.processor("gpu")
@pytest.mark.gpu_test
def test_sm_trained_model_gpu(sagemaker_session, framework_version, ecr_image, instance_type):
    instance_type = instance_type or "ml.p2.xlarge"
    try:
        _test_hub_model(sagemaker_session, framework_version, ecr_image, instance_type, model_dir)
    except UnexpectedStatusException as e:
        get_logs_from_cloudwatch(e)
        raise


def _test_hub_model(sagemaker_session, framework_version, ecr_image, instance_type, model_dir, accelerator_type=None):
    endpoint_name = sagemaker.utils.unique_name_from_base("sagemaker-huggingface-serving-hub-model")

    env = {
        "HF_MODEL_ID": "sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english",
        "HF_TASK": "text-classification",
    }

    hf_model = Model(
        env=env,
        role="SageMakerRole",
        image_uri=ecr_image,
        sagemaker_session=sagemaker_session,
        predictor_cls=Predictor,
    )

    with timeout_and_delete_endpoint(endpoint_name, sagemaker_session, minutes=30):
        predictor = hf_model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
        )

        data = {
            "inputs": "Camera - You are awarded a SiPix Digital Camera! call 09061221066 fromm landline. Delivery within 28 days."
        }
        predictor.serializer = JSONSerializer()
        predictor.deserializer = JSONDeserializer()

        output = predictor.predict(data)

        assert "score" in output[0]
