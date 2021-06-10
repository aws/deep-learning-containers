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

from ...integration import model_dir, pt_model, tf_model
from ...integration.sagemaker.timeout import timeout_and_delete_endpoint

# TODO: remove
os.environ["AWS_PROFILE"] = "hf-sm"  # setting aws profile for our boto3 client
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"  # current DLCs are only in us-east-1 available


@pytest.mark.model("tiny-distilbert")
@pytest.mark.processor("cpu")
@pytest.mark.cpu_test
def test_sm_trained_model_cpu(sagemaker_session, framework_version, ecr_image, instance_type):
    instance_type = instance_type or "ml.m5.xlarge"
    _test_sm_trained_model(sagemaker_session, framework_version, ecr_image, instance_type, model_dir)


@pytest.mark.model("tiny-distilbert")
@pytest.mark.processor("gpu")
@pytest.mark.gpu_test
def test_sm_trained_model_gpu(sagemaker_session, framework_version, ecr_image, instance_type):
    instance_type = instance_type or "ml.p2.xlarge"
    _test_sm_trained_model(sagemaker_session, framework_version, ecr_image, instance_type, model_dir)


def _test_sm_trained_model(
    sagemaker_session, framework_version, ecr_image, instance_type, model_dir, accelerator_type=None
):
    endpoint_name = sagemaker.utils.unique_name_from_base("sagemaker-huggingface-serving-trained-model")

    model_data = sagemaker_session.upload_data(
        path=model_dir,
        key_prefix="sagemaker-huggingface-serving-trained-model/models",
    )

    model_file = pt_model if "pytorch" in ecr_image else tf_model

    hf_model = Model(
        model_data=f"{model_data}/{model_file}",
        role="sagemaker_execution_role",  # TODO: what is the correct role name for CI-pipeline, is it "SageMakerRole"
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
