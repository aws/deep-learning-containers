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
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

from ...integration import model_dir, pt_model, tf_model, dump_logs_from_cloudwatch
from ...integration.sagemaker.timeout import timeout_and_delete_endpoint
from ..... import invoke_sm_endpoint_helper_function


@pytest.mark.model("tiny-distilbert")
@pytest.mark.processor("cpu")
@pytest.mark.cpu_test
@pytest.mark.team("sagemaker-1p-algorithms")
def test_sm_trained_model_cpu(framework_version, ecr_image, instance_type, sagemaker_regions):
    instance_type = instance_type or "ml.m5.xlarge"
    invoke_sm_endpoint_helper_function(
        ecr_image=ecr_image,
        sagemaker_regions=sagemaker_regions,
        test_function=_test_sm_trained_model,
        framework_version=framework_version,
        instance_type=instance_type,
        model_dir=model_dir,
        dump_logs_from_cloudwatch=dump_logs_from_cloudwatch,
    )


@pytest.mark.model("tiny-distilbert")
@pytest.mark.processor("gpu")
@pytest.mark.gpu_test
@pytest.mark.team("sagemaker-1p-algorithms")
def test_sm_trained_model_gpu(framework_version, ecr_image, instance_type, sagemaker_regions):
    instance_type = instance_type or "ml.p3.2xlarge"
    invoke_sm_endpoint_helper_function(
        ecr_image=ecr_image,
        sagemaker_regions=sagemaker_regions,
        test_function=_test_sm_trained_model,
        framework_version=framework_version,
        instance_type=instance_type,
        model_dir=model_dir,
        dump_logs_from_cloudwatch=dump_logs_from_cloudwatch,
    )


def _test_sm_trained_model(
    sagemaker_session,
    framework_version,
    image_uri,
    instance_type,
    model_dir,
    accelerator_type=None,
    **kwargs,
):
    endpoint_name = sagemaker.utils.unique_name_from_base(
        "sagemaker-huggingface-serving-trained-model"
    )

    model_data = sagemaker_session.upload_data(
        path=model_dir,
        key_prefix="sagemaker-huggingface-serving-trained-model/models",
    )

    model_file = pt_model if "pytorch" in image_uri else tf_model

    hf_model = Model(
        model_data=f"{model_data}/{model_file}",
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

        data = {
            "inputs": "Camera - You are awarded a SiPix Digital Camera! call 09061221066 fromm landline. Delivery within 28 days."
        }
        predictor.serializer = JSONSerializer()
        predictor.deserializer = JSONDeserializer()

        output = predictor.predict(data)

        assert "score" in output[0]
