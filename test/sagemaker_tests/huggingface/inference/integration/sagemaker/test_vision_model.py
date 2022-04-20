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
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.predictor import Predictor
from sagemaker.serializers import DataSerializer
from sagemaker.deserializers import JSONDeserializer

from ...integration import model_dir, dump_logs_from_cloudwatch, image_sample_file_path
from ...integration.sagemaker.timeout import timeout_and_delete_endpoint


@pytest.mark.model("vit")
@pytest.mark.processor("cpu")
@pytest.mark.cpu_test
def test_vision_model_cpu(sagemaker_session, framework_version, ecr_image, instance_type, region):
    instance_type = instance_type or "ml.m5.xlarge"
    try:
        _test_vision_model(sagemaker_session, framework_version, ecr_image, instance_type, model_dir)
    except Exception as e:
        dump_logs_from_cloudwatch(e, region)
        raise


@pytest.mark.model("vit")
@pytest.mark.processor("gpu")
@pytest.mark.gpu_test
def test_vision_model_gpu(sagemaker_session, framework_version, ecr_image, instance_type, region):
    instance_type = instance_type or "ml.p3.2xlarge"
    try:
        _test_vision_model(sagemaker_session, framework_version, ecr_image, instance_type, model_dir)
    except Exception as e:
        dump_logs_from_cloudwatch(e, region)
        raise


def _test_vision_model(
    sagemaker_session, framework_version, ecr_image, instance_type, model_dir, accelerator_type=None
):
    endpoint_name = sagemaker.utils.unique_name_from_base("sagemaker-huggingface-serving-vision-model")

    env = {
        "HF_MODEL_ID": "hf-internal-testing/tiny-random-vit",
        "HF_TASK": "image-classification",
    }

    hf_model = HuggingFaceModel(
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

        predictor.serializer = DataSerializer(content_type="image/png")
        predictor.deserializer = JSONDeserializer()

        output = predictor.predict(image_sample_file_path)

        assert "score" in output[0]
