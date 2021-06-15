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

from .resources.timeout import timeout_and_delete_endpoint
import time
import numpy as np


@pytest.mark.model("bert-base-uncased")
@pytest.mark.processor("cpu")
@pytest.mark.cpu_test
def test_sm_trained_model_cpu(sagemaker_session, framework_version, ecr_image, instance_type):
    instance_type = "ml.m5.xlarge"
    model = "s3://huggingface-benchmark-test/bert-base-uncased/model.tar.gz"
    task = "text-classification"
    latencies = \
        _test_sm_trained_model(sagemaker_session, framework_version, ecr_image, instance_type, model, task)
    assert np.average(latencies) <= 120


@pytest.mark.model("bert-base-uncased")
@pytest.mark.processor("gpu")
@pytest.mark.gpu_test
def test_sm_trained_model_gpu(sagemaker_session, framework_version, ecr_image, instance_type):
    instance_type = "ml.p2.xlarge"
    model = "s3://huggingface-benchmark-test/bert-base-uncased/model.tar.gz"
    task = "text-classification"
    latencies = \
        _test_sm_trained_model(sagemaker_session, framework_version, ecr_image, instance_type, model, task)
    assert np.average(latencies) <= 120


def _test_sm_trained_model(
    sagemaker_session, framework_version, ecr_image, instance_type, model, task, accelerator_type=None):
    endpoint_name = sagemaker.utils.unique_name_from_base("sagemaker-huggingface-serving-pt")

    hf_model = Model(
        model_data=model,
        role="SageMakerRole",
        image_uri=ecr_image,
        sagemaker_session=sagemaker_session,
        predictor_cls=Predictor,
        env={'HF_TASK': task}
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
        latencies = []
        for i in range(1020):
            start = time.time()
            output = predictor.predict(data)
            if i > 20:  # Warmup 20 iterations..
                latencies.append((time.time() - start) * 1000)

    return latencies