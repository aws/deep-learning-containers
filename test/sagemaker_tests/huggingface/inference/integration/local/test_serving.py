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

from ...integration import model_dir, ROLE, pt_model, tf_model
from ...utils import local_mode_utils


@contextmanager
def _predictor(model_dir, image, framework_version, sagemaker_local_session, instance_type):

    model_file = pt_model if "pytorch" in image else tf_model

    model = Model(
        model_data=f"file://{model_dir}/{model_file}",
        role=ROLE,
        image_uri=image,
        sagemaker_session=sagemaker_local_session,
        predictor_cls=Predictor,
    )
    with local_mode_utils.lock():
        try:
            predictor = model.deploy(1, instance_type)
            yield predictor
        finally:
            predictor.delete_endpoint()


def _assert_prediction(predictor):
    predictor.serializer = JSONSerializer()
    predictor.deserializer = JSONDeserializer()

    data = {
        "inputs": "Camera - You are awarded a SiPix Digital Camera! call 09061221066 fromm landline. Delivery within 28 days."
    }
    output = predictor.predict(data)

    assert "score" in output[0]


@pytest.mark.model("tiny-distilbert")
def test_serve_json(ecr_image, framework_version, sagemaker_local_session, instance_type):
    with _predictor(model_dir, ecr_image, framework_version, sagemaker_local_session, instance_type) as predictor:
        _assert_prediction(predictor)
