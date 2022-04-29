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
from contextlib import contextmanager

import pandas as pd
import pytest
from sagemaker.deserializers import JSONDeserializer
from sagemaker.mxnet import MXNetModel
from sagemaker.serializers import CSVSerializer

from .. import RESOURCE_PATH
from ...integration import ROLE
from ...utils import local_mode_utils


@contextmanager
def _predictor(image, framework_version, sagemaker_local_session, instance_type):
    model_dir = os.path.join(RESOURCE_PATH, 'model')
    source_dir = os.path.join(RESOURCE_PATH, 'scripts')

    ag_framework_version = '0.3.1' if framework_version == '0.3.2' else framework_version
    model = MXNetModel(
        model_data=f"file://{model_dir}/model_{ag_framework_version}.tar.gz",
        role=ROLE,
        image_uri=image,
        sagemaker_session=sagemaker_local_session,
        source_dir=source_dir,
        entry_point="tabular_serve.py",
        framework_version="1.9.0"
    )
    with local_mode_utils.lock():
        try:
            predictor = model.deploy(1, instance_type)
            yield predictor
        finally:
            predictor.delete_endpoint()


def _assert_prediction(predictor):
    predictor.serializer = CSVSerializer()
    predictor.deserializer = JSONDeserializer()

    data_path = os.path.join(RESOURCE_PATH, 'data')
    data = pd.read_csv(f'{data_path}/data.csv')
    assert 3 == len(data)

    preds = predictor.predict(data.values)
    assert preds == [' <=50K', ' <=50K', ' <=50K']


@pytest.mark.integration("ag_local")
@pytest.mark.processor("cpu")
@pytest.mark.model("autogluon")
def test_serve_json(docker_image, framework_version, sagemaker_local_session, instance_type):
    with _predictor(docker_image, framework_version, sagemaker_local_session, instance_type) as predictor:
        _assert_prediction(predictor)
