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
import sagemaker
from sagemaker.deserializers import JSONDeserializer
from sagemaker.exceptions import UnexpectedStatusException
from sagemaker.mxnet import MXNetModel
from sagemaker.serializers import CSVSerializer

from test.sagemaker_tests.autogluon.training.integration import dump_logs_from_cloudwatch
from .timeout import timeout_and_delete_endpoint
from .. import RESOURCE_PATH


@contextmanager
def _test_sm_trained_model(sagemaker_session, ecr_image, instance_type, framework_version):
    model_dir = os.path.join(RESOURCE_PATH, 'model')
    source_dir = os.path.join(RESOURCE_PATH, 'scripts')

    endpoint_name = sagemaker.utils.unique_name_from_base("sagemaker-autogluon-serving-trained-model")
    ag_framework_version = '0.3.1' if framework_version == '0.3.2' else framework_version
    model_data = sagemaker_session.upload_data(path=os.path.join(model_dir, f'model_{ag_framework_version}.tar.gz'), key_prefix='sagemaker-autogluon-serving-trained-model/models')

    model = MXNetModel(
        model_data=model_data,
        role='SageMakerRole',
        image_uri=ecr_image,
        sagemaker_session=sagemaker_session,
        source_dir=source_dir,
        entry_point="tabular_serve.py",
        framework_version="1.8.0"
    )

    with timeout_and_delete_endpoint(endpoint_name, sagemaker_session, minutes=30):
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
        )
        predictor.serializer = CSVSerializer()
        predictor.deserializer = JSONDeserializer()

        data_path = os.path.join(RESOURCE_PATH, 'data')
        data = pd.read_csv(f'{data_path}/data.csv')
        assert 3 == len(data)

        preds = predictor.predict(data.values)
        assert preds == [' <=50K', ' <=50K', ' <=50K']


@pytest.mark.integration("smexperiments")
@pytest.mark.processor("cpu")
@pytest.mark.model("autogluon")
@pytest.mark.cpu_test
def test_sm_trained_model_cpu(sagemaker_session, framework_version, ecr_image, instance_type):
    instance_type = instance_type or "ml.m5.xlarge"
    try:
        _test_sm_trained_model(sagemaker_session, ecr_image, instance_type, framework_version)
    except UnexpectedStatusException as e:
        dump_logs_from_cloudwatch(e)
        raise
