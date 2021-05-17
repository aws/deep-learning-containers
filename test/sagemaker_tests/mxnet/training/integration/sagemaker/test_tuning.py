# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
from __future__ import absolute_import

import os

import pytest

from sagemaker import utils
from sagemaker.mxnet.estimator import MXNet
from sagemaker.tuner import ContinuousParameter, HyperparameterTuner

from ...integration import RESOURCE_PATH
from .timeout import timeout
from . import invoke_mxnet_estimator

DATA_PATH = os.path.join(RESOURCE_PATH, 'mnist')
SCRIPT_PATH = os.path.join(DATA_PATH, 'mnist.py')


@pytest.mark.integration("hpo")
@pytest.mark.model("mnist")
def test_tuning(sagemaker_session, n_virginia_sagemaker_session, ecr_image, n_virginia_ecr_image, instance_type, framework_version, multi_region_support):
    
    estimator_parameter = {
            'entry_point': SCRIPT_PATH,
            'role': 'SageMakerRole',
            'instance_count': 1,
            'instance_type': instance_type,
            'framework_version': framework_version,
            'hyperparameters': {'epochs': 1}
        }
    prefix = 'mxnet_mnist/{}'.format(utils.sagemaker_timestamp())
    job_name = utils.unique_name_from_base('test-mxnet-image', max_length=32)
    upload_s3_train_data_args = {
        'path': os.path.join(DATA_PATH, 'train'),
        'key_prefix': prefix + '/train'
    }
    upload_s3_test_data_args = {
        'path': os.path.join(DATA_PATH, 'test'),
        'key_prefix': prefix + '/test'
    }
    hyperparameter_args = {
        'objective_metric_name': 'Validation-accuracy',
        'hyperparameter_ranges': {'learning-rate': ContinuousParameter(0.01, 0.2)},
        'metric_definitions': [{'Name': 'Validation-accuracy', 'Regex': 'Validation-accuracy=([0-9\\.]+)'}],
        'max_jobs': 2,
        'max_parallel_jobs': 2
    }

    with timeout(minutes=20):
        tuner, _ = invoke_mxnet_estimator(ecr_image, n_virginia_ecr_image, sagemaker_session, n_virginia_sagemaker_session, estimator_parameter, multi_region_support, job_name, upload_s3_train_data_args=upload_s3_train_data_args, upload_s3_test_data_args=upload_s3_test_data_args, hyperparameter_args=hyperparameter_args)
        tuner.wait()


