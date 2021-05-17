# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from ...integration import RESOURCE_PATH
from .timeout import timeout
from . import invoke_mxnet_estimator

DATA_PATH = os.path.join(RESOURCE_PATH, 'mnist')
SCRIPT_PATH = os.path.join(DATA_PATH, 'mnist_gluon_basic_hook_demo.py')


@pytest.mark.integration("smdebug")
@pytest.mark.model("mnist")
@pytest.mark.skip_py2_containers
def test_training(sagemaker_session, n_virginia_sagemaker_session, ecr_image, n_virginia_ecr_image, instance_type, instance_count, framework_version, multi_region_support):
    hyperparameters = {'random_seed': True,
                       'num_steps': 50,
                       'smdebug_path': '/tmp/ml/output/tensors',
                       'epochs': 1}

    estimator_parameter = {
            'entry_point': SCRIPT_PATH,
            'role': 'SageMakerRole',
            'instance_count': instance_count,
            'instance_type': instance_type,
            'framework_version': framework_version,
            'hyperparameters': hyperparameters
        }
    job_name = utils.unique_name_from_base('test-mxnet-image')
    prefix = 'mxnet_mnist_gluon_basic_hook_demo/{}'.format(utils.sagemaker_timestamp())
    upload_s3_train_data_args = {
        'path': os.path.join(DATA_PATH, 'train'),
        'key_prefix': prefix + '/train'
    }
    upload_s3_test_data_args = {
        'path': os.path.join(DATA_PATH, 'test'),
        'key_prefix': prefix + '/test'
    }

    with timeout(minutes=15):
        invoke_mxnet_estimator(ecr_image, n_virginia_ecr_image, sagemaker_session, n_virginia_sagemaker_session, estimator_parameter, multi_region_support, job_name, upload_s3_train_data_args, upload_s3_test_data_args)

