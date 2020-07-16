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
from sagemaker.mxnet import MXNet

from ...integration.local import local_mode_utils
from ...integration import MODEL_SUCCESS_FILES, RESOURCE_PATH

MNIST_PATH = os.path.join(RESOURCE_PATH, 'mnist')
SCRIPT_PATH = os.path.join(MNIST_PATH, 'mnist.py')

TRAIN_INPUT = 'file://{}'.format(os.path.join(MNIST_PATH, 'train'))
TEST_INPUT = 'file://{}'.format(os.path.join(MNIST_PATH, 'test'))


@pytest.mark.model("mnist")
def test_single_machine(docker_image, sagemaker_local_session, local_instance_type,
                        framework_version, tmpdir):
    mx = MXNet(entry_point=SCRIPT_PATH, role='SageMakerRole', train_instance_count=1,
               train_instance_type=local_instance_type, sagemaker_session=sagemaker_local_session,
               image_name=docker_image, framework_version=framework_version,
               output_path='file://{}'.format(tmpdir))

    _train_and_assert_success(mx, str(tmpdir))


@pytest.mark.model("mnist")
@pytest.mark.multinode("multinode")
def test_distributed(docker_image, sagemaker_local_session, framework_version, processor, tmpdir):
    if processor == 'gpu':
        pytest.skip('Local Mode does not support distributed training on GPU.')

    mx = MXNet(entry_point=SCRIPT_PATH, role='SageMakerRole', train_instance_count=2,
               train_instance_type='local', sagemaker_session=sagemaker_local_session,
               image_name=docker_image, framework_version=framework_version,
               output_path='file://{}'.format(tmpdir),
               hyperparameters={'sagemaker_parameter_server_enabled': True})
    _train_and_assert_success(mx, str(tmpdir))


def _train_and_assert_success(estimator, output_path):
    estimator.fit({'train': TRAIN_INPUT, 'test': TEST_INPUT})

    for directory, files in MODEL_SUCCESS_FILES.items():
        local_mode_utils.assert_output_files_exist(output_path, directory, files)
