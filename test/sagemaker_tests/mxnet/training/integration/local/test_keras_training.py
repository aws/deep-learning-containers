#  Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  A copy of the License is located at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  or in the "license" file accompanying this file. This file is distributed
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#  express or implied. See the License for the specific language governing
#  permissions and limitations under the License.
from __future__ import absolute_import

import os

from sagemaker.mxnet import MXNet

from test.sagemaker_tests.mxnet.training.integration import local_mode_utils
from test.sagemaker_tests.mxnet.training.integration import MODEL_SUCCESS_FILES, RESOURCE_PATH


def test_keras_training(docker_image, sagemaker_local_session, local_instance_type,
                        framework_version, tmpdir):
    keras_path = os.path.join(RESOURCE_PATH, 'keras')
    script_path = os.path.join(keras_path, 'keras_mnist.py')

    mx = MXNet(entry_point=script_path, role='SageMakerRole', train_instance_count=1,
               train_instance_type=local_instance_type, sagemaker_session=sagemaker_local_session,
               image_name=docker_image, framework_version=framework_version,
               output_path='file://{}'.format(tmpdir))

    train = 'file://{}'.format(os.path.join(keras_path, 'data'))
    mx.fit({'train': train})

    for directory, files in MODEL_SUCCESS_FILES.items():
        local_mode_utils.assert_output_files_exist(str(tmpdir), directory, files)
