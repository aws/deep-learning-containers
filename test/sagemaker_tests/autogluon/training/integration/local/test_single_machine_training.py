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

import pytest

from ag_tools import AutoGluon
from .local_mode_utils import assert_output_files_exist
from .. import RESOURCE_PATH, MODEL_SUCCESS_FILES


@pytest.mark.integration("ag_local")
@pytest.mark.model("N/A")
def test_linear_regression(docker_image, sagemaker_local_session, local_instance_type, framework_version, tmpdir):

    ag = AutoGluon(
        entry_point=os.path.join(RESOURCE_PATH, 'scripts', 'train.py'),
        role='SageMakerRole', instance_count=1, instance_type=local_instance_type,
        sagemaker_session=sagemaker_local_session, image_uri=docker_image,
        framework_version=framework_version, output_path='file://{}'.format(tmpdir),
    )

    data_path = os.path.join(RESOURCE_PATH, 'data')
    s3_prefix = 'integ-test-data/autogluon'
    train_input = sagemaker_local_session.upload_data(path=os.path.join(data_path, 'training', 'train.csv'), key_prefix=s3_prefix)
    eval_input = sagemaker_local_session.upload_data(path=os.path.join(data_path, 'evaluation', 'eval.csv'), key_prefix=s3_prefix)
    config_input = sagemaker_local_session.upload_data(path=os.path.join(data_path, 'config', 'config.yaml'), key_prefix=s3_prefix)

    ag.fit({'config': config_input, 'train': train_input, 'test': eval_input})

    for directory, files in MODEL_SUCCESS_FILES.items():
        assert_output_files_exist(str(tmpdir), directory, files)
