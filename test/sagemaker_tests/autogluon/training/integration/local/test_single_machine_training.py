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

from .ag_tools import AutoGluon
from .local_mode_utils import assert_output_files_exist
from .. import RESOURCE_PATH


@pytest.mark.integration("ag_local")
@pytest.mark.processor("cpu")
@pytest.mark.model("autogluon")
def test_autogluon_local_cpu(docker_image, sagemaker_local_session, instance_type, framework_version, tmpdir):
    _test_autogluon_local('cpu', docker_image, sagemaker_local_session, instance_type, framework_version, tmpdir)


@pytest.mark.integration("ag_local")
@pytest.mark.processor("gpu")
@pytest.mark.model("autogluon")
@pytest.mark.skip_cpu
def test_autogluon_local_gpu(docker_image, sagemaker_local_session, instance_type, framework_version, tmpdir):
    _test_autogluon_local('gpu', docker_image, sagemaker_local_session, instance_type, framework_version, tmpdir)


@pytest.mark.integration("ag_local")
@pytest.mark.processor("gpu")
@pytest.mark.model("autogluon")
@pytest.mark.skip_cpu
def test_autogluon_local_vision_gpu(docker_image, sagemaker_local_session, instance_type, framework_version, tmpdir):
    ag = AutoGluon(
        entry_point=os.path.join(RESOURCE_PATH, 'scripts', 'train_cv.py'),
        role='SageMakerRole', instance_count=1, instance_type=instance_type,
        sagemaker_session=sagemaker_local_session, image_uri=docker_image,
        framework_version=framework_version, output_path='file://{}'.format(tmpdir),
    )

    data_path = os.path.join(RESOURCE_PATH, 'data')
    s3_prefix = 'integ-test-data/autogluon'
    config_input = sagemaker_local_session.upload_data(path=os.path.join(data_path, 'config', 'config.vision.gpu.yaml'), key_prefix=s3_prefix)
    ag.fit({'config': config_input})

    model_success_files = {
        'model': ['predictor.pkl'],
        'output': ['success'],
    }

    for directory, files in model_success_files.items():
        assert_output_files_exist(str(tmpdir), directory, files)


def _test_autogluon_local(device, docker_image, sagemaker_local_session, instance_type, framework_version, tmpdir):
    ag = AutoGluon(
        entry_point=os.path.join(RESOURCE_PATH, 'scripts', 'train.py'),
        role='SageMakerRole', instance_count=1, instance_type=instance_type,
        sagemaker_session=sagemaker_local_session, image_uri=docker_image,
        framework_version=framework_version, output_path='file://{}'.format(tmpdir),
    )

    data_path = os.path.join(RESOURCE_PATH, 'data')
    s3_prefix = 'integ-test-data/autogluon'
    train_input = sagemaker_local_session.upload_data(path=os.path.join(data_path, 'training', f'train.{device}.csv'), key_prefix=s3_prefix)
    eval_input = sagemaker_local_session.upload_data(path=os.path.join(data_path, 'evaluation', f'eval.{device}.csv'), key_prefix=s3_prefix)
    config_input = sagemaker_local_session.upload_data(path=os.path.join(data_path, 'config', f'config.{device}.yaml'), key_prefix=s3_prefix)

    ag.fit({'config': config_input, 'train': train_input, 'test': eval_input})

    model_success_files = {
        'model': ['learner.pkl', 'predictor.pkl'],
        'output': ['success'],
    }

    for directory, files in model_success_files.items():
        assert_output_files_exist(str(tmpdir), directory, files)
