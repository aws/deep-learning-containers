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

from .timeout import timeout
from ..local.ag_tools import AutoGluon
from ...integration import RESOURCE_PATH


@pytest.mark.model("autogluon")
@pytest.mark.integration("smexperiments")
@pytest.mark.skip_test_in_region
def test_training(sagemaker_session, ecr_image, instance_type, framework_version):
    ag = AutoGluon(
        entry_point=os.path.join(RESOURCE_PATH, 'scripts', 'train_tab.py'),
        role='SageMakerRole',
        instance_count=1,
        instance_type=instance_type,
        sagemaker_session=sagemaker_session,
        image_uri=ecr_image,
        framework_version=framework_version,
    )

    ag = _disable_sm_profiler(sagemaker_session.boto_region_name, ag)

    with timeout(minutes=15):
        device = 'cpu'
        data_path = os.path.join(RESOURCE_PATH, 'data')
        s3_prefix = 'autogluon_sm/{}'.format(utils.sagemaker_timestamp())
        train_input = ag.sagemaker_session.upload_data(path=os.path.join(data_path, 'training', f'train.{device}.csv'), key_prefix=s3_prefix)
        eval_input = ag.sagemaker_session.upload_data(path=os.path.join(data_path, 'evaluation', f'eval.{device}.csv'), key_prefix=s3_prefix)
        config_input = ag.sagemaker_session.upload_data(path=os.path.join(data_path, 'config', f'config.{device}.yaml'), key_prefix=s3_prefix)

        job_name = utils.unique_name_from_base('test-autogluon-image')
        ag.fit({'config': config_input, 'train': train_input, 'test': eval_input}, job_name=job_name)



def _disable_sm_profiler(region, estimator):
    """Disable SMProfiler feature for China regions
    """

    if region in ('cn-north-1', 'cn-northwest-1'):
        estimator.disable_profiler = True
    return estimator
