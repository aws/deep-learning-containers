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
from ... import MXNetWrapper as MXNet

from ...integration import RESOURCE_PATH
from .timeout import timeout

DGL_DATA_PATH = os.path.join(RESOURCE_PATH, 'dgl_gcn')
DGL_SCRIPT_PATH = os.path.join(DGL_DATA_PATH, 'gcn.py')


@pytest.mark.skip(reason="Skip until DGL with cuda 11.0 is available")
@pytest.mark.integration("dgl")
@pytest.mark.model("gcn")
@pytest.mark.skip_py2_containers
def test_dgl_training(sagemaker_regions, ecr_image, instance_type, framework_version):

    dgl = MXNet(entry_point=DGL_SCRIPT_PATH,
                role='SageMakerRole',
                instance_count=1,
                instance_type=instance_type,
                sagemaker_regions=sagemaker_regions,
                image_uri=ecr_image,
                framework_version=framework_version)

    with timeout(minutes=15):
        job_name = utils.unique_name_from_base('test-dgl-image')
        dgl.fit(job_name=job_name)
