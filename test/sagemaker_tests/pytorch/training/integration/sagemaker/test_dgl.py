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
from sagemaker.pytorch import PyTorch

from ...integration import resources_path, DEFAULT_TIMEOUT
from ...integration.sagemaker.timeout import timeout

from test.test_utils import get_framework_and_version_from_tag, get_cuda_version_from_tag
from packaging.version import Version
from .... import invoke_pytorch_helper_function


DGL_DATA_PATH = os.path.join(resources_path, 'dgl-gcn')
DGL_SCRIPT_PATH = os.path.join(DGL_DATA_PATH, 'gcn.py')


@pytest.mark.integration("dgl")
@pytest.mark.processor("cpu")
@pytest.mark.model("gcn")
@pytest.mark.skip_gpu
@pytest.mark.skip_py2_containers
def test_dgl_gcn_training_cpu(sagemaker_session, n_virginia_sagemaker_session, ecr_image, n_virginia_ecr_image, instance_type, multi_region_support):
    instance_type = instance_type or 'ml.c4.xlarge'
    function_args = {
            'instance_type': instance_type,
        }
    invoke_pytorch_helper_function(ecr_image, n_virginia_ecr_image, sagemaker_session, n_virginia_sagemaker_session, multi_region_support, _test_dgl_training, function_args)


@pytest.mark.integration("dgl")
@pytest.mark.processor("gpu")
@pytest.mark.model("gcn")
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
def test_dgl_gcn_training_gpu(sagemaker_session, n_virginia_sagemaker_session, ecr_image, n_virginia_ecr_image, instance_type, multi_region_support):

    _, image_framework_version = get_framework_and_version_from_tag(ecr_image)
    image_cuda_version = get_cuda_version_from_tag(ecr_image)
    if Version(image_framework_version) == Version("1.6") and image_cuda_version == "cu110":
        pytest.skip("DGL does not support CUDA 11 for PyTorch 1.6")

    instance_type = instance_type or 'ml.p2.xlarge'
    function_args = {
            'instance_type': instance_type,
        }

    invoke_pytorch_helper_function(ecr_image, n_virginia_ecr_image, sagemaker_session, n_virginia_sagemaker_session, multi_region_support, _test_dgl_training, function_args)


def _test_dgl_training(ecr_image, sagemaker_session, instance_type):
    dgl = PyTorch(
        entry_point=DGL_SCRIPT_PATH,
        role='SageMakerRole',
        instance_count=1,
        instance_type=instance_type,
        sagemaker_session=sagemaker_session,
        image_uri=ecr_image,
    )
    with timeout(minutes=DEFAULT_TIMEOUT):
        job_name = utils.unique_name_from_base('test-pytorch-dgl-image')
        dgl.fit(job_name=job_name)
