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
from sagemaker.pytorch import PyTorchModel

from ..... import invoke_sm_helper_function
from ...integration import RESOURCE_PATH, model_cpu_dir, mnist_gpu_script
from ...integration.sagemaker import timeout

@pytest.mark.model("mnist")
@pytest.mark.processor("gpu")
@pytest.mark.gpu_test
def test_sagemaker_endpoint_gpu(ecr_image, sagemaker_regions, instance_type, framework_version):
    instance_type = instance_type or 'ml.p2.xlarge'
    invoke_sm_helper_function(ecr_image, sagemaker_regions, _test_sagemaker_endpoint_function, instance_type, framework_version)

@pytest.mark.model("mnist")
@pytest.mark.processor("cpu")
@pytest.mark.cpu_test
def test_sagemaker_endpoint_cpu(ecr_image, sagemaker_regions, instance_type, framework_version):
    instance_type = instance_type or 'ml.c4.xlarge'
    invoke_sm_helper_function(ecr_image, sagemaker_regions, _test_sagemaker_endpoint_function, instance_type, framework_version)

def _test_sagemaker_endpoint_function(ecr_image, sagemaker_session, instance_type, framework_version):
    prefix = 'sagemaker-pytorch-serving/models'
    model_dir = os.path.join(model_cpu_dir, "model_mnist.tar.gz")
    model_data = sagemaker_session.upload_data(path=model_dir, key_prefix=prefix)
    model = PyTorchModel(
        model_data=model_data,
        role="SageMakerRole",
        entry_point=mnist_gpu_script,
        framework_version=framework_version,
        image_uri=ecr_image,
        sagemaker_session=sagemaker_session,
    )

    endpoint_name = utils.unique_name_from_base("sagemaker-pytorch-serving")
    with timeout.timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        predictor = model.deploy(initial_instance_count=1, instance_type=instance_type, endpoint_name=endpoint_name)
