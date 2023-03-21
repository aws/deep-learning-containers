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
from sagemaker.mxnet.model import MXNetModel

from ..... import invoke_sm_helper_function
from ...integration import RESOURCE_PATH
from ...integration.sagemaker import timeout

DEFAULT_HANDLER_PATH = os.path.join(RESOURCE_PATH, 'default_handlers')
MODEL_PATH = os.path.join(DEFAULT_HANDLER_PATH, 'model.tar.gz')
SCRIPT_PATH = os.path.join(DEFAULT_HANDLER_PATH, 'model', 'code', 'empty_module.py')


@pytest.mark.model("linear_regression")
@pytest.mark.processor("gpu")
@pytest.mark.gpu_test
def test_sagemaker_endpoint_gpu(ecr_image, sagemaker_regions, instance_type, framework_version, skip_neuron_containers):
    instance_type = instance_type or 'ml.p2.xlarge'
    invoke_sm_helper_function(ecr_image, sagemaker_regions, _test_sagemaker_endpoint_function, instance_type, framework_version)


@pytest.mark.model("linear_regression")
@pytest.mark.processor("cpu")
@pytest.mark.cpu_test
def test_sagemaker_endpoint_cpu(ecr_image, sagemaker_regions, instance_type, framework_version, skip_neuron_containers):
    instance_type = instance_type or 'ml.c4.xlarge'
    invoke_sm_helper_function(ecr_image, sagemaker_regions, _test_sagemaker_endpoint_function, instance_type, framework_version)


def _test_sagemaker_endpoint_function(ecr_image, sagemaker_session, instance_type, framework_version):
    prefix = 'mxnet-serving/default-handlers'
    model_data = sagemaker_session.upload_data(path=MODEL_PATH, key_prefix=prefix)
    model = MXNetModel(model_data,
                       'SageMakerRole',
                       SCRIPT_PATH,
                       framework_version=framework_version,
                       image_uri=ecr_image,
                       sagemaker_session=sagemaker_session)

    endpoint_name = utils.unique_name_from_base('test-mxnet-serving')
    with timeout.timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        predictor = model.deploy(initial_instance_count=1, instance_type=instance_type, endpoint_name=endpoint_name)

# --docker-base-name=beta-mxnet-inference --region=us-west-2 --framework-version=1.9.0 --py-version=38 --processor=gpu --aws-id=669063966089 --tag=1.9.0-gpu-py38-cu112-ubuntu20.04-sagemaker-2023-02-24-17-16-59 --sagemaker-regions=us-west-2