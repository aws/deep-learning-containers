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
from sagemaker.mxnet import MXNetModel

from ...integration import EI_SUPPORTED_REGIONS, RESOURCE_PATH
from ...integration.sagemaker.timeout import timeout_and_delete_endpoint_by_name

DEFAULT_HANDLER_PATH = os.path.join(RESOURCE_PATH, 'default_handlers')
MODEL_PATH = os.path.join(DEFAULT_HANDLER_PATH, 'model.tar.gz')
SCRIPT_PATH = os.path.join(DEFAULT_HANDLER_PATH, 'model', 'code', 'empty_module.py')


@pytest.fixture(autouse=True)
def skip_if_no_accelerator(accelerator_type):
    if accelerator_type is None:
        pytest.skip('Skipping because accelerator type was not provided')


@pytest.fixture(autouse=True)
def skip_if_non_supported_ei_region(region):
    if region not in EI_SUPPORTED_REGIONS:
        pytest.skip('EI is not supported in {}'.format(region))


@pytest.mark.processor("eia")
@pytest.mark.integration("elastic_inference")
@pytest.mark.skip_if_non_supported_ei_region()
@pytest.mark.skip_if_no_accelerator()
def test_elastic_inference(ecr_image, sagemaker_session, instance_type, accelerator_type, framework_version):
    endpoint_name = utils.unique_name_from_base('test-mxnet-ei')

    with timeout_and_delete_endpoint_by_name(endpoint_name=endpoint_name,
                                             sagemaker_session=sagemaker_session,
                                             minutes=20):
        prefix = 'mxnet-serving/default-handlers'
        model_data = sagemaker_session.upload_data(path=MODEL_PATH, key_prefix=prefix)
        model = MXNetModel(model_data=model_data,
                           entry_point=SCRIPT_PATH,
                           role='SageMakerRole',
                           image=ecr_image,
                           framework_version=framework_version,
                           sagemaker_session=sagemaker_session)

        predictor = model.deploy(initial_instance_count=1,
                                 instance_type=instance_type,
                                 accelerator_type=accelerator_type,
                                 endpoint_name=endpoint_name)

        output = predictor.predict([[1, 2]])
        assert [[4.9999918937683105]] == output
