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
import numpy as np
import json

from ...integration import RESOURCE_PATH
from ...integration.sagemaker import timeout

DEFAULT_HANDLER_PATH = os.path.join(RESOURCE_PATH, 'neuron_handlers')
MODEL_PATH = os.path.join(DEFAULT_HANDLER_PATH, 'model-neuron.tar.gz')
SCRIPT_PATH = os.path.join(DEFAULT_HANDLER_PATH, 'model', 'code', 'mnist-neuron.py')
INPUT_PATH = os.path.join(DEFAULT_HANDLER_PATH, 'model', 'input.npy')
OUTPUT_PATH = os.path.join(DEFAULT_HANDLER_PATH, 'model', 'output.json')

@pytest.fixture(autouse=True)
def skip_if_no_neuron(ecr_image, instance_type):
    if 'neuron' not in ecr_image:
        pytest.skip('Skipping neuron test for non neuron images')
    if 'inf1' not in instance_type:
        pytest.skip('Skipping neuron test for non inf1 instances')


@pytest.mark.integration("neuron-hosting")
@pytest.mark.model("mnist")
@pytest.mark.skip_py2_containers
@pytest.mark.skip_eia_containers
@pytest.mark.skip_if_no_neuron()
def test_neuron_hosting(sagemaker_session, ecr_image, instance_type, framework_version):
    prefix = 'mxnet-serving/neuron-handlers'
    model_data = sagemaker_session.upload_data(path=MODEL_PATH, key_prefix=prefix)
    model = MXNetModel(model_data,
                       'SageMakerRole',
                       SCRIPT_PATH,
                       framework_version=framework_version,
                       image_uri=ecr_image,
                       sagemaker_session=sagemaker_session)

    endpoint_name = utils.unique_name_from_base('test-mxnet-serving')
    with timeout.timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        predictor = model.deploy(1, instance_type, endpoint_name=endpoint_name)

        numpy_ndarray = np.load(INPUT_PATH)
        output = predictor.predict(data=numpy_ndarray)
        with open(OUTPUT_PATH) as outputfile:
            expected_output = json.load(outputfile)
        
        assert expected_output == output
        print(output)
        
