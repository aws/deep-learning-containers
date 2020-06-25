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

import numpy
from sagemaker.mxnet import MXNetModel

from ...integration.local import local_mode_utils
from ...integration import RESOURCE_PATH

ONNX_PATH = os.path.join(RESOURCE_PATH, 'onnx')
MODEL_PATH = os.path.join(ONNX_PATH, 'onnx_model')
SCRIPT_PATH = os.path.join(MODEL_PATH, 'code', 'onnx_import.py')


def test_onnx_import(docker_image, sagemaker_local_session, local_instance_type):
    model = MXNetModel('file://{}'.format(MODEL_PATH),
                       'SageMakerRole',
                       SCRIPT_PATH,
                       image=docker_image,
                       sagemaker_session=sagemaker_local_session)

    input = numpy.zeros(shape=(1, 1, 28, 28))

    with local_mode_utils.lock():
        try:
            predictor = model.deploy(1, local_instance_type)
            output = predictor.predict(input)
        finally:
            predictor.delete_endpoint()

    # Check that there is a probability for each possible class in the prediction
    assert len(output[0]) == 10
