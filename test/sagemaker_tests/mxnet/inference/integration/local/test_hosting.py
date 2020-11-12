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

from sagemaker.mxnet.model import MXNetModel
from sagemaker.deserializers import StringDeserializer

from ...integration.local import local_mode_utils
from ...integration import RESOURCE_PATH


HOSTING_RESOURCE_PATH = os.path.join(RESOURCE_PATH, 'dummy_hosting')
MODEL_PATH = os.path.join(HOSTING_RESOURCE_PATH, 'model.tar.gz')
SCRIPT_PATH = os.path.join(HOSTING_RESOURCE_PATH, 'code', 'dummy_hosting_module.py')


# The image should use the model_fn and transform_fn defined
# in the user-provided script when serving.
@pytest.mark.integration("hosting")
@pytest.mark.model("dummy_model")
def test_hosting(docker_image, sagemaker_local_session, local_instance_type, framework_version):
    model = MXNetModel(model_data="file://{}".format(MODEL_PATH),
                       image_uri=docker_image,
                       role='SageMakerRole',
                       entry_point=SCRIPT_PATH,
                       framework_version=framework_version,
                       sagemaker_session=sagemaker_local_session)

    with local_mode_utils.lock():
        try:
            predictor = model.deploy(1, local_instance_type, deserializer=StringDeserializer())

            input = 'some data'
            output = predictor.predict(input)
            assert '"'+input+'"' == output
        finally:
            predictor.delete_endpoint()
