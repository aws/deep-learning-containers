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

from sagemaker import utils
from sagemaker.mxnet.model import MXNetModel

from test.integration import RESOURCE_PATH
from test.integration.sagemaker import timeout

DEFAULT_HANDLER_PATH = os.path.join(RESOURCE_PATH, 'default_handlers')
MODEL_PATH = os.path.join(DEFAULT_HANDLER_PATH, 'model.tar.gz')
SCRIPT_PATH = os.path.join(DEFAULT_HANDLER_PATH, 'model', 'code', 'empty_module.py')


def test_hosting(sagemaker_session, ecr_image, instance_type, framework_version):
    prefix = 'mxnet-serving/default-handlers'
    model_data = sagemaker_session.upload_data(path=MODEL_PATH, key_prefix=prefix)
    model = MXNetModel(model_data,
                       'SageMakerRole',
                       SCRIPT_PATH,
                       image=ecr_image,
                       framework_version=framework_version,
                       sagemaker_session=sagemaker_session)

    endpoint_name = utils.unique_name_from_base('test-mxnet-serving')
    with timeout.timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        predictor = model.deploy(1, instance_type, endpoint_name=endpoint_name)

        output = predictor.predict([[1, 2]])
        assert [[4.9999918937683105]] == output
