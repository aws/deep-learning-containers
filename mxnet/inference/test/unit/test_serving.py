# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import os

from mock import patch

from sagemaker_mxnet_serving_container.serving import _update_mxnet_env_vars, DEFAULT_ENV_VARS, HANDLER_SERVICE, main


@patch('sagemaker_inference.model_server.start_model_server')
@patch('sagemaker_mxnet_serving_container.serving._update_mxnet_env_vars')
def test_main(update_mxnet_env_vars, model_server):
    main()

    update_mxnet_env_vars.assert_called_once()
    model_server.assert_called_once_with(handler_service=HANDLER_SERVICE)


@patch.dict(os.environ, dict(), clear=True)
def test_update_env_vars():
    assert bool(os.environ) is False

    _update_mxnet_env_vars()

    assert DEFAULT_ENV_VARS == os.environ
