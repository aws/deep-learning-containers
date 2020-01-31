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

from mock import patch


@patch('sagemaker_pytorch_serving_container.default_inference_handler.DefaultPytorchInferenceHandler')
@patch('sagemaker_inference.transformer.Transformer')
def test_hosting_start(Transformer, DefaultPytorchInferenceHandler):
    from sagemaker_pytorch_serving_container import handler_service

    handler_service.HandlerService()

    Transformer.assert_called_with(default_inference_handler=DefaultPytorchInferenceHandler())
