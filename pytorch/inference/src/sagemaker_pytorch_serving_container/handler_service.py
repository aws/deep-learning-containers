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

from sagemaker_inference.default_handler_service import DefaultHandlerService
from sagemaker_inference.transformer import Transformer
from sagemaker_pytorch_serving_container.default_inference_handler import \
    DefaultPytorchInferenceHandler


class HandlerService(DefaultHandlerService):
    """Handler service that is executed by the model server.

    Determines specific default inference handlers to use based on the type MXNet model being used.

    This class extends ``DefaultHandlerService``, which define the following:
        - The ``handle`` method is invoked for all incoming inference requests to the model server.
        - The ``initialize`` method is invoked at model server start up.

    Based on: https://github.com/awslabs/mxnet-model-server/blob/master/docs/custom_service.md

    """
    def __init__(self):
        transformer = Transformer(default_inference_handler=DefaultPytorchInferenceHandler())
        super(HandlerService, self).__init__(transformer=transformer)
