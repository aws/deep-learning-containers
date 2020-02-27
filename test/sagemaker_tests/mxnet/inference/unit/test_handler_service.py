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

from mock import Mock, patch
import mxnet as mx
import pytest
from sagemaker_inference import environment
from sagemaker_inference.default_inference_handler import DefaultInferenceHandler
from sagemaker_inference.transformer import Transformer

from sagemaker_mxnet_serving_container.default_inference_handler import DefaultGluonBlockInferenceHandler
from sagemaker_mxnet_serving_container.handler_service import HandlerService
from sagemaker_mxnet_serving_container.mxnet_module_transformer import MXNetModuleTransformer

MODULE_NAME = 'module_name'


@patch('sagemaker_mxnet_serving_container.handler_service.HandlerService._user_module_transformer')
def test_handler_service(user_module_transformer):
    service = HandlerService()

    assert service._service == user_module_transformer()


class UserModuleTransformFn:
    def __init__(self):
        self.transform_fn = Mock()


@patch('sagemaker_inference.environment.Environment')
@patch('importlib.import_module', return_value=UserModuleTransformFn())
def test_user_module_transform_fn(import_module, env):
    env.return_value.module_name = MODULE_NAME
    transformer = HandlerService._user_module_transformer()

    import_module.assert_called_once_with(MODULE_NAME)
    assert isinstance(transformer._default_inference_handler, DefaultInferenceHandler)
    assert isinstance(transformer, Transformer)


class UserModuleModelFn:
    def __init__(self):
        self.model_fn = Mock()


@patch('sagemaker_inference.environment.Environment')
@patch('importlib.import_module', return_value=UserModuleModelFn())
def test_user_module_mxnet_module_transformer(import_module, env):
    env.return_value.module_name = MODULE_NAME
    import_module.return_value.model_fn.return_value = mx.module.BaseModule()

    transformer = HandlerService._user_module_transformer()

    import_module.assert_called_once_with(MODULE_NAME)
    assert isinstance(transformer, MXNetModuleTransformer)


@patch('sagemaker_inference.environment.Environment')
@patch('sagemaker_mxnet_serving_container.default_inference_handler.DefaultMXNetInferenceHandler.default_model_fn')
@patch('importlib.import_module', return_value=object())
def test_default_inference_handler_mxnet_gluon_transformer(import_module, model_fn, env):
    env.return_value.module_name = MODULE_NAME
    model_fn.return_value = mx.gluon.block.Block()

    transformer = HandlerService._user_module_transformer()

    import_module.assert_called_once_with(MODULE_NAME)
    model_fn.assert_called_once_with(environment.model_dir)
    assert isinstance(transformer, Transformer)
    assert isinstance(transformer._default_inference_handler, DefaultGluonBlockInferenceHandler)


@patch('sagemaker_inference.environment.Environment')
@patch('importlib.import_module', return_value=UserModuleModelFn())
def test_user_module_unsupported(import_module, env):
    env.return_value.module_name = MODULE_NAME

    with pytest.raises(ValueError) as e:
        HandlerService._user_module_transformer()

    import_module.assert_called_once_with(MODULE_NAME)
    e.match('Unsupported model type')
