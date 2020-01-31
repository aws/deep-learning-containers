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

from sagemaker_mxnet_serving_container.mxnet_module_transformer import MXNetModuleTransformer

CONTENT_TYPE = 'content'
ACCEPT = 'accept'
DATA = 'data'
MODEL = 'foo'

PREPROCESSED_DATA = 'preprocessed_data'
PREDICT_RESULT = 'prediction_result'
PROCESSED_RESULT = 'processed_result'


def _input_fn_with_model(input_data, content_type, model):
    return PREPROCESSED_DATA


@patch('importlib.import_module', return_value=object())
def test_default_transform_fn(import_module):
    predict_fn = Mock(return_value=PREDICT_RESULT)
    output_fn = Mock(return_value=PROCESSED_RESULT)

    module_transformer = MXNetModuleTransformer()
    module_transformer._input_fn = _input_fn_with_model
    module_transformer._predict_fn = predict_fn
    module_transformer._output_fn = output_fn

    result = module_transformer._default_transform_fn(MODEL, DATA, CONTENT_TYPE, ACCEPT)

    predict_fn.assert_called_once_with(PREPROCESSED_DATA, MODEL)
    output_fn.assert_called_once_with(PREDICT_RESULT, ACCEPT)
    assert PROCESSED_RESULT == result


@patch('importlib.import_module', return_value=object())
def test_call_input_fn(import_module):
    module_transformer = MXNetModuleTransformer()
    module_transformer._input_fn = _input_fn_with_model

    result = module_transformer._call_input_fn(DATA, CONTENT_TYPE, MODEL)

    assert PREPROCESSED_DATA == result


def _input_fn_without_model(input_data, content_type):
    return PREPROCESSED_DATA


@patch('importlib.import_module', return_value=object())
def test_call_input_fn_without_model_arg(import_module):
    module_transformer = MXNetModuleTransformer()
    module_transformer._input_fn = _input_fn_without_model

    result = module_transformer._call_input_fn(MODEL, DATA, CONTENT_TYPE)

    assert PREPROCESSED_DATA == result
