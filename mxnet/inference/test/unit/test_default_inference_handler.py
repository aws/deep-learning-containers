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

import json
import os

from mock import call, Mock, mock_open, patch
import mxnet as mx
import pytest
from sagemaker_inference import content_types, errors

from sagemaker_mxnet_serving_container.default_inference_handler import DefaultGluonBlockInferenceHandler, \
    DefaultModuleInferenceHandler, DefaultMXNetInferenceHandler

MODEL_DIR = 'foo/model'

#################################################################################
# DefaultMXNetInferenceHandler Tests


def test_default_mxnet_valid_content_types():
    assert DefaultMXNetInferenceHandler().VALID_CONTENT_TYPES == (content_types.JSON, content_types.NPY)


@patch('mxnet.cpu')
@patch('mxnet.mod.Module')
@patch('mxnet.model.load_checkpoint')
@patch('os.path.exists', return_value=True)
def test_default_model_fn(path_exists, mx_load_checkpoint, mx_module, mx_cpu):
    sym = Mock()
    args = Mock()
    aux = Mock()
    mx_load_checkpoint.return_value = [sym, args, aux]

    mx_context = Mock()
    mx_cpu.return_value = mx_context

    data_name = 'foo'
    data_shape = [1]
    signature = json.dumps([{'name': data_name, 'shape': data_shape}])

    with patch('six.moves.builtins.open', mock_open(read_data=signature)):
        DefaultMXNetInferenceHandler().default_model_fn(MODEL_DIR)

    mx_load_checkpoint.assert_called_with(os.path.join(MODEL_DIR, 'model'), 0)

    init_call = call(symbol=sym, context=mx_context, data_names=[data_name], label_names=None)
    assert init_call in mx_module.mock_calls

    model = mx_module.return_value
    model.bind.assert_called_with(for_training=False, data_shapes=[(data_name, data_shape)])
    model.set_params.assert_called_with(args, aux, allow_missing=True)


@patch('mxnet.eia', create=True)
@patch('mxnet.mod.Module')
@patch('mxnet.model.load_checkpoint')
@patch('os.path.exists', return_value=True)
@patch.dict(os.environ, {'SAGEMAKER_INFERENCE_ACCELERATOR_PRESENT': 'true'}, clear=True)
def test_default_model_fn_with_accelerator(path_exists, mx_load_checkpoint, mx_module, mx_eia):
    sym = Mock()
    args = Mock()
    aux = Mock()
    mx_load_checkpoint.return_value = [sym, args, aux]

    eia_context = Mock()
    mx_eia.return_value = eia_context

    data_name = 'foo'
    data_shape = [1]
    signature = json.dumps([{'name': data_name, 'shape': data_shape}])

    with patch('six.moves.builtins.open', mock_open(read_data=signature)):
        DefaultMXNetInferenceHandler().default_model_fn(MODEL_DIR)

    mx_load_checkpoint.assert_called_with(os.path.join(MODEL_DIR, 'model'), 0)

    init_call = call(symbol=sym, context=eia_context, data_names=[data_name], label_names=None)
    assert init_call in mx_module.mock_calls

    model = mx_module.return_value
    model.bind.assert_called_with(for_training=False, data_shapes=[(data_name, data_shape)])
    model.set_params.assert_called_with(args, aux, allow_missing=True)


@patch('sagemaker_inference.decoder.decode', return_value=[0])
def test_mxnet_default_input_fn_with_json(decode):
    input_data = Mock()
    content_type = 'application/json'

    deserialized_data = DefaultMXNetInferenceHandler().default_input_fn(input_data, content_type)

    decode.assert_called_with(input_data, content_type)
    assert deserialized_data == mx.nd.array([0])


@patch('sagemaker_inference.decoder.decode', return_value=[0])
def test_mxnet_default_input_fn_with_npy(decode):
    input_data = Mock()
    content_type = 'application/x-npy'

    deserialized_data = DefaultMXNetInferenceHandler().default_input_fn(input_data, content_type)

    decode.assert_called_with(input_data, content_type)
    assert deserialized_data == mx.nd.array([0])


@patch('mxnet.eia', create=True)
@patch('mxnet.nd.array')
@patch('sagemaker_inference.decoder.decode', return_value=[0])
@patch.dict(os.environ, {'SAGEMAKER_INFERENCE_ACCELERATOR_PRESENT': 'true'}, clear=True)
def test_mxnet_default_input_fn_with_accelerator(decode, mx_ndarray, mx_eia):
    ndarray = Mock()
    mx_ndarray.return_value = ndarray

    DefaultMXNetInferenceHandler().default_input_fn(Mock(), 'application/json')

    ndarray.as_in_context.assert_called_with(mx.cpu())


def test_mxnet_default_input_fn_invalid_content_type():
    with pytest.raises(errors.UnsupportedFormatError) as e:
        DefaultMXNetInferenceHandler().default_input_fn(None, 'bad/content-type')
    e.match('Content type bad/content-type is not supported by this framework')


@patch('sagemaker_inference.encoder.encode', return_value=str())
def test_mxnet_default_output_fn(encode):
    prediction = mx.ndarray.zeros(1)
    accept = 'application/json'

    response = DefaultMXNetInferenceHandler().default_output_fn(prediction, accept)

    flattened_prediction = prediction.asnumpy().tolist()
    encode.assert_called_with(flattened_prediction, accept)

    assert isinstance(response, str)


def test_mxnet_default_output_fn_invalid_content_type():
    with pytest.raises(errors.UnsupportedFormatError) as e:
        DefaultMXNetInferenceHandler().default_output_fn(None, 'bad/content-type')
    e.match('Content type bad/content-type is not supported by this framework')


#################################################################################
# DefaultModuleInferenceHandler Tests


def test_default_module_valid_content_types():
    assert DefaultModuleInferenceHandler().VALID_CONTENT_TYPES == \
        (content_types.JSON, content_types.CSV, content_types.NPY)


@patch('mxnet.io.NDArrayIter')
@patch('sagemaker_inference.decoder.decode', return_value=[0])
def test_module_default_input_fn_with_json(decode, mx_ndarray_iter):
    model = Mock(data_shapes=[(1, (1,))])

    input_data = Mock()
    content_type = 'application/json'
    DefaultModuleInferenceHandler().default_input_fn(input_data, content_type, model)

    decode.assert_called_with(input_data, content_type)
    init_call = call(mx.nd.array([0]), batch_size=1, last_batch_handle='pad')
    assert init_call in mx_ndarray_iter.mock_calls


@patch('mxnet.nd.array')
@patch('mxnet.io.NDArrayIter')
@patch('sagemaker_inference.decoder.decode', return_value=[0])
def test_module_default_input_fn_with_csv(decode, mx_ndarray_iter, mx_ndarray):
    ndarray = Mock(shape=(1, (1,)))
    ndarray.reshape.return_value = ndarray
    ndarray.as_in_context.return_value = ndarray
    mx_ndarray.return_value = ndarray

    model = Mock(data_shapes=[(1, (1,))])

    input_data = Mock()
    content_type = 'text/csv'
    DefaultModuleInferenceHandler().default_input_fn(input_data, content_type, model)

    decode.assert_called_with(input_data, content_type)
    ndarray.reshape.assert_called_with((1,))
    init_call = call(mx.nd.array([0]), batch_size=1, last_batch_handle='pad')
    assert init_call in mx_ndarray_iter.mock_calls


@patch('mxnet.io.NDArrayIter')
@patch('sagemaker_inference.decoder.decode', return_value=[0])
def test_module_default_input_fn_with_npy(decode, mx_ndarray_iter):
    model = Mock(data_shapes=[(1, (1,))])

    input_data = Mock()
    content_type = 'application/x-npy'
    DefaultModuleInferenceHandler().default_input_fn(input_data, content_type, model)

    decode.assert_called_with(input_data, content_type)
    init_call = call(mx.nd.array([0]), batch_size=1, last_batch_handle='pad')
    assert init_call in mx_ndarray_iter.mock_calls


@patch('mxnet.eia', create=True)
@patch('mxnet.nd.array')
@patch('mxnet.io.NDArrayIter')
@patch('sagemaker_inference.decoder.decode', return_value=[0])
@patch.dict(os.environ, {'SAGEMAKER_INFERENCE_ACCELERATOR_PRESENT': 'true'}, clear=True)
def test_module_default_input_fn_with_accelerator(decode, mx_ndarray_iter, mx_ndarray, mx_eia):
    ndarray = Mock(shape=(1, (1,)))
    ndarray.as_in_context.return_value = ndarray
    mx_ndarray.return_value = ndarray

    model = Mock(data_shapes=[(1, (1,))])
    DefaultModuleInferenceHandler().default_input_fn(Mock(), 'application/json', model)

    ndarray.as_in_context.assert_called_with(mx.cpu())


def test_module_default_input_fn_invalid_content_type():
    with pytest.raises(errors.UnsupportedFormatError) as e:
        DefaultModuleInferenceHandler().default_input_fn(None, 'bad/content-type')
    e.match('Content type bad/content-type is not supported by this framework')


def test_module_default_predict_fn():
    module = Mock()
    data = Mock()

    DefaultModuleInferenceHandler().default_predict_fn(data, module)
    module.predict.assert_called_with(data)

#################################################################################
# DefaultGluonBlockInferenceHandler Tests


def test_gluon_default_predict_fn():
    data = [0]
    block = Mock()

    DefaultGluonBlockInferenceHandler().default_predict_fn(data, block)

    block.assert_called_with(data)
