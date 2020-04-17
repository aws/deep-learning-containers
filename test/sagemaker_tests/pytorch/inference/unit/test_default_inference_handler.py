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

import csv
import json

import mock
import numpy as np
import pytest
import torch
import torch.nn as nn
from sagemaker_inference import content_types, errors
from six import StringIO, BytesIO
from torch.autograd import Variable

from sagemaker_pytorch_serving_container import default_inference_handler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DummyModel(nn.Module):

    def __init__(self, ):
        super(DummyModel, self).__init__()

    def forward(self, x):
        pass

    def __call__(self, tensor):
        return 3 * tensor


@pytest.fixture(scope="session", name="tensor")
def fixture_tensor():
    tensor = torch.rand(5, 10, 7, 9)
    return tensor.to(device)


@pytest.fixture()
def inference_handler():
    return default_inference_handler.DefaultPytorchInferenceHandler()


@pytest.fixture()
def eia_inference_handler():
    return default_inference_handler.DefaultPytorchInferenceHandler()


def test_default_model_fn(inference_handler):
    with pytest.raises(NotImplementedError):
        inference_handler.default_model_fn("model_dir")


def test_default_input_fn_json(inference_handler, tensor):
    json_data = json.dumps(tensor.cpu().numpy().tolist())
    deserialized_np_array = inference_handler.default_input_fn(json_data, content_types.JSON)

    assert deserialized_np_array.is_cuda == torch.cuda.is_available()
    assert torch.equal(tensor, deserialized_np_array)


def test_default_input_fn_csv(inference_handler):
    array = [[1, 2, 3], [4, 5, 6]]
    str_io = StringIO()
    csv.writer(str_io, delimiter=",").writerows(array)

    deserialized_np_array = inference_handler.default_input_fn(str_io.getvalue(), content_types.CSV)

    tensor = torch.FloatTensor(array).to(device)
    assert torch.equal(tensor, deserialized_np_array)
    assert deserialized_np_array.is_cuda == torch.cuda.is_available()


def test_default_input_fn_csv_bad_columns(inference_handler):
    str_io = StringIO()
    csv_writer = csv.writer(str_io, delimiter=",")
    csv_writer.writerow([1, 2, 3])
    csv_writer.writerow([1, 2, 3, 4])

    with pytest.raises(ValueError):
        inference_handler.default_input_fn(str_io.getvalue(), content_types.CSV)


def test_default_input_fn_npy(inference_handler, tensor):
    stream = BytesIO()
    np.save(stream, tensor.cpu().numpy())
    deserialized_np_array = inference_handler.default_input_fn(stream.getvalue(), content_types.NPY)

    assert deserialized_np_array.is_cuda == torch.cuda.is_available()
    assert torch.equal(tensor, deserialized_np_array)


def test_default_input_fn_bad_content_type(inference_handler):
    with pytest.raises(errors.UnsupportedFormatError):
        inference_handler.default_input_fn("", "application/not_supported")


def test_default_predict_fn(inference_handler, tensor):
    model = DummyModel()
    prediction = inference_handler.default_predict_fn(tensor, model)
    assert torch.equal(model(Variable(tensor)), prediction)
    assert prediction.is_cuda == torch.cuda.is_available()


def test_default_predict_fn_cpu_cpu(inference_handler, tensor):
    prediction = inference_handler.default_predict_fn(tensor.cpu(), DummyModel().cpu())

    model = DummyModel().to(device)
    assert torch.equal(model(Variable(tensor)), prediction)
    assert prediction.is_cuda == torch.cuda.is_available()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
def test_default_predict_fn_cpu_gpu(inference_handler, tensor):
    model = DummyModel().cuda()
    prediction = inference_handler.default_predict_fn(tensor.cpu(), model)
    assert torch.equal(model(tensor), prediction)
    assert prediction.is_cuda is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
def test_default_predict_fn_gpu_cpu(inference_handler, tensor):
    prediction = inference_handler.default_predict_fn(tensor.cpu(), DummyModel().cpu())
    model = DummyModel().cuda()
    assert torch.equal(model(tensor), prediction)
    assert prediction.is_cuda is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
def test_default_predict_fn_gpu_gpu(inference_handler, tensor):
    tensor = tensor.cuda()
    model = DummyModel().cuda()
    prediction = inference_handler.default_predict_fn(tensor, model)
    assert torch.equal(model(tensor), prediction)
    assert prediction.is_cuda is True


def test_default_output_fn_json(inference_handler, tensor):
    output = inference_handler.default_output_fn(tensor, content_types.JSON)

    assert json.dumps(tensor.cpu().numpy().tolist()) == output


def test_default_output_fn_csv_long(inference_handler):
    tensor = torch.LongTensor([[1, 2, 3], [4, 5, 6]])
    output = inference_handler.default_output_fn(tensor, content_types.CSV)

    assert '1,2,3\n4,5,6\n'.encode("utf-8") == output


def test_default_output_fn_csv_float(inference_handler):
    tensor = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
    output = inference_handler.default_output_fn(tensor, content_types.CSV)

    assert '1.0,2.0,3.0\n4.0,5.0,6.0\n'.encode("utf-8") == output


def test_default_output_fn_bad_accept(inference_handler):
    with pytest.raises(errors.UnsupportedFormatError):
        inference_handler.default_output_fn("", "application/not_supported")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
def test_default_output_fn_gpu(inference_handler):
    tensor_gpu = torch.LongTensor([[1, 2, 3], [4, 5, 6]]).cuda()

    output = inference_handler.default_output_fn(tensor_gpu, content_types.CSV)

    assert "1,2,3\n4,5,6\n".encode("utf-8") == output


def test_eia_default_model_fn(eia_inference_handler):
    with mock.patch("sagemaker_pytorch_serving_container.default_inference_handler.os") as mock_os:
        mock_os.getenv.return_value = "true"
        mock_os.path.join.return_value = "model_dir"
        mock_os.path.exists.return_value = True
        with mock.patch("torch.jit.load") as mock_torch:
            mock_torch.return_value = DummyModel()
            model = eia_inference_handler.default_model_fn("model_dir")
    assert model is not None


def test_eia_default_model_fn_error(eia_inference_handler):
    with mock.patch("sagemaker_pytorch_serving_container.default_inference_handler.os") as mock_os:
        mock_os.getenv.return_value = "true"
        mock_os.path.join.return_value = "model_dir"
        mock_os.path.exists.return_value = False
        with pytest.raises(FileNotFoundError):
            eia_inference_handler.default_model_fn("model_dir")


def test_eia_default_predict_fn(eia_inference_handler, tensor):
    model = DummyModel()
    with mock.patch("sagemaker_pytorch_serving_container.default_inference_handler.os") as mock_os:
        mock_os.getenv.return_value = "true"
        with mock.patch("torch.jit.optimized_execution") as mock_torch:
            mock_torch.__enter__.return_value = "dummy"
            eia_inference_handler.default_predict_fn(tensor, model)
        mock_torch.assert_called_once()
