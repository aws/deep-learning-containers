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

from contextlib import contextmanager

import numpy as np
import pytest
import torch
import torch.utils.data
import torch.utils.data.distributed
from sagemaker.pytorch import PyTorchModel
from sagemaker import deserializers
from sagemaker import serializers
from sagemaker_inference import content_types
from torchvision import datasets, transforms

from ...integration import training_dir, mnist_script, mnist_1d_script, model_cpu_dir, \
    model_gpu_dir, model_cpu_1d_dir, call_model_fn_once_script, ROLE
from ...utils import local_mode_utils

CONTENT_TYPE_TO_SERIALIZER_MAP = {
    content_types.CSV: serializers.CSVSerializer(),
    content_types.JSON: serializers.JSONSerializer(),
    content_types.NPY: serializers.NumpySerializer(),
}

ACCEPT_TYPE_TO_DESERIALIZER_MAP = {
    content_types.CSV: deserializers.CSVDeserializer(),
    content_types.JSON: deserializers.JSONDeserializer(),
    content_types.NPY: deserializers.NumpyDeserializer(),
}


@pytest.fixture(name='test_loader')
def fixture_test_loader():
    #  Largest batch size is only 300 because client_max_body_size is 5M
    return _get_test_data_loader(batch_size=300)


@pytest.mark.model("mnist")
def test_serve_json_npy(test_loader, use_gpu, docker_image, framework_version, sagemaker_local_session, instance_type):
    model_dir = model_gpu_dir if use_gpu else model_cpu_dir
    with _predictor(
            model_dir, mnist_script, docker_image, framework_version, sagemaker_local_session, instance_type
    ) as predictor:
        for content_type in (content_types.JSON, content_types.NPY):
            for accept in (content_types.JSON, content_types.CSV, content_types.NPY):
                _assert_prediction_npy_json(predictor, test_loader, content_type, accept)


@pytest.mark.model("mnist")
def test_serve_csv(test_loader, use_gpu, docker_image, framework_version, sagemaker_local_session, instance_type):
    with _predictor(
            model_cpu_1d_dir, mnist_1d_script, docker_image, framework_version, sagemaker_local_session, instance_type
    ) as predictor:
        for accept in (content_types.JSON, content_types.CSV, content_types.NPY):
            _assert_prediction_csv(predictor, test_loader, accept)


@pytest.mark.model("mnist")
@pytest.mark.processor("gpu")
@pytest.mark.skip_cpu
def test_serve_cpu_model_on_gpu(test_loader, docker_image, framework_version, sagemaker_local_session, instance_type):
    with _predictor(
            model_cpu_1d_dir, mnist_1d_script, docker_image, framework_version, sagemaker_local_session, instance_type
    ) as predictor:
        _assert_prediction_npy_json(predictor, test_loader, content_types.NPY, content_types.JSON)


@pytest.mark.model("mnist")
@pytest.mark.processor("cpu")
@pytest.mark.skip_gpu_py2
def test_serving_calls_model_fn_once(docker_image, framework_version, sagemaker_local_session, instance_type):
    with _predictor(
            model_cpu_dir,
            call_model_fn_once_script,
            docker_image,
            framework_version,
            sagemaker_local_session,
            instance_type,
            model_server_workers=2,
    ) as predictor:
        predictor.deserializer = deserializers.BytesDeserializer()

        # call enough times to ensure multiple requests to a worker
        for i in range(3):
            # will return 500 error if model_fn called during request handling
            output = predictor.predict(b'input')
            assert output == b'output'


@contextmanager
def _predictor(
        model_dir, script, image, framework_version, sagemaker_local_session, instance_type, model_server_workers=None
):
    model = PyTorchModel(
        'file://{}/model.tar.gz'.format(model_dir),
        ROLE,
        script,
        image_uri=image,
        framework_version=framework_version,
        sagemaker_session=sagemaker_local_session,
        model_server_workers=model_server_workers,
    )

    with local_mode_utils.lock():
        try:
            predictor = model.deploy(1, instance_type)
            yield predictor
        finally:
            predictor.delete_endpoint()


def _assert_prediction_npy_json(predictor, test_loader, content_type, accept):
    predictor.serializer = CONTENT_TYPE_TO_SERIALIZER_MAP[content_type]
    predictor.deserializer = ACCEPT_TYPE_TO_DESERIALIZER_MAP[accept]

    data = _get_mnist_batch(test_loader).numpy()
    output = predictor.predict(data)

    assert np.asarray(output).shape == (test_loader.batch_size, 10)


def _assert_prediction_csv(predictor, test_loader, accept):
    predictor.deserializer = ACCEPT_TYPE_TO_DESERIALIZER_MAP[accept]

    data = _get_mnist_batch(test_loader).view(test_loader.batch_size, -1)
    output = predictor.predict(data)
    assert np.asarray(output).shape == (test_loader.batch_size, 10)


def _get_test_data_loader(batch_size):
    return torch.utils.data.DataLoader(
        datasets.MNIST(
            training_dir,
            train=False,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
        ),
        batch_size=batch_size, shuffle=True)


def _get_mnist_batch(test_loader):
    for data in test_loader:
        return data[0]
