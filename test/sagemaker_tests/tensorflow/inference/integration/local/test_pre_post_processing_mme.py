# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import json
import os
import shutil
import subprocess
import sys
import time

import pytest

import requests

from multi_model_endpoint_test_utils import make_invocation_request, make_list_model_request, \
    make_get_model_request, make_load_model_request, make_unload_model_request, make_headers


PING_URL = 'http://localhost:8080/ping'
INVOCATION_URL = 'http://localhost:8080/models/{}/invoke'
MODEL_NAME = 'half_plus_three'


@pytest.fixture(scope='module', autouse=True)
def volume(tmpdir_factory, request):
    try:
        print(str(tmpdir_factory))
        model_dir = os.path.join(tmpdir_factory.mktemp('test'), 'model')
        code_dir = os.path.join(model_dir, 'code')
        test_example = 'test/resources/examples/test1'

        model_src_dir = 'test/resources/models'
        shutil.copytree(model_src_dir, model_dir)
        shutil.copytree(test_example, code_dir)

        volume_name = f'model_volume_1'
        subprocess.check_call(
            'docker volume create --name {} --opt type=none '
            '--opt device={} --opt o=bind'.format(volume_name, model_dir).split())
        yield volume_name
    finally:
        subprocess.check_call(f'docker volume rm {volume_name}'.split())


@pytest.fixture(scope='module', autouse=True)
def container(volume, docker_base_name, tag, runtime_config):
    try:
        command = (
            'docker run {}--name sagemaker-tensorflow-serving-test -p 8080:8080'
            ' --mount type=volume,source={},target=/opt/ml/models,readonly'
            ' -e SAGEMAKER_TFS_NGINX_LOGLEVEL=info'
            ' -e SAGEMAKER_BIND_TO_PORT=8080'
            ' -e SAGEMAKER_SAFE_PORT_RANGE=9000-9999'
            ' -e SAGEMAKER_MULTI_MODEL=True'
            ' {}:{} serve'
        ).format(runtime_config, volume, docker_base_name, tag)

        proc = subprocess.Popen(command.split(), stdout=sys.stdout, stderr=subprocess.STDOUT)

        attempts = 0
        while attempts < 40:
            time.sleep(3)
            try:
                res_code = requests.get('http://localhost:8080/ping').status_code
                if res_code == 200:
                    break
            except:
                attempts += 1
                pass

        yield proc.pid
    finally:
        subprocess.check_call('docker rm -f sagemaker-tensorflow-serving-test'.split())


@pytest.fixture
def model():
    model_data = {
        'model_name': MODEL_NAME,
        'url': '/opt/ml/models/half_plus_three'
    }
    make_load_model_request(json.dumps(model_data))
    return MODEL_NAME


def test_ping_service():
    response = requests.get(PING_URL)
    assert 200 == response.status_code


def test_predict_json(model):
    headers = make_headers()
    data = '{"instances": [1.0, 2.0, 5.0]}'
    response = requests.post(INVOCATION_URL.format(model), data=data, headers=headers).json()
    assert response == {'predictions': [3.5, 4.0, 5.5]}


def test_zero_content():
    headers = make_headers()
    x = ''
    response = requests.post(INVOCATION_URL.format(MODEL_NAME), data=x, headers=headers)
    assert 500 == response.status_code
    assert 'document is empty' in response.text


def test_large_input():
    data_file = 'test/resources/inputs/test-large.csv'

    with open(data_file, 'r') as file:
        x = file.read()
        headers = make_headers(content_type='text/csv')
        response = requests.post(INVOCATION_URL.format(MODEL_NAME), data=x, headers=headers).json()
        predictions = response['predictions']
        assert len(predictions) == 753936


def test_csv_input():
    headers = make_headers(content_type='text/csv')
    data = '1.0,2.0,5.0'
    response = requests.post(INVOCATION_URL.format(MODEL_NAME), data=data, headers=headers).json()
    assert response == {'predictions': [3.5, 4.0, 5.5]}


def test_unsupported_content_type():
    headers = make_headers('unsupported-type', 'predict')
    data = 'aW1hZ2UgYnl0ZXM='
    response = requests.post(INVOCATION_URL.format(MODEL_NAME), data=data, headers=headers)
    assert 500 == response.status_code
    assert 'unsupported content type' in response.text
