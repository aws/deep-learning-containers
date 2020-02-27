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

import encodings
import json
import os
import subprocess
import sys
import time

import pytest
import requests

PING_URL = 'http://localhost:8080/ping'
INVOCATION_URL = 'http://localhost:8080/models/{}/invoke'
MODELS_URL = 'http://localhost:8080/models'
DELETE_MODEL_URL = 'http://localhost:8080/models/{}'


@pytest.fixture(scope='session', autouse=True)
def volume():
    try:
        model_dir = os.path.abspath('test/resources/mme')
        subprocess.check_call(
           'docker volume create --name dynamic_endpoint_model_volume --opt type=none '
           '--opt device={} --opt o=bind'.format(model_dir).split())
        yield model_dir
    finally:
        subprocess.check_call('docker volume rm dynamic_endpoint_model_volume'.split())


@pytest.fixture(scope='module', autouse=True)
def container(request, docker_base_name, tag, runtime_config):
    try:
        command = (
            'docker run {}--name sagemaker-tensorflow-serving-test -p 8080:8080'
            ' --mount type=volume,source=dynamic_endpoint_model_volume,target=/opt/ml/models,readonly'
            ' -e SAGEMAKER_TFS_NGINX_LOGLEVEL=info'
            ' -e SAGEMAKER_BIND_TO_PORT=8080'
            ' -e SAGEMAKER_SAFE_PORT_RANGE=9000-9999'
            ' -e SAGEMAKER_MULTI_MODEL=true'
            ' {}:{} serve'
        ).format(runtime_config, docker_base_name, tag)

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


def make_invocation_request(data, model_name, content_type='application/json'):
    headers = {
        'Content-Type': content_type,
        'X-Amzn-SageMaker-Custom-Attributes': 'tfs-method=predict'
    }
    response = requests.post(INVOCATION_URL.format(model_name), data=data, headers=headers)
    return response.status_code, json.loads(response.content.decode(encodings.utf_8.getregentry().name))


def make_list_model_request():
    response = requests.get(MODELS_URL)
    return response.status_code, json.loads(response.content.decode(encodings.utf_8.getregentry().name))


def make_get_model_request(model_name):
    response = requests.get(MODELS_URL + '/{}'.format(model_name))
    return response.status_code, json.loads(response.content.decode(encodings.utf_8.getregentry().name))


def make_load_model_request(data, content_type='application/json'):
    headers = {
        'Content-Type': content_type
    }
    response = requests.post(MODELS_URL, data=data, headers=headers)
    return response.status_code, response.content.decode(encodings.utf_8.getregentry().name)


def make_unload_model_request(model_name):
    response = requests.delete(DELETE_MODEL_URL.format(model_name))
    return response.status_code, response.content.decode(encodings.utf_8.getregentry().name)


def test_ping():
    res = requests.get(PING_URL)
    assert res.status_code == 200


def test_delete_unloaded_model():
    # unloads the given model/version, no-op if not loaded
    model_name = 'non-existing-model'
    code, res = make_unload_model_request(model_name)
    assert code == 404
    assert res == '{} not loaded yet.'.format(model_name)


def test_delete_model():
    model_name = 'half_plus_three'
    model_data = {
        'model_name': model_name,
        'url': '/opt/ml/models/half_plus_three'
    }
    code, res = make_load_model_request(json.dumps(model_data))
    assert code == 200
    assert res == 'Successfully loaded model {}'.format(model_name)

    x = {
        'instances': [1.0, 2.0, 5.0]
    }
    _, y = make_invocation_request(json.dumps(x), model_name)
    assert y == {'predictions': [3.5, 4.0, 5.5]}

    code_unload, res2 = make_unload_model_request(model_name)
    assert code_unload == 200

    code_invoke, y2 = make_invocation_request(json.dumps(x), model_name)
    assert code_invoke == 404
    assert y2['error'].startswith('Servable not found for request')


def test_list_models_empty():
    code, res = make_list_model_request()
    assert code == 200
    assert res == {'models': []}


def test_container_start_invocation_fail():
    x = {
        'instances': [1.0, 2.0, 5.0]
    }
    code, y = make_invocation_request(json.dumps(x), 'half_plus_three')
    assert code == 404
    assert y['error'].startswith('Servable not found for request')


def test_load_two_models():
    model_name_1 = 'half_plus_two'
    model_data_1 = {
        'model_name': model_name_1,
        'url': '/opt/ml/models/half_plus_two'
    }
    code1, res1 = make_load_model_request(json.dumps(model_data_1))
    assert code1 == 200
    assert res1 == 'Successfully loaded model {}'.format(model_name_1)

    # load second model
    model_name_2 = 'half_plus_three'
    model_data_2 = {
        'model_name': model_name_2,
        'url': '/opt/ml/models/half_plus_three'
    }
    code2, res2 = make_load_model_request(json.dumps(model_data_2))
    assert code2 == 200
    assert res2 == 'Successfully loaded model {}'.format(model_name_2)

    # make invocation request to the first model
    x = {
        'instances': [1.0, 2.0, 5.0]
    }
    code_invoke1, y1 = make_invocation_request(json.dumps(x), model_name_1)
    assert code_invoke1 == 200
    assert y1 == {'predictions': [2.5, 3.0, 4.5]}

    # make invocation request to the second model
    code_invoke2, y2 = make_invocation_request(json.dumps(x), 'half_plus_three')
    assert code_invoke2 == 200
    assert y2 == {'predictions': [3.5, 4.0, 5.5]}

    code_list, res3 = make_list_model_request()
    res3 = res3['models']
    models = [json.loads(model) for model in res3]
    assert code_list == 200
    assert models == [
        {
            "modelName": "half_plus_two",
            "modelUrl": "/opt/ml/models/half_plus_two"
        },
        {
            "modelName": "half_plus_three",
            "modelUrl": "/opt/ml/models/half_plus_three"
        }]


def test_load_one_model_two_times():
    model_name = 'cifar'
    model_data = {
        'model_name': model_name,
        'url': '/opt/ml/models/cifar'
    }
    code_load, res = make_load_model_request(json.dumps(model_data))
    assert code_load == 200
    assert res == 'Successfully loaded model {}'.format(model_name)

    code_load2, res2 = make_load_model_request(json.dumps(model_data))
    assert code_load2 == 409
    assert res2 == 'Illegal to list model {} multiple times in config list'.format(model_name)


def test_load_non_existing_model():
    model_name = 'non-existing'
    base_path = '/opt/ml/models/non-existing'
    model_data = {
        'model_name': model_name,
        'url': base_path
    }
    code, res = make_load_model_request(json.dumps(model_data))
    assert code == 404
    assert res == 'Could not find valid base path {} for servable {}'.format(base_path, model_name)


def test_bad_model_reqeust():
    bad_model_data = {
        'model_name': 'model_name',
        'uri': '/opt/ml/models/non-existing'
    }
    code, _ = make_load_model_request(json.dumps(bad_model_data))
    assert code == 500


def test_invalid_model_version():
    model_name = 'invalid_version'
    base_path = '/opt/ml/models/invalid_version'
    invalid_model_version_data = {
        'model_name': model_name,
        'url': base_path
    }
    code, res = make_load_model_request(json.dumps(invalid_model_version_data))
    assert code == 404
    assert res == 'Could not find valid base path {} for servable {}'.format(base_path, model_name)
