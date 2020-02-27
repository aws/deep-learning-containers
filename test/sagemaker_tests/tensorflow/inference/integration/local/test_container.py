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

import json
import os
import subprocess
import sys
import time

import pytest
import requests

BASE_URL = 'http://localhost:8080/invocations'


@pytest.fixture(scope='session', autouse=True)
def volume():
    try:
        model_dir = os.path.abspath('test/resources/models')
        subprocess.check_call(
            'docker volume create --name model_volume --opt type=none '
            '--opt device={} --opt o=bind'.format(model_dir).split())
        yield model_dir
    finally:
        subprocess.check_call('docker volume rm model_volume'.split())


@pytest.fixture(scope='module', autouse=True, params=[True, False])
def container(request, docker_base_name, tag, runtime_config):
    try:
        if request.param:
            batching_config = ' -e SAGEMAKER_TFS_ENABLE_BATCHING=true'
        else:
            batching_config = ''
        command = (
            'docker run {}--name sagemaker-tensorflow-serving-test -p 8080:8080'
            ' --mount type=volume,source=model_volume,target=/opt/ml/model,readonly'
            ' -e SAGEMAKER_TFS_DEFAULT_MODEL_NAME=half_plus_three'
            ' -e SAGEMAKER_TFS_NGINX_LOGLEVEL=info'
            ' -e SAGEMAKER_BIND_TO_PORT=8080'
            ' -e SAGEMAKER_SAFE_PORT_RANGE=9000-9999'
            ' {}'
            ' {}:{} serve'
        ).format(runtime_config, batching_config, docker_base_name, tag)

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


def make_request(data, content_type='application/json', method='predict'):
    headers = {
        'Content-Type': content_type,
        'X-Amzn-SageMaker-Custom-Attributes':
            'tfs-model-name=half_plus_three,tfs-method=%s' % method
    }
    response = requests.post(BASE_URL, data=data, headers=headers)
    return json.loads(response.content.decode('utf-8'))


def test_predict():
    x = {
        'instances': [1.0, 2.0, 5.0]
    }

    y = make_request(json.dumps(x))
    assert y == {'predictions': [3.5, 4.0, 5.5]}


def test_predict_twice():
    x = {
        'instances': [1.0, 2.0, 5.0]
    }

    y = make_request(json.dumps(x))
    z = make_request(json.dumps(x))
    assert y == {'predictions': [3.5, 4.0, 5.5]}
    assert z == {'predictions': [3.5, 4.0, 5.5]}


def test_predict_two_instances():
    x = {
        'instances': [[1.0, 2.0, 5.0], [1.0, 2.0, 5.0]]
    }

    y = make_request(json.dumps(x))
    assert y == {'predictions': [[3.5, 4.0, 5.5], [3.5, 4.0, 5.5]]}


def test_predict_jsons_json_content_type():
    x = '[1.0, 2.0, 5.0]\n[1.0, 2.0, 5.0]'
    y = make_request(x)
    assert y == {'predictions': [[3.5, 4.0, 5.5], [3.5, 4.0, 5.5]]}


def test_predict_jsonlines():
    x = '[1.0, 2.0, 5.0]\n[1.0, 2.0, 5.0]'
    y = make_request(x, 'application/jsonlines')
    assert y == {'predictions': [[3.5, 4.0, 5.5], [3.5, 4.0, 5.5]]}


def test_predict_jsons():
    x = '[1.0, 2.0, 5.0]\n[1.0, 2.0, 5.0]'
    y = make_request(x, 'application/jsons')
    assert y == {'predictions': [[3.5, 4.0, 5.5], [3.5, 4.0, 5.5]]}


def test_predict_jsons_2():
    x = '{"x": [1.0, 2.0, 5.0]}\n{"x": [1.0, 2.0, 5.0]}'
    y = make_request(x)
    assert y == {'predictions': [[3.5, 4.0, 5.5], [3.5, 4.0, 5.5]]}


def test_predict_generic_json():
    x = [1.0, 2.0, 5.0]
    y = make_request(json.dumps(x))
    assert y == {'predictions': [[3.5, 4.0, 5.5]]}


def test_predict_generic_json_two_instances():
    x = [[1.0, 2.0, 5.0], [1.0, 2.0, 5.0]]
    y = make_request(json.dumps(x))
    assert y == {'predictions': [[3.5, 4.0, 5.5], [3.5, 4.0, 5.5]]}


def test_predict_csv():
    x = '1.0'
    y = make_request(x, 'text/csv')
    assert y == {'predictions': [3.5]}


def test_predict_csv_with_zero():
    x = '0.0'
    y = make_request(x, 'text/csv')
    assert y == {'predictions': [3.0]}


def test_predict_csv_one_instance_three_values_with_zero():
    x = '0.0,2.0,5.0'
    y = make_request(x, 'text/csv')
    assert y == {'predictions': [[3.0, 4.0, 5.5]]}


def test_predict_csv_one_instance_three_values():
    x = '1.0,2.0,5.0'
    y = make_request(x, 'text/csv')
    assert y == {'predictions': [[3.5, 4.0, 5.5]]}


def test_predict_csv_two_instances_three_values():
    x = '1.0,2.0,5.0\n1.0,2.0,5.0'
    y = make_request(x, 'text/csv')
    assert y == {'predictions': [[3.5, 4.0, 5.5], [3.5, 4.0, 5.5]]}


def test_predict_csv_three_instances():
    x = '1.0\n2.0\n5.0'
    y = make_request(x, 'text/csv')
    assert y == {'predictions': [3.5, 4.0, 5.5]}


def test_predict_csv_wide_categorical_input():
    x = ('0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,0.0\n'   # noqa
         '0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,6.0,0.0\n')  # noqa

    y = make_request(x, 'text/csv')
    predictions = y['predictions']

    assert 2 == len(predictions)
    assert 30 == len(predictions[0])
    assert 97 == sum(predictions[0])  # half_plus_three with row sum 14 and n = 30
    assert 100 == sum(predictions[1])  # half_plus_three with row sum 20 and n = 30


def test_regress():
    x = {
        'signature_name': 'tensorflow/serving/regress',
        'examples': [{'x': 1.0}, {'x': 2.0}]
    }

    y = make_request(json.dumps(x), method='regress')
    assert y == {'results': [3.5, 4.0]}


def test_regress_one_instance():
    # tensorflow serving docs indicate response should have 'result' key,
    # but it is actually 'results'
    # this test will catch if they change api to match docs (unlikely)
    x = {
        'signature_name': 'tensorflow/serving/regress',
        'examples': [{'x': 1.0}]
    }

    y = make_request(json.dumps(x), method='regress')
    assert y == {'results': [3.5]}


def test_predict_bad_input():
    y = make_request('whatever')
    assert 'error' in y


def test_predict_bad_input_instances():
    x = json.dumps({'junk': 'data'})
    y = make_request(x)
    assert y['error'].startswith('Failed to process element: 0 key: junk of \'instances\' list.')


def test_predict_no_custom_attributes_header():
    x = {
        'instances': [1.0, 2.0, 5.0]
    }

    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.post(BASE_URL, data=json.dumps(x), headers=headers)
    y = json.loads(response.content.decode('utf-8'))

    assert y == {'predictions': [3.5, 4.0, 5.5]}


def test_predict_with_jsonlines():
    x = {
        'instances': [1.0, 2.0, 5.0]
    }

    headers = {
        'Content-Type': 'application/json',
        'Accept':  'application/jsonlines'
    }
    response = requests.post(BASE_URL, data=json.dumps(x), headers=headers)
    assert response.headers['Content-Type'] == 'application/jsonlines'
    assert response.content.decode('utf-8') == '{    "predictions": [3.5, 4.0, 5.5    ]}'
