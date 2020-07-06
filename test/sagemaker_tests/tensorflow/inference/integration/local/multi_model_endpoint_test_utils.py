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

import encodings
import json
import requests

INVOCATION_URL = 'http://localhost:8080/models/{}/invoke'
MODELS_URL = 'http://localhost:8080/models'
DELETE_MODEL_URL = 'http://localhost:8080/models/{}'


def make_headers(content_type='application/json', method='predict'):
    headers = {
        'Content-Type': content_type,
        'X-Amzn-SageMaker-Custom-Attributes': 'tfs-method=%s' % method
    }
    return headers


def make_invocation_request(data, model_name, content_type='application/json'):
    headers = {
        'Content-Type': content_type,
        'X-Amzn-SageMaker-Custom-Attributes': 'tfs-method=predict'
    }
    response = requests.post(INVOCATION_URL.format(model_name), data=data, headers=headers)
    return response.status_code, response.content.decode(encodings.utf_8.getregentry().name)


def make_list_model_request():
    response = requests.get(MODELS_URL)
    return response.status_code, response.content.decode(encodings.utf_8.getregentry().name)


def make_get_model_request(model_name):
    response = requests.get(MODELS_URL + '/{}'.format(model_name))
    return response.status_code, response.content.decode(encodings.utf_8.getregentry().name)


def make_load_model_request(data, content_type='application/json'):
    headers = {
        'Content-Type': content_type
    }
    response = requests.post(MODELS_URL, data=data, headers=headers)
    return response.status_code, response.content.decode(encodings.utf_8.getregentry().name)


def make_unload_model_request(model_name):
    response = requests.delete(DELETE_MODEL_URL.format(model_name))
    return response.status_code, response.content.decode(encodings.utf_8.getregentry().name)
