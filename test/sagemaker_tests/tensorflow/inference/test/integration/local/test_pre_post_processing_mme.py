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
import subprocess
import sys
import time

import pytest

import requests

from .multi_model_endpoint_test_utils import make_load_model_request, make_headers


PING_URL = "http://localhost:8080/ping"
INVOCATION_URL = "http://localhost:8080/models/{}/invoke"
MODEL_NAME = "half_plus_three"


@pytest.fixture(scope="session", autouse=True)
def volume():
    try:
        model_dir = os.path.abspath("test/resources/mme_universal_script")
        subprocess.check_call(
            "docker volume create --name model_volume_mme --opt type=none "
            "--opt device={} --opt o=bind".format(model_dir).split())
        yield model_dir
    finally:
        subprocess.check_call("docker volume rm model_volume_mme".split())


@pytest.fixture(scope="module", autouse=True)
def container(docker_base_name, tag, runtime_config):
    try:
        command = (
            "docker run {}--name sagemaker-tensorflow-serving-test -p 8080:8080"
            " --mount type=volume,source=model_volume_mme,target=/opt/ml/models,readonly"
            " -e SAGEMAKER_TFS_NGINX_LOGLEVEL=info"
            " -e SAGEMAKER_BIND_TO_PORT=8080"
            " -e SAGEMAKER_SAFE_PORT_RANGE=9000-9999"
            " -e SAGEMAKER_MULTI_MODEL=True"
            " {}:{} serve"
        ).format(runtime_config, docker_base_name, tag)

        proc = subprocess.Popen(command.split(), stdout=sys.stdout, stderr=subprocess.STDOUT)

        attempts = 0
        while attempts < 40:
            time.sleep(3)
            try:
                res_code = requests.get("http://localhost:8080/ping").status_code
                if res_code == 200:
                    break
            except:
                attempts += 1
                pass

        yield proc.pid
    finally:
        subprocess.check_call("docker rm -f sagemaker-tensorflow-serving-test".split())


@pytest.fixture
def model():
    model_data = {
        "model_name": MODEL_NAME,
        "url": "/opt/ml/models/half_plus_three/model/half_plus_three"
    }
    make_load_model_request(json.dumps(model_data))
    return MODEL_NAME


@pytest.mark.processor("cpu")
@pytest.mark.model("half_plus_three")
@pytest.mark.integration("mme")
@pytest.mark.skip_gpu
def test_ping_service():
    response = requests.get(PING_URL)
    assert 200 == response.status_code


@pytest.mark.processor("cpu")
@pytest.mark.model("half_plus_three")
@pytest.mark.integration("mme")
@pytest.mark.skip_gpu
def test_predict_json(model):
    headers = make_headers()
    data = "{\"instances\": [1.0, 2.0, 5.0]}"
    response = requests.post(INVOCATION_URL.format(model), data=data, headers=headers).json()
    assert response == {"predictions": [3.5, 4.0, 5.5]}


@pytest.mark.processor("cpu")
@pytest.mark.model("half_plus_three")
@pytest.mark.integration("mme")
@pytest.mark.skip_gpu
def test_zero_content():
    headers = make_headers()
    x = ""
    response = requests.post(INVOCATION_URL.format(MODEL_NAME), data=x, headers=headers)
    assert 500 == response.status_code
    assert "document is empty" in response.text


@pytest.mark.processor("cpu")
@pytest.mark.model("half_plus_three")
@pytest.mark.integration("mme")
@pytest.mark.skip_gpu
def test_large_input():
    data_file = "test/resources/inputs/test-large.csv"

    with open(data_file, "r") as file:
        x = file.read()
        headers = make_headers(content_type="text/csv")
        response = requests.post(INVOCATION_URL.format(MODEL_NAME), data=x, headers=headers).json()
        predictions = response["predictions"]
        assert len(predictions) == 753936


@pytest.mark.processor("cpu")
@pytest.mark.model("half_plus_three")
@pytest.mark.integration("mme")
@pytest.mark.skip_gpu
def test_csv_input():
    headers = make_headers(content_type="text/csv")
    data = "1.0,2.0,5.0"
    response = requests.post(INVOCATION_URL.format(MODEL_NAME), data=data, headers=headers).json()
    assert response == {"predictions": [3.5, 4.0, 5.5]}


@pytest.mark.processor("cpu")
@pytest.mark.model("half_plus_three")
@pytest.mark.integration("mme")
@pytest.mark.skip_gpu
def test_specific_versions():
    for version in ("123", "124"):
        headers = make_headers(content_type="text/csv", version=version)
        data = "1.0,2.0,5.0"
        response = requests.post(
            INVOCATION_URL.format(MODEL_NAME), data=data, headers=headers
        ).json()
        assert response == {"predictions": [3.5, 4.0, 5.5]}


@pytest.mark.processor("cpu")
@pytest.mark.model("half_plus_three")
@pytest.mark.integration("mme")
@pytest.mark.skip_gpu
def test_unsupported_content_type():
    headers = make_headers("unsupported-type", "predict")
    data = "aW1hZ2UgYnl0ZXM="
    response = requests.post(INVOCATION_URL.format(MODEL_NAME), data=data, headers=headers)
    assert 500 == response.status_code
    assert "unsupported content type" in response.text
