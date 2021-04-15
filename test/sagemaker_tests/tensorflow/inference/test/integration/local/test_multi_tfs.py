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

BASE_URL = "http://localhost:8080/invocations"


@pytest.fixture(scope="session", autouse=True)
def volume():
    try:
        model_dir = os.path.abspath("test/resources/models")
        subprocess.check_call(
            "docker volume create --name multi_tfs_model_volume --opt type=none "
            "--opt device={} --opt o=bind".format(model_dir).split())
        yield model_dir
    finally:
        subprocess.check_call("docker volume rm multi_tfs_model_volume".split())


@pytest.fixture(scope="module", autouse=True, params=[True, False])
def container(request, docker_base_name, tag, runtime_config):
    try:
        if request.param:
            batching_config = " -e SAGEMAKER_TFS_ENABLE_BATCHING=true"
        else:
            batching_config = ""
        command = (
            "docker run {}--name sagemaker-tensorflow-serving-test -p 8080:8080"
            " --mount type=volume,source=multi_tfs_model_volume,target=/opt/ml/model,readonly"
            " -e SAGEMAKER_TFS_NGINX_LOGLEVEL=info"
            " -e SAGEMAKER_BIND_TO_PORT=8080"
            " -e SAGEMAKER_SAFE_PORT_RANGE=9000-9999"
            " -e SAGEMAKER_TFS_INSTANCE_COUNT=2"
            " -e SAGEMAKER_GUNICORN_WORKERS=4"
            " -e SAGEMAKER_TFS_INTER_OP_PARALLELISM=1"
            " -e SAGEMAKER_TFS_INTRA_OP_PARALLELISM=1"          
            " {}"
            " {}:{} serve"
        ).format(runtime_config, batching_config, docker_base_name, tag)

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


def make_request(data, content_type="application/json", method="predict", version=None):
    custom_attributes = "tfs-model-name=half_plus_three,tfs-method={}".format(method)
    if version:
        custom_attributes += ",tfs-model-version={}".format(version)

    headers = {
        "Content-Type": content_type,
        "X-Amzn-SageMaker-Custom-Attributes": custom_attributes,
    }
    response = requests.post(BASE_URL, data=data, headers=headers)
    return json.loads(response.content.decode("utf-8"))


def test_predict():
    x = {
        "instances": [1.0, 2.0, 5.0]
    }

    y = make_request(json.dumps(x))
    assert y == {"predictions": [3.5, 4.0, 5.5]}
