# Copyright 2019-2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import os
import subprocess

import pytest


@pytest.fixture(scope="session", autouse=True)
def volume():
    try:
        model_dir = os.path.abspath("test/resources/models")
        subprocess.check_call(
            "docker volume create --name nginx_model_volume --opt type=none "
            "--opt device={} --opt o=bind".format(model_dir).split()
        )
        yield model_dir
    finally:
        subprocess.check_call("docker volume rm nginx_model_volume".split())


@pytest.mark.model("N/A")
@pytest.mark.integration("nginx-config")
@pytest.mark.team("inference-toolkit")
def test_run_nginx_with_default_parameters(docker_base_name, tag, runtime_config):
    try:
        command = (
            "docker run {}--name sagemaker-tensorflow-serving-test -p 8080:8080"
            " --mount type=volume,source=nginx_model_volume,target=/opt/ml/model,readonly"
            " {}:{} serve"
        ).format(runtime_config, docker_base_name, tag)

        proc = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        lines_seen = {
            "error_log  /dev/stderr error;": 0,
            "proxy_read_timeout 60;": 0,
        }

        for stdout_line in iter(proc.stdout.readline, ""):
            stdout_line = str(stdout_line)
            for line in lines_seen.keys():
                if line in stdout_line:
                    lines_seen[line] += 1
            if "started nginx" in stdout_line:
                for value in lines_seen.values():
                    assert value == 1
                break

    finally:
        subprocess.check_call("docker rm -f sagemaker-tensorflow-serving-test".split())


@pytest.mark.model("N/A")
@pytest.mark.integration("nginx-config")
@pytest.mark.team("inference-toolkit")
def test_run_nginx_with_env_var_parameters(docker_base_name, tag, runtime_config):
    try:
        command = (
            "docker run {}--name sagemaker-tensorflow-serving-test -p 8080:8080"
            " --mount type=volume,source=nginx_model_volume,target=/opt/ml/model,readonly"
            " -e SAGEMAKER_TFS_NGINX_LOGLEVEL=info"
            " -e SAGEMAKER_NGINX_PROXY_READ_TIMEOUT_SECONDS=63"
            " {}:{} serve"
        ).format(runtime_config, docker_base_name, tag)

        proc = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        lines_seen = {
            "error_log  /dev/stderr info;": 0,
            "proxy_read_timeout 63;": 0,
        }

        for stdout_line in iter(proc.stdout.readline, ""):
            stdout_line = str(stdout_line)
            for line in lines_seen.keys():
                if line in stdout_line:
                    lines_seen[line] += 1
            if "started nginx" in stdout_line:
                for value in lines_seen.values():
                    assert value == 1
                break

    finally:
        subprocess.check_call("docker rm -f sagemaker-tensorflow-serving-test".split())


@pytest.mark.model("N/A")
@pytest.mark.integration("nginx-config")
@pytest.mark.team("inference-toolkit")
def test_run_nginx_with_higher_gunicorn_parameter(docker_base_name, tag, runtime_config):
    try:
        command = (
            "docker run {}--name sagemaker-tensorflow-serving-test -p 8080:8080"
            " --mount type=volume,source=nginx_model_volume,target=/opt/ml/model,readonly"
            " -e SAGEMAKER_NGINX_PROXY_READ_TIMEOUT_SECONDS=60"
            " -e SAGEMAKER_GUNICORN_TIMEOUT_SECONDS=120"
            " {}:{} serve"
        ).format(runtime_config, docker_base_name, tag)

        proc = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        lines_seen = {
            "proxy_read_timeout 120;": 0,  # When GUnicorn is higher, set timeout to match.
        }

        for stdout_line in iter(proc.stdout.readline, ""):
            stdout_line = str(stdout_line)
            for line in lines_seen.keys():
                if line in stdout_line:
                    lines_seen[line] += 1
            if "started nginx" in stdout_line:
                for value in lines_seen.values():
                    assert value == 1
                break

    finally:
        subprocess.check_call("docker rm -f sagemaker-tensorflow-serving-test".split())
