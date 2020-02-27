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

import os
import subprocess
import sys
import time

import pytest


@pytest.fixture(scope='session', autouse=True)
def volume():
    try:
        model_dir = os.path.abspath('test/resources/models')
        subprocess.check_call(
            'docker volume create --name batching_model_volume --opt type=none '
            '--opt device={} --opt o=bind'.format(model_dir).split())
        yield model_dir
    finally:
        subprocess.check_call('docker volume rm batching_model_volume'.split())


def test_run_tfs_with_batching_parameters(docker_base_name, tag, runtime_config):
    try:
        command = (
            'docker run {}--name sagemaker-tensorflow-serving-test -p 8080:8080'
            ' --mount type=volume,source=batching_model_volume,target=/opt/ml/model,readonly'
            ' -e SAGEMAKER_TFS_ENABLE_BATCHING=true'
            ' -e SAGEMAKER_TFS_MAX_BATCH_SIZE=16'
            ' -e SAGEMAKER_TFS_BATCH_TIMEOUT_MICROS=500'
            ' -e SAGEMAKER_TFS_NUM_BATCH_THREADS=100'
            ' -e SAGEMAKER_TFS_MAX_ENQUEUED_BATCHES=1'
            ' -e SAGEMAKER_TFS_DEFAULT_MODEL_NAME=half_plus_three'
            ' -e SAGEMAKER_TFS_NGINX_LOGLEVEL=info'
            ' -e SAGEMAKER_BIND_TO_PORT=8080'
            ' -e SAGEMAKER_SAFE_PORT_RANGE=9000-9999'
            ' {}:{} serve'
        ).format(runtime_config, docker_base_name, tag)

        proc = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        lines_seen = {
            'max_batch_size { value: 16 }': 0,
            'batch_timeout_micros { value: 500 }': 0,
            'num_batch_threads { value: 100 }': 0,
            'max_enqueued_batches { value: 1 }': 0
        }

        for stdout_line in iter(proc.stdout.readline, ""):
            stdout_line = str(stdout_line)
            for line in lines_seen.keys():
                if line in stdout_line:
                    lines_seen[line] += 1
            if "Entering the event loop" in stdout_line:
                for value in lines_seen.values():
                    assert value == 1
                break

    finally:
        subprocess.check_call('docker rm -f sagemaker-tensorflow-serving-test'.split())
