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

import os
from subprocess import CalledProcessError

from retrying import retry
from sagemaker_inference import model_server

from sagemaker_mxnet_serving_container import handler_service

HANDLER_SERVICE = handler_service.__name__

DEFAULT_ENV_VARS = {
    'MXNET_CPU_WORKER_NTHREADS': '1',
    'MXNET_CPU_PRIORITY_NTHREADS': '1',
    'MXNET_KVSTORE_REDUCTION_NTHREADS': '1',
    'OMP_NUM_THREADS': '1',
}


def _update_mxnet_env_vars():
    for k, v in DEFAULT_ENV_VARS.items():
        if k not in os.environ:
            os.environ[k] = v


def _retry_if_error(exception):
    return isinstance(exception, CalledProcessError or OSError)


@retry(stop_max_delay=1000 * 50,
       retry_on_exception=_retry_if_error)
def _start_model_server():
    # there's a race condition that causes the model server command to
    # sometimes fail with 'bad address'. more investigation needed
    # retry starting mms until it's ready
    model_server.start_model_server(handler_service=HANDLER_SERVICE)


def main():
    _update_mxnet_env_vars()
    _start_model_server()
