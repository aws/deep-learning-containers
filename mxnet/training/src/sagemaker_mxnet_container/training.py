# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License'). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the 'license' file accompanying this file. This file is
# distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import logging
import os
import socket
import subprocess

from retrying import retry
import sagemaker_containers.beta.framework as framework

from sagemaker_mxnet_container.training_utils import scheduler_host

LAUNCH_PS_ENV_NAME = 'sagemaker_parameter_server_enabled'
ROLES = ['worker', 'scheduler', 'server']

logger = logging.getLogger(__name__)


def _env_vars_for_role(role, hosts, ps_port, ps_verbose):
    if role in ROLES:
        return {
            'DMLC_NUM_WORKER': str(len(hosts)),
            'DMLC_NUM_SERVER': str(len(hosts)),
            'DMLC_ROLE': role,
            'DMLC_PS_ROOT_URI': _host_lookup(scheduler_host(hosts)),
            'DMLC_PS_ROOT_PORT': ps_port,
            'PS_VERBOSE': ps_verbose,
        }

    raise ValueError('Unexpected role: {}'.format(role))


def _run_mxnet_process(role, hosts, ps_port, ps_verbose):
    role_env = os.environ.copy()
    role_env.update(_env_vars_for_role(role, hosts, ps_port, ps_verbose))
    subprocess.Popen("python -c 'import mxnet'", shell=True, env=role_env).pid


@retry(stop_max_delay=1000 * 60 * 15, wait_exponential_multiplier=100,
       wait_exponential_max=30000)
def _host_lookup(host):
    return socket.gethostbyname(host)


def _verify_hosts(hosts):
    for host in hosts:
        _host_lookup(host)


def train(env):
    logger.info('MXNet training environment: {}'.format(env.to_env_vars()))

    if env.additional_framework_parameters.get(LAUNCH_PS_ENV_NAME, False):
        _verify_hosts(env.hosts)

        ps_port = env.hyperparameters.get('_ps_port', '8000')
        ps_verbose = env.hyperparameters.get('_ps_verbose', '0')

        logger.info('Starting distributed training task')
        if scheduler_host(env.hosts) == env.current_host:
            _run_mxnet_process('scheduler', env.hosts, ps_port, ps_verbose)
        _run_mxnet_process('server', env.hosts, ps_port, ps_verbose)
        os.environ.update(_env_vars_for_role('worker', env.hosts, ps_port, ps_verbose))

    framework.modules.download_and_install(env.module_dir)
    framework.entry_point.run(env.module_dir,
                              env.user_entry_point,
                              env.to_cmd_args(),
                              env.to_env_vars(),
                              runner=framework.runner.ProcessRunnerType)


def main():
    train(framework.training_env())
