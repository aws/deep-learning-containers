# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from mock import call, MagicMock, patch
import pytest
import sagemaker_containers.beta.framework as framework

from sagemaker_mxnet_container import training

MODULE_DIR = 's3://my/bucket'
MODULE_NAME = 'script_name'

SCHEDULER = 'host-1'
SINGLE_HOST_LIST = [SCHEDULER]
MULTIPLE_HOST_LIST = [SCHEDULER, 'host-2', 'host-3']

IP_ADDRESS = '0.0.0.0000'
DEFAULT_PORT = '8000'
DEFAULT_VERBOSITY = '0'
BASE_ENV_VARS = {
    'DMLC_NUM_WORKER': str(len(MULTIPLE_HOST_LIST)),
    'DMLC_NUM_SERVER': str(len(MULTIPLE_HOST_LIST)),
    'DMLC_PS_ROOT_URI': IP_ADDRESS,
    'DMLC_PS_ROOT_PORT': DEFAULT_PORT,
    'PS_VERBOSE': DEFAULT_VERBOSITY,
}

MXNET_COMMAND = "python -c 'import mxnet'"


@pytest.fixture
def single_machine_training_env():
    env = MagicMock()

    env.module_dir = MODULE_DIR
    env.user_entry_point = MODULE_NAME
    env.hyperparameters = {}
    env.additional_framework_parameters = {}

    return env


@pytest.fixture
def distributed_training_env():
    env = MagicMock()

    env.module_dir = MODULE_DIR
    env.user_entry_point = MODULE_NAME
    env.hyperparameters = {}

    env.hosts = MULTIPLE_HOST_LIST
    env.additional_framework_parameters = {
        training.LAUNCH_PS_ENV_NAME: True,
    }

    return env


@patch('os.environ', {})
@patch('subprocess.Popen')
@patch('sagemaker_mxnet_container.training._host_lookup')
@patch('sagemaker_mxnet_container.training._verify_hosts')
@patch('sagemaker_containers.beta.framework.modules.download_and_install')
@patch('sagemaker_containers.beta.framework.entry_point.run')
def test_train_for_distributed_scheduler(run_entry_point, download_and_install, verify_hosts,
                                         host_lookup, popen, distributed_training_env):
    host_lookup.return_value = IP_ADDRESS

    distributed_training_env.current_host = SCHEDULER
    training.train(distributed_training_env)

    verify_hosts.assert_called_with(MULTIPLE_HOST_LIST)

    scheduler_env = BASE_ENV_VARS.copy()
    scheduler_env.update({'DMLC_ROLE': 'scheduler'})

    server_env = BASE_ENV_VARS.copy()
    server_env.update({'DMLC_ROLE': 'server'})

    calls = [call(MXNET_COMMAND, shell=True, env=scheduler_env),
             call(MXNET_COMMAND, shell=True, env=server_env)]

    popen.assert_has_calls(calls)

    download_and_install.assert_called_with(MODULE_DIR)
    run_entry_point.assert_called_with(MODULE_DIR,
                                       MODULE_NAME,
                                       distributed_training_env.to_cmd_args(),
                                       distributed_training_env.to_env_vars(),
                                       runner=framework.runner.ProcessRunnerType)


@patch('os.environ', {})
@patch('subprocess.Popen')
@patch('sagemaker_mxnet_container.training._host_lookup')
@patch('sagemaker_mxnet_container.training._verify_hosts')
@patch('sagemaker_containers.beta.framework.modules.download_and_install')
@patch('sagemaker_containers.beta.framework.entry_point.run')
def test_train_for_distributed_worker(run_entry_point, download_and_install, verify_hosts,
                                      host_lookup, popen, distributed_training_env):
    host_lookup.return_value = IP_ADDRESS

    distributed_training_env.current_host = 'host-2'
    training.train(distributed_training_env)

    verify_hosts.assert_called_with(MULTIPLE_HOST_LIST)

    server_env = BASE_ENV_VARS.copy()
    server_env.update({'DMLC_ROLE': 'server'})

    popen.assert_called_once_with(MXNET_COMMAND, shell=True, env=server_env)

    download_and_install.assert_called_with(MODULE_DIR)
    run_entry_point.assert_called_with(MODULE_DIR,
                                       MODULE_NAME,
                                       distributed_training_env.to_cmd_args(),
                                       distributed_training_env.to_env_vars(),
                                       runner=framework.runner.ProcessRunnerType)


@patch('sagemaker_containers.beta.framework.modules.download_and_install')
@patch('sagemaker_containers.beta.framework.entry_point.run')
def test_train_for_single_machine(run_entry_point, download_and_install,
                                  single_machine_training_env):
    training.train(single_machine_training_env)
    download_and_install.assert_called_with(MODULE_DIR)
    run_entry_point.assert_called_with(MODULE_DIR,
                                       MODULE_NAME,
                                       single_machine_training_env.to_cmd_args(),
                                       single_machine_training_env.to_env_vars(),
                                       runner=framework.runner.ProcessRunnerType)


@patch('sagemaker_mxnet_container.training.train')
@patch('sagemaker_containers.beta.framework.training_env')
def test_main(env, train, single_machine_training_env):
    env.return_value = single_machine_training_env

    training.main()
    train.assert_called_with(single_machine_training_env)
