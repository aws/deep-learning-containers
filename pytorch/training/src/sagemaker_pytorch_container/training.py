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
import os
import logging
from retrying import retry
import six
import socket
import sys
import sagemaker_containers.beta.framework as framework

MASTER_PORT = '7777'

logger = logging.getLogger(__name__)


def train(training_environment):
    """ Runs PyTorch training on a user supplied module in either a local or distributed
    SageMaker environment.
    The user supplied module and its dependencies are downloaded from S3.
    Training is invoked by calling a "train" function in the user supplied module.
    if the environment contains multiple hosts, then a distributed learning
    task is started.

    Args:
        training_environment: training environment object containing environment variables,
                               training arguments and hyperparameters
    """
    # Block until all host DNS lookups succeed. Relies on retrying dns_lookup.
    logger.info('Block until all host DNS lookups succeed.')
    for host in training_environment.hosts:
        _dns_lookup(host)

    _set_nccl_environment(training_environment.network_interface_name)

    _set_distributed_environment(training_environment.hosts)

    logger.info('Invoking user training script.')
    try:
        framework.modules.download_and_install(training_environment.module_dir)
        framework.entry_point.run(training_environment.module_dir, training_environment.user_entry_point,
                                  training_environment.to_cmd_args(), training_environment.to_env_vars(),
                                  capture_error=True, runner=framework.runner.ProcessRunnerType)
    except framework.errors.ExecuteUserScriptError as err:
        message = str(err)
        if message.find('terminate called after throwing an instance of \'gloo::EnforceNotMet\'') > -1:
            logger.warn('Known exception: {}'.format(message))
        else:
            info = sys.exc_info()
            six.reraise(info[0], err, info[2])


@retry(stop_max_delay=1000 * 60 * 15,
       wait_exponential_multiplier=100,
       wait_exponential_max=30000)
def _dns_lookup(host):
    """ Retrying dns lookup on host """
    return socket.gethostbyname(host)


def _set_distributed_environment(hosts):
    """
    Sets environment variable for distributed training.
    Args:
        hosts: list of hosts that are used for training.
    """
    # According to https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html
    # hosts are sorted lexicographically.
    os.environ['MASTER_ADDR'] = hosts[0]
    os.environ['MASTER_PORT'] = MASTER_PORT


def _set_nccl_environment(network_interface_name):
    """ Sets NCCL environment variables for the container:
    https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html#ncclknobs

    Args:
        network_interface_name: The name of the network interface to use for distributed training.
    """
    # Set the network interface for inter node communication
    os.environ['NCCL_SOCKET_IFNAME'] = network_interface_name
    # Disable IB transport and force to use IP sockets by default
    os.environ['NCCL_IB_DISABLE'] = '1'
    # Set to INFO for more NCCL debugging information
    os.environ['NCCL_DEBUG'] = 'WARN'


def main():
    train(framework.training_env())
