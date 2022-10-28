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

import shlex
import subprocess
import sys
import argparse
import json
import logging
import os
import sys
from sagemaker_training import environment
import tarfile

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def main():
    print('Starting')
    parser = argparse.ArgumentParser()


    parser.add_argument('--nproc-per-node', type=int, default=32)
    parser.add_argument('--nnodes', type=int, default=1)
    parser.add_argument('--master-port', type=str, default='55555')
    parser.add_argument('--nccl-socket-ifname', type=str, default=os.environ["SM_NETWORK_INTERFACE_NAME"])
    parser.add_argument('--train-script-args', type=str, default=" ")
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ["SM_HOSTS"]))

    args = parser.parse_args()
    env = environment.Environment()
    master_addr = env.master_hostname
    #master_addr = 'localhost'
    master_port = args.master_port
    current_host = env.current_host

    hosts = args.hosts
    node_rank = hosts.index(current_host)

    nccl_socket_ifname = args.nccl_socket_ifname

    torchrun_cmd = f'NEURON_RT_LOG_LEVEL="INFO" FI_EFA_USE_DEVICE_RDMA="1" FI_PROVIDER="efa" NCCL_DEBUG="INFO" NCCL_INIT="INFO" NCCL_DEBUG_SUBSYS="ALL" NCCL_SOCKET_IFNAME={nccl_socket_ifname} torchrun  --nproc_per_node={args.nproc_per_node} --nnodes={args.nnodes} --node_rank={node_rank} --master_addr={master_addr} --master_port={master_port} train_torchrun.py'
    logger.info(f'Calling {torchrun_cmd}')
    os.system(torchrun_cmd)

if __name__ == '__main__':
    main()