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
import argparse
import json
import logging
import os
import sys
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import torch.utils.data
import torch.utils.data.distributed

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def _get_tensor(rank, rows, columns):
    device = torch.device(
        "cuda:{}".format(dist.get_rank() % torch.cuda.device_count()) if torch.cuda.is_available()
        else "cpu"
    )
    tensor = torch.ones(rows, columns) * (rank + 1)
    return tensor.to(device)


def _get_zeros_tensor(rows, columns):
    device = torch.device(
        "cuda:{}".format(dist.get_rank() % torch.cuda.device_count()) if torch.cuda.is_available()
        else "cpu"
    )
    tensor = torch.zeros(rows, columns)
    return tensor.to(device)


def _get_zeros_tensors_list(rows, columns):
    return [_get_zeros_tensor(rows, columns) for _ in range(dist.get_world_size())]


def _get_tensors_sum(rows, columns):
    device = torch.device(
        "cuda:{}".format(dist.get_rank() % torch.cuda.device_count()) if torch.cuda.is_available()
        else "cpu"
    )
    result = (1 + dist.get_world_size()) * dist.get_world_size() / 2
    tensor = torch.ones(rows, columns) * result
    return tensor.to(device)


def _send_recv(rank, rows, columns):
    source = 0
    tensor = _get_tensor(rank, rows, columns)
    logger.debug('Rank: {},\nTensor BEFORE send_recv: {}'.format(rank, tensor))
    if rank == 0:
        for i in range(1, dist.get_world_size()):
            dist.send(tensor=tensor, dst=i)
    else:
        dist.recv(tensor=tensor, src=source)
    logger.debug('Rank: {},\nTensor AFTER send_recv: {}\n'.format(rank, tensor))

    assert torch.equal(tensor, _get_tensor(source, rows, columns)),\
        'Rank {}: Tensor was not equal to rank {} tensor after send-recv.'.format(rank, source)


def _broadcast(rank, rows, columns):
    source = 0
    tensor = _get_tensor(rank, rows, columns)
    logger.debug('Rank: {},\nTensor BEFORE broadcast: {}'.format(rank, tensor))
    dist.broadcast(tensor, src=source)
    logger.debug('Rank: {},\nTensor AFTER broadcast: {}\n'.format(rank, tensor))

    assert torch.equal(tensor, _get_tensor(source, rows, columns)), \
        'Rank {}: Tensor was not equal to rank {} tensor after broadcast.'.format(rank, source)


def _all_reduce(rank, rows, columns):
    tensor = _get_tensor(rank, rows, columns)
    logger.debug('Rank: {},\nTensor BEFORE all_reduce: {}'.format(rank, tensor))
    dist.all_reduce(tensor, op=dist.reduce_op.SUM)
    logger.debug('Rank: {},\nTensor AFTER all_reduce: {}\n'.format(rank, tensor))

    assert torch.equal(tensor, _get_tensors_sum(rows, columns)), \
        'Rank {}: Tensor was not equal to SUM of {} tensors after all_reduce.'.format(rank, dist.get_world_size())


def _reduce(rank, rows, columns):
    dest = 0
    tensor = _get_tensor(rank, rows, columns)
    logger.debug('Rank: {},\nTensor BEFORE reduce: {}'.format(rank, tensor))
    # this is inplace operation
    dist.reduce(tensor, op=dist.reduce_op.SUM, dst=dest)
    logger.debug('Rank: {},\nTensor AFTER reduce: {}\n'.format(rank, tensor))

    if rank == dest:
        assert torch.equal(tensor, _get_tensors_sum(rows, columns)), \
            'Rank {}: Tensor was not equal to SUM of {} tensors after reduce.'.format(rank, dist.get_world_size())


def _all_gather(rank, rows, columns):
    tensor = _get_tensor(rank, rows, columns)
    tensors_list = _get_zeros_tensors_list(rows, columns)
    logger.debug('Rank: {},\nTensor BEFORE all_gather: {}'.format(rank, tensor))
    dist.all_gather(tensors_list, tensor)
    logger.debug('Rank: {},\nTensor AFTER all_gather: {}. tensors_list: {}\n'.format(
        rank, tensor, tensors_list))

    # tensor shouldn't have changed
    assert torch.equal(tensor, _get_tensor(rank, rows, columns)), \
        'Rank {}: Tensor got changed after all_gather.'.format(rank)
    for i in range(dist.get_world_size()):
        assert torch.equal(tensors_list[i], _get_tensor(i, rows, columns)), \
            'Rank {}: tensors lists are not the same after all_gather.'


def _gather(rank, rows, columns):
    dest = 0
    tensor = _get_tensor(rank, rows, columns)
    if rank == dest:
        tensors_list = _get_zeros_tensors_list(rows, columns)
        logger.debug('Rank: {},\nTensor BEFORE gather: {}. tensors_list: {}'.format(
            rank, tensor, tensors_list))
        dist.gather(tensor=tensor, gather_list=tensors_list)
        logger.debug('Rank: {},\nTensor AFTER gather: {}. tensors_list: {}\n'.format(
            rank, tensor, tensors_list))
        for i in range(dist.get_world_size()):
            assert torch.equal(tensors_list[i], _get_tensor(i, rows, columns)), \
                'Rank {}: tensors lists are not the same after gather.'
    else:
        logger.debug('Rank: {},\nTensor BEFORE gather: {}\n'.format(rank, tensor))
        dist.gather(tensor=tensor, dst=dest)
        logger.debug('Rank: {},\nTensor AFTER gather: {}\n'.format(rank, tensor))

    # tensor shouldn't have changed
    assert torch.equal(tensor, _get_tensor(rank, rows, columns)), \
        'Rank {}: Tensor got changed after gather.'.format(rank)


def _scatter(rank, rows, columns):
    source = 0
    tensor = _get_tensor(rank, rows, columns)
    if rank == source:
        tensors_list = _get_zeros_tensors_list(rows, columns)
        logger.debug('Rank: {},\nTensor BEFORE scatter: {}. tensors_list: {}'.format(
            rank, tensor, tensors_list))
        dist.scatter(tensor=tensor, scatter_list=tensors_list)
    else:
        logger.debug('Rank: {},\nTensor BEFORE scatter: {}\n'.format(rank, tensor))
        dist.scatter(tensor=tensor, src=source)
    logger.debug('Rank: {},\nTensor AFTER scatter: {}\n'.format(rank, tensor))

    assert torch.equal(tensor, _get_zeros_tensor(rows, columns)), \
        'Rank {}: Tensor should be all zeroes after scatter.'.format(rank)


def _barrier(rank):
    logger.debug('Rank: {}, Waiting for other processes before the barrier.'.format(rank))
    dist.barrier()
    logger.debug('Rank: {}, Passing the barrier'.format(rank))


def main():
    print('Starting')
    parser = argparse.ArgumentParser()
    # Configurable hyperparameters
    parser.add_argument('--rows', type=int, default=1,
                        help='Number of rows in the tensor.')
    parser.add_argument('--columns', type=int, default=1,
                        help='Number of columns in the tensor.')
    parser.add_argument('--backend', type=str, default=None,
                        help='backend for distributed operations.')

    # Container environment
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument('--current-host', type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument('--model-dir', type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument('--num-gpus', type=int, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument('--num-cpus', type=int, default=os.environ["SM_NUM_CPUS"])

    args = parser.parse_args()

    number_of_processes = args.num_gpus if args.num_gpus > 0 else args.num_cpus
    world_size = number_of_processes * len(args.hosts)
    logger.info('Running \'{}\' backend on {} nodes and {} processes. World size is {}.'.format(
        args.backend, len(args.hosts), number_of_processes, world_size
    ))
    host_rank = args.hosts.index(args.current_host)
    master_addr = args.hosts[0]
    master_port = '55555'
    processes = []
    for rank in range(number_of_processes):
        process_rank = host_rank * number_of_processes + rank
        p = Process(
            target=init_processes,
            args=(args.backend,
                  master_addr,
                  master_port,
                  process_rank,
                  world_size,
                  args.rows,
                  args.columns,
                  args.current_host,
                  args.num_gpus)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    save('success', args.model_dir)


def init_processes(backend, master_addr, master_port, rank, world_size,
                   rows, columns, host, num_gpus):
    # Initialize the distributed environment.
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    logger.info('Init process rank {} on host \'{}\''.format(rank, host))
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    run(backend, rank, rows, columns, num_gpus)


def run(backend, rank, rows, columns, num_gpus):
    # https://pytorch.org/docs/master/distributed.html
    if backend == 'gloo':
        print('Run operations supported by \'gloo\' backend.')
        _broadcast(rank, rows, columns)
        _all_reduce(rank, rows, columns)
        _barrier(rank)

        # this operation supported only on cpu
        if num_gpus == 0:
            _send_recv(rank, rows, columns)
    elif backend == 'nccl':
        print('Run operations supported by \'nccl\' backend.')
        # Note: nccl does not support gather or scatter as well:
        # https://github.com/pytorch/pytorch/blob/v0.4.0/torch/lib/THD/base/data_channels/DataChannelNccl.cpp
        _broadcast(rank, rows, columns)
        _all_reduce(rank, rows, columns)
        _reduce(rank, rows, columns)
        _all_gather(rank, rows, columns)


def save(result, model_dir):
    filename = os.path.join(model_dir, result)
    if not os.path.exists(filename):
        logger.info("Saving success result")
        with open(filename, 'w') as f:
            f.write(result)


if __name__ == '__main__':
    main()
