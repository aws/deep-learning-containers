# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import time
import sys
import numpy as np
import tensorflow as tf
import argparse

THRESHOLD = 40.0

parser = argparse.ArgumentParser()
parser.add_argument("--size",
                    type=int,
                    default=64,
                    help="Size of tensor to allreduce in MB")
parser.add_argument("--iterations",
                    type=int,
                    default=20,
                    help="Number of times to run allreduce")
parser.add_argument(
    "--num_tensors",
    type=int,
    default=16,
    help="How many tensors of size --size to allreduce during a single pass")
parser.add_argument("--warmup",
                    type=int,
                    default=10,
                    help="Number of times to run allreduce and ignore")
parser.add_argument("--bucket_size",
                    type=int,
                    default=25,
                    help="Bucket size in MB to perform allreduce as a group")
parser.add_argument("--info",
                    type=str,
                    default="",
                    help="Add info to test result printout")
parser.add_argument('--fp32',
                    dest='fp32',
                    action='store_true',
                    help="Data type as fp16 or fp32")
parser.add_argument('--nccl',
                    dest='nccl',
                    action='store_true',
                    help="Run nccl or herring")
parser.set_defaults(fp32=False)
parser.set_defaults(nccl=False)
args, unknown = parser.parse_known_args()
print(args)

size, rank, local_size, local_rank = None, None, None, None
if args.nccl:
    import horovod.tensorflow as dist
else:
    import smdistributed.dataparallel.tensorflow as dist
    import smddpcommon as hm

    hm.setBucketSize(args.bucket_size * 1024 * 1024)

dist.init()
size = dist.size()
rank = dist.rank()
local_size = dist.local_size()
local_rank = dist.local_rank()

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[dist.local_rank()], 'GPU')

if args.fp32:
    DTYPE, DTSIZE = tf.dtypes.float32, 4
else:
    DTYPE, DTSIZE = tf.dtypes.float16, 2


@tf.function
def hrg_one_iteration(grads):
    grads_list = []
    for i, tensor in enumerate(grads):
        tensor = dist.allreduce(grad=tensor,
                                param_index=i,
                                num_params=len(grads))
        grads_list.append(tensor)
    return grads_list


@tf.function
def hvd_one_iteration(grads):
    grads_list = []
    for i, tensor in enumerate(grads):
        tensor = dist.allreduce(tensor, average=True)
        grads_list.append(tensor)
    return grads_list


bandwidth = []
artime = []


def test(warmup=False,
         tensor_size_bytes=104857600,
         num_tensors=100,
         iterations=1):
    grad_list = [
        tf.ones(shape=int(tensor_size_bytes / DTSIZE), dtype=DTYPE)
        for _ in range(num_tensors)
    ]

    # RUN
    for k in range(int(iterations)):
        results = []

        before = time.time()
        if args.nccl:
            results = hvd_one_iteration(grad_list)
        else:
            results = hrg_one_iteration(grad_list)

        if rank == 0:
            tdif = (time.time() - before)
            print("[%2d/%d %s %d %s]" % (
                k, iterations, "%5.3fs" % tdif, tensor_size_bytes,
                "%5.2fGB/s" %
                (tensor_size_bytes * num_tensors / 1024 / 1024 / 1024 / tdif)),
                  end="\t" if k % 4 != 3 else "\n\t")
            sys.stdout.flush()
            if warmup: continue
            bandwidth.append(tensor_size_bytes * num_tensors / 1024 / 1024 /
                             1024 / tdif)
            artime.append(tdif)


get_size = lambda: int(args.size * 1024 * 1024)
test(True,
     tensor_size_bytes=get_size(),
     num_tensors=args.num_tensors,
     iterations=args.warmup)
test(tensor_size_bytes=get_size(),
     num_tensors=args.num_tensors,
     iterations=args.iterations / 2)
test(tensor_size_bytes=get_size(),
     num_tensors=args.num_tensors,
     iterations=args.iterations / 2)

# Report
if rank == 0:

    def net_throughput(alg_throughput):
        BITPERBYTE = 8
        SENDTIMES = 2
        if size == local_size:
            # single node
            return (alg_throughput * BITPERBYTE * SENDTIMES * 1)
        else:
            # multi-node
            return (alg_throughput * BITPERBYTE * SENDTIMES *
                    (1 - (local_size / size)))

    alg_mn, alg_mx = np.mean(bandwidth), np.max(bandwidth)
    net_mn, net_mx = net_throughput(alg_mn), net_throughput(alg_mx)
    print("\nFRAMEWORK TEST", str(args))
    print(
        "%s %s: %2d %s %20s [MEAN|MAX] net[%4.2f|%4.2f]GB/s alg[%4.3f|%4.3f]Gb/s time[%7.6f|%7.6f]s"
        % (args.info, "TF2.SMDATAPARALLEL" if not args.nccl else "TF2.NCCL   ",
           size, "fp32" if args.fp32 else "fp16", "[%dMB x%d]x%d iter" %
           (args.size, args.num_tensors, args.iterations), net_mn, net_mx,
           alg_mn, alg_mx, np.mean(artime), np.max(artime)))
    # TODO: Change the threshold value
    if net_mn < THRESHOLD:
        assert False, "The network throughput value {} GB/s is below threshold {} GB/s".format(net_mn, THRESHOLD)
