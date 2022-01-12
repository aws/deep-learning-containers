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
import torch
import numpy as np
import time
import os
import sys
import argparse

THRESHOLD = 28.0

parser = argparse.ArgumentParser()
parser.add_argument("--size",
                    type=int,
                    default=1,
                    help="Size of tensor to allreduce in MB")
parser.add_argument("--iterations",
                    type=int,
                    default=100,
                    help="Number of times to run allreduce")
parser.add_argument("--num_tensors",
                    type=int,
                    default=512,
                    help="How many tensors to allreduce on a single pass")
parser.add_argument("--warmup",
                    type=int,
                    default=20,
                    help="Number of times to run allreduce and ignore")
parser.add_argument("--bucket_size",
                    type=int,
                    default=25,
                    help="Number of times to run allreduce and ignore")
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
    if "MASTER_ADDR" not in os.environ:
        assert "SMDATAPARALLEL_SERVER_ADDR" in os.environ
        os.environ["MASTER_ADDR"] = os.environ["SMDATAPARALLEL_SERVER_ADDR"]
        os.environ["MASTER_PORT"] = str(
            int(os.environ["SMDATAPARALLEL_SERVER_PORT"]) + 2)
    size = int(os.getenv("OMPI_COMM_WORLD_SIZE"))
    rank = int(os.getenv("OMPI_COMM_WORLD_RANK"))
    local_size = int(os.getenv("OMPI_COMM_WORLD_LOCAL_SIZE"))
    local_rank = int(os.getenv("OMPI_COMM_WORLD_LOCAL_RANK"))

    import torch.distributed as dist
    dist.init_process_group("nccl",
                            world_size=size,
                            rank=rank,
                            store=None,
                            group_name='')
else:
    import smdistributed.dataparallel.torch.distributed as dist
    import smdistributed.dataparallel as h
    import smddpcommon as hm

    dist.init_process_group()
    hm.setBucketSize(args.bucket_size * 1024 * 1024)
    size = dist.get_world_size()
    rank = dist.get_rank()
    local_size = 8
    local_rank = dist.get_local_rank()
    HDTYPE = hm.herringFloat16
    if args.fp32:
        HDTYPE = hm.herringFloat32

torch.cuda.set_device(local_rank)

if rank == 0:
    print("NUM WORKERS", size, "WSIZE/RANK/LSIZE/LRANK:", size, rank,
          local_size, local_rank)

DTYPE = torch.float16
DTSIZE = 2

if args.fp32:
    DTYPE = torch.float32
    DTSIZE = 4
else:
    DTYPE = torch.float16
    DTSIZE = 2

bandwidth = []
artime = []


def test(warmup=False, size=104857600, num_tensors=100, iterations=1,
         first_iteration=False):
    if rank == 0: print("Warmup  " if warmup else "\n", end="\t")

    # SETUP
    device = torch.device("cuda", local_rank)
    # Create about 800MB worth of gradients that are a few times larger than a single bucket
    tests = [
        torch.ones(int(size / DTSIZE), dtype=DTYPE, device=device)
        for _ in range(num_tensors)
    ]

    # tests_ref = [x.cpu().numpy() for x in tests]

    # RUN
    for k in range(int(iterations)):
        results = []
        torch.cuda.synchronize()

        before = time.time()

        for i, test_array in enumerate(tests):
            if i < len(tests):
                if not args.nccl:
                    if (first_iteration):
                        out = test_array.data_ptr()
                    else:
                        out = 0
                    results.append(
                        hm.allReduce(test_array.data_ptr(),
                                     out, HDTYPE,
                                     test_array.numel(), i, len(tests),
                                     h._get_id_for_herring_task()).request)
                else:
                    results.append(dist.all_reduce(test_array, async_op=True))

        for i, res in enumerate(results):
            if not args.nccl:
                hm.wait(res)
            else:
                res.wait()

        torch.cuda.synchronize()  # ????^)*(&*&***??????
        if rank == 0:
            tdif = (time.time() - before)
            print("[%2d/%d %s %d %s]" %
                  (k, iterations, "%5.3fs" % tdif, size, "%5.2fGB/s" %
                   (size * num_tensors / 1024 / 1024 / 1024 /
                    (time.time() - before))),
                  end="\t" if k % 4 != 3 else "\n\t")
            sys.stdout.flush()
            if warmup: continue
            bandwidth.append(size * num_tensors / 1024 / 1024 / 1024 /
                             (time.time() - before))
            artime.append(tdif)


get_size = lambda: int(args.size * 1024 * 1024)
test(
    True,
    size=get_size(),
    num_tensors=args.num_tensors,
    iterations=1,
    first_iteration=True,
)
test(True,
     size=get_size(),
     num_tensors=args.num_tensors,
     iterations=args.warmup)
test(size=get_size(),
     num_tensors=args.num_tensors,
     iterations=args.iterations / 2)
test(size=get_size(),
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
        % (args.info, "PT.SMDATAPARALLEL" if not args.nccl else "PT.NCCL   ",
           size, "fp32" if args.fp32 else "fp16", "[%dMB x%d]x%d iter" %
           (args.size, args.num_tensors, args.iterations), net_mn, net_mx,
           alg_mn, alg_mx, np.mean(artime), np.max(artime)))

    if net_mn < THRESHOLD:
        assert False, "The network throughput value {} GB/s is below threshold {} GB/s".format(net_mn, THRESHOLD)
