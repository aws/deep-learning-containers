"""NCCL all_reduce across 2 nodes — launched by torchrun on each node.

Usage (workflow runs on each node):
    torchrun --nnodes=2 --nproc_per_node=1 --node_rank=$RANK \
        --master_addr=node0 --master_port=29401 \
        test/pytorch/multi_node/test_nccl_efa.py
"""

import os

import torch
import torch.distributed as dist


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    t = torch.tensor([rank + 1.0], device="cuda")
    dist.all_reduce(t)
    expected = 1.0 + 2.0  # sum of rank0=1 and rank1=2
    assert t.item() == expected, f"all_reduce got {t.item()}, expected {expected}"

    if rank == 0:
        print("ok")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
