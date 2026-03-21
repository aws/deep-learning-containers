"""FSDP training on 2 GPUs — launched by torchrun.

Usage (workflow runs this):
    torchrun --nproc_per_node=2 --master_port=29501 test/pytorch/multi_gpu/test_fsdp.py
"""

import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.manual_seed(42)

    model = FSDP(nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 1)).cuda())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(64, 32, device="cuda")
    y = torch.randn(64, 1, device="cuda")

    first = None
    for i in range(10):
        loss = nn.functional.mse_loss(model(x), y)
        if i == 0:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()

    if rank == 0:
        assert loss.item() < first, f"Loss did not decrease: {first} -> {loss.item()}"
        print("ok")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
