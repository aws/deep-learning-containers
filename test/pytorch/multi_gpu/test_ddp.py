"""DDP training on 2 GPUs — launched by torchrun.

Usage (workflow runs this):
    torchrun --nproc_per_node=2 --master_port=29500 test/pytorch/multi_gpu/test_ddp.py
"""

import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def test_ddp_training():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.manual_seed(42)

    model = DDP(nn.Linear(32, 1).cuda(), device_ids=[local_rank])
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
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


def test_ddp_gradient_sync():
    """Verify DDP actually synchronizes gradients across ranks."""
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.manual_seed(42)
    model = DDP(nn.Linear(32, 1).cuda(), device_ids=[local_rank])

    # Each rank uses DIFFERENT data — DDP should still sync gradients
    x = torch.randn(16, 32, device="cuda") * (rank + 1)
    y = torch.randn(16, 1, device="cuda")
    loss = nn.functional.mse_loss(model(x), y)
    loss.backward()

    grad = model.module.weight.grad.clone()
    gathered = [torch.zeros_like(grad) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, grad)
    for i in range(1, len(gathered)):
        assert torch.allclose(gathered[0], gathered[i], atol=1e-6), (
            f"Gradients differ between rank 0 and rank {i}"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    test_ddp_training()
    test_ddp_gradient_sync()
    if int(os.environ.get("RANK", 0)) == 0:
        print("ok")
