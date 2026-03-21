"""Multi-node DDP training — launched by torchrun on each node.

Usage (workflow runs on each node):
    torchrun --nnodes=2 --nproc_per_node=1 --node_rank=$RANK \
        --master_addr=node0 --master_port=29400 \
        test/pytorch/multi_node/test_multinode_ddp.py
"""

import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.manual_seed(42)

    # --- Test 1: training loss decreases ---
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

    assert loss.item() < first, f"Loss did not decrease: {first} -> {loss.item()}"

    # --- Test 2: gradients are synchronized across nodes ---
    torch.manual_seed(42)
    model2 = DDP(nn.Linear(32, 1).cuda(), device_ids=[local_rank])
    x2 = torch.randn(16, 32, device="cuda") * (rank + 1)
    y2 = torch.randn(16, 1, device="cuda")
    loss2 = nn.functional.mse_loss(model2(x2), y2)
    loss2.backward()

    grad = model2.module.weight.grad.clone()
    gathered = [torch.zeros_like(grad) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, grad)
    for i in range(1, len(gathered)):
        assert torch.allclose(gathered[0], gathered[i], atol=1e-6), (
            f"Gradients differ between rank 0 and rank {i}"
        )

    dist.destroy_process_group()
    if rank == 0:
        print("ok")


if __name__ == "__main__":
    main()
