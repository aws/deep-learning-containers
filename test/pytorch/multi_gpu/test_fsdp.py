"""FSDP training + checkpoint on 2 GPUs — launched by torchrun.

Usage (workflow runs this):
    torchrun --nproc_per_node=2 --master_port=29501 test/pytorch/multi_gpu/test_fsdp.py
"""

import os
import shutil

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.manual_seed(42)

    # --- Test 1: training loss decreases ---
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

    assert loss.item() < first, f"Loss did not decrease: {first} -> {loss.item()}"

    # --- Test 2: FSDP checkpoint round-trip ---
    ckpt_dir = "/tmp/fsdp_ckpt"
    state = {"model": get_model_state_dict(model)}
    dcp.save(state, storage_writer=dcp.FileSystemWriter(ckpt_dir))
    dist.barrier()

    # Reload into a fresh FSDP model
    torch.manual_seed(99)
    model2 = FSDP(nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 1)).cuda())
    state2 = {"model": get_model_state_dict(model2)}
    dcp.load(state2, storage_reader=dcp.FileSystemReader(ckpt_dir))
    set_model_state_dict(model2, state2["model"])

    # Verify: forward pass produces same output
    with torch.no_grad():
        out1 = model(x)
        out2 = model2(x)
    assert torch.allclose(out1, out2, atol=1e-6), "Checkpoint round-trip changed model output"

    if rank == 0:
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    dist.destroy_process_group()
    if rank == 0:
        print("ok")


if __name__ == "__main__":
    main()
