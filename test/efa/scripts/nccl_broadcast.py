"""NCCL broadcast smoke test: 2 nodes x 8 GPUs, ~1.24 GB bfloat16 tensor over EFA.

Warms up with small all-reduces, then broadcasts a large tensor from rank 0.
Exercises the collective at multi-GB message size across nodes.
"""

import os
import time

import torch
import torch.distributed as dist


def main():
    rank = int(os.environ.get("RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", "1")))
    local_rank = int(
        os.environ.get("LOCAL_RANK", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
    )
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    dev = f"cuda:{local_rank}"

    x = torch.empty(151693, 4096, dtype=torch.bfloat16, device=dev)
    if rank == 0:
        x.normal_()

    for _ in range(6):
        s = torch.ones(1024, device=dev)
        dist.all_reduce(s)
    torch.cuda.synchronize()
    if rank == 0:
        print(f"warmup OK; starting {x.numel() * 2 / 1e9:.2f} GB broadcast", flush=True)

    t0 = time.time()
    dist.broadcast(x, src=0)
    torch.cuda.synchronize()
    if rank == 0:
        print(f"broadcast completed in {time.time() - t0:.1f}s", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
