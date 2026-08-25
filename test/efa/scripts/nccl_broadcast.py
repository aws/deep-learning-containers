"""NCCL broadcast smoke test: 2 nodes x 8 GPUs, ~1.24 GB bfloat16 tensor over EFA.

Warms up with small all-reduces, then broadcasts a large tensor from rank 0.
Exercises the collective at multi-GB message size across nodes. Launched with
torchrun (one agent per node), which sets RANK / LOCAL_RANK / WORLD_SIZE /
MASTER_ADDR / MASTER_PORT in the environment.
"""

import os
import sys
import time
import traceback

import torch
import torch.distributed as dist


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dev = f"cuda:{local_rank}"
    print(
        f"[rank {rank}/{dist.get_world_size()} local={local_rank}] started; "
        f"MASTER_ADDR={os.environ.get('MASTER_ADDR')} MASTER_PORT={os.environ.get('MASTER_PORT')}",
        flush=True,
    )

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
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
