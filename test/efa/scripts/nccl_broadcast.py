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

    # Loop broadcast + small all_reduce to churn the send-request freelist
    # across mixed completion modes; one-shot transfers don't recycle slots.
    iters = int(os.environ.get("BROADCAST_ITERS", "500"))
    if rank == 0:
        print(
            f"looping {iters}x broadcast({x.numel() * 2 / 1e9:.2f} GB) + small all_reduce",
            flush=True,
        )

    t0 = time.time()
    for i in range(iters):
        dist.broadcast(x, src=0)
        s = torch.ones(1024, device=dev)
        dist.all_reduce(s)
        if rank == 0 and (i % 20 == 0 or i == iters - 1):
            torch.cuda.synchronize()
            print(f"  iter {i + 1}/{iters} ok ({time.time() - t0:.1f}s)", flush=True)
    torch.cuda.synchronize()
    if rank == 0:
        print(f"broadcast completed in {time.time() - t0:.1f}s ({iters} iters)", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
