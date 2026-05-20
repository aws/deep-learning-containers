"""Multi-node NCCL all_reduce via torch.distributed, run under mpirun.

Replaces nccl-tests' all_reduce_perf in the EFA test path. nccl-tests' build
OOM-kills nvcc on the test host (verifiable.cu is heavy). Torch is already in
the vLLM image, dispatches to libnccl, and exercises the same EFA path.

Each rank initializes process group with NCCL backend, runs an all_reduce on
a 1 GiB tensor, and rank 0 prints a JSON summary the orchestrator greps.
NCCL_DEBUG=INFO output (set by the launcher) is what proves EFA was used.
"""

import json
import os
import time

import torch
import torch.distributed as dist


def main() -> None:
    rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    world = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    local = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    torch.cuda.set_device(local)

    # MASTER_ADDR/PORT supplied by the launcher script.
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world,
    )

    elem = 1 << 28  # 1 GiB of fp32
    x = torch.ones(elem, dtype=torch.float32, device=f"cuda:{local}")

    # Warm up.
    dist.all_reduce(x)
    torch.cuda.synchronize()

    iters = 20
    t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters

    # Bandwidth: bus_bw uses NCCL's standard 2*(N-1)/N factor for ring all-reduce.
    bytes_xfer = elem * 4
    algo_bw = bytes_xfer / dt / 1e9
    bus_bw = algo_bw * 2 * (world - 1) / world

    if rank == 0:
        print(
            json.dumps(
                {
                    "world": world,
                    "size_bytes": bytes_xfer,
                    "iters": iters,
                    "time_s": round(dt, 4),
                    "algo_bw_GBps": round(algo_bw, 2),
                    "bus_bw_GBps": round(bus_bw, 2),
                }
            )
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
