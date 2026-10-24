"""Run under torchrun: NCCL collectives return exact results across every GPU.

The values are small integers, which float32 and bfloat16 both represent exactly, so
every comparison is exact.
"""

import os

import torch
import torch.distributed as dist

SUCCESS_MARKER = "MILES_MULTI_GPU_OK"


def check_collectives(dtype, rank, world):
    def full(size, value):
        return torch.full((size,), float(value), dtype=dtype, device="cuda")

    summed = full(1024, rank + 1)
    dist.all_reduce(summed)
    assert torch.all(summed == world * (world + 1) // 2), (dtype, summed[0].item())

    gathered = [torch.empty(16, dtype=dtype, device="cuda") for _ in range(world)]
    dist.all_gather(gathered, full(16, rank))
    assert all(torch.all(part == source) for source, part in enumerate(gathered)), dtype

    # Every rank contributes the same vector, so each slice comes back multiplied by world.
    contribution = (torch.arange(world * 8, device="cuda") % 16).to(dtype)
    scattered = torch.empty(8, dtype=dtype, device="cuda")
    dist.reduce_scatter_tensor(scattered, contribution)
    assert torch.all(scattered == contribution[rank * 8 : (rank + 1) * 8] * world), dtype

    broadcast = full(16, rank)
    dist.broadcast(broadcast, src=world - 1)
    assert torch.all(broadcast == world - 1), dtype


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    for dtype in (torch.float32, torch.bfloat16):
        check_collectives(dtype, rank, world)
    dist.barrier()
    if rank == 0:
        print(SUCCESS_MARKER)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
