"""Validate NCCL all_reduce across 2 containers via Docker network."""

import concurrent.futures

NCCL_SCRIPT = """\
import os, torch, torch.distributed as dist
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
"""


def _write_and_run(cluster, node, rank):
    cmd = (
        f"cat > /tmp/nccl_test.py << 'SCRIPT'\n{NCCL_SCRIPT}SCRIPT\n"
        f"torchrun --nnodes=2 --nproc_per_node=1 --node_rank={rank} "
        f"--master_addr=node0 --master_port=29401 /tmp/nccl_test.py 2>&1"
    )
    return cluster.exec(node, cmd, timeout=120)


def test_nccl_allreduce(multinode_cluster):
    """Run NCCL all_reduce across 2 containers, verify sum is correct."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f0 = pool.submit(_write_and_run, multinode_cluster, "node0", 0)
        f1 = pool.submit(_write_and_run, multinode_cluster, "node1", 1)
        r0 = f0.result(timeout=180)
        r1 = f1.result(timeout=180)

    assert r0.returncode == 0, f"node0 failed:\n{r0.stderr}\n{r0.stdout}"
    assert r1.returncode == 0, f"node1 failed:\n{r1.stderr}\n{r1.stdout}"
    assert "ok" in r0.stdout
