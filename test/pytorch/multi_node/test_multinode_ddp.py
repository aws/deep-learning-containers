"""Validate multi-node DDP training across 2 containers via Docker network."""

import concurrent.futures

DDP_SCRIPT = """\
import os, torch, torch.distributed as dist, torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
dist.init_process_group("nccl")
rank = dist.get_rank()
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
model = DDP(nn.Linear(32, 1).cuda(), device_ids=[local_rank])
opt = torch.optim.SGD(model.parameters(), lr=0.01)
x = torch.randn(64, 32, device="cuda")
y = torch.randn(64, 1, device="cuda")
first = None
for i in range(10):
    loss = nn.functional.mse_loss(model(x), y)
    if i == 0:
        first = loss.item()
    opt.zero_grad(); loss.backward(); opt.step()
if rank == 0:
    assert loss.item() < first, f"Loss did not decrease: {first} -> {loss.item()}"
    print("ok")
dist.destroy_process_group()
"""


def _write_and_run(cluster, node, rank):
    """Write the training script and launch torchrun on the given node."""
    cmd = (
        f"cat > /tmp/ddp_mn.py << 'SCRIPT'\n{DDP_SCRIPT}SCRIPT\n"
        f"torchrun --nnodes=2 --nproc_per_node=1 --node_rank={rank} "
        f"--master_addr=node0 --master_port=29400 /tmp/ddp_mn.py 2>&1"
    )
    return cluster.exec(node, cmd, timeout=120)


def test_multinode_ddp(multinode_cluster):
    """Run DDP across 2 containers, verify loss decreases on rank 0."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f0 = pool.submit(_write_and_run, multinode_cluster, "node0", 0)
        f1 = pool.submit(_write_and_run, multinode_cluster, "node1", 1)
        r0 = f0.result(timeout=180)
        r1 = f1.result(timeout=180)

    assert r0.returncode == 0, f"node0 failed:\n{r0.stderr}\n{r0.stdout}"
    assert r1.returncode == 0, f"node1 failed:\n{r1.stderr}\n{r1.stdout}"
    assert "ok" in r0.stdout
