"""Validate DDP training with torchrun on 2 GPUs."""

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

DDP_GRAD_SYNC_SCRIPT = """\
import os, torch, torch.distributed as dist, torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
dist.init_process_group("nccl")
rank = dist.get_rank()
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
torch.manual_seed(42)
model = DDP(nn.Linear(32, 1).cuda(), device_ids=[local_rank])
# Each rank uses DIFFERENT data but DDP should sync gradients
x = torch.randn(16, 32, device="cuda") * (rank + 1)
y = torch.randn(16, 1, device="cuda")
loss = nn.functional.mse_loss(model(x), y)
loss.backward()
# After backward, DDP averages gradients across ranks.
# Collect the gradient from each rank and verify they match.
grad = model.module.weight.grad.clone()
gathered = [torch.zeros_like(grad) for _ in range(dist.get_world_size())]
dist.all_gather(gathered, grad)
for i in range(1, len(gathered)):
    assert torch.allclose(gathered[0], gathered[i], atol=1e-6), \
        f"Gradients differ between rank 0 and rank {i}"
if rank == 0:
    print("ok")
dist.destroy_process_group()
"""


def test_ddp_training(run_in_container):
    cmd = f"cat > /tmp/ddp.py << 'SCRIPT'\n{DDP_SCRIPT}SCRIPT\ntorchrun --nproc_per_node=2 --master_port=29500 /tmp/ddp.py"
    out = run_in_container(cmd, gpu=True, shm_size="1g", timeout=180)
    assert "ok" in out


def test_ddp_gradient_sync(run_in_container):
    """Verify DDP actually synchronizes gradients across ranks (not independent training)."""
    cmd = f"cat > /tmp/ddp_sync.py << 'SCRIPT'\n{DDP_GRAD_SYNC_SCRIPT}SCRIPT\ntorchrun --nproc_per_node=2 --master_port=29503 /tmp/ddp_sync.py"
    out = run_in_container(cmd, gpu=True, shm_size="1g", timeout=180)
    assert "ok" in out
