"""Validate FSDP training with torchrun on 2 GPUs."""

FSDP_SCRIPT = """\
import os, torch, torch.distributed as dist, torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
dist.init_process_group("nccl")
rank = dist.get_rank()
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
model = FSDP(nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 1)).cuda())
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
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


def test_fsdp_training(run_in_container):
    cmd = f"cat > /tmp/fsdp.py << 'SCRIPT'\n{FSDP_SCRIPT}SCRIPT\ntorchrun --nproc_per_node=2 --master_port=29501 /tmp/fsdp.py"
    out = run_in_container(cmd, gpu=True, shm_size="1g", timeout=180)
    assert "ok" in out
