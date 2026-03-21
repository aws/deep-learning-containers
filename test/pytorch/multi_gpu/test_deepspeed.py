"""Validate DeepSpeed ZeRO-2 training on 2 GPUs."""

DS_SCRIPT = """\
import os, torch, torch.nn as nn, deepspeed
ds_config = {"train_batch_size": 8, "gradient_accumulation_steps": 1,
             "zero_optimization": {"stage": 2}, "fp16": {"enabled": True}}
model = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 1))
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
engine, opt, _, _ = deepspeed.initialize(model=model, optimizer=opt, config=ds_config)
x = torch.randn(4, 32, device=engine.device, dtype=torch.float16)
y = torch.ones(4, 1, device=engine.device, dtype=torch.float16)
losses = []
for i in range(20):
    loss = nn.functional.mse_loss(engine(x), y)
    losses.append(loss.item())
    engine.backward(loss)
    engine.step()
if engine.local_rank == 0:
    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]} -> {losses[-1]}"
    print("ok")
deepspeed.comm.destroy_process_group()
"""


def test_deepspeed_zero2(run_in_container):
    cmd = f"cat > /tmp/ds_train.py << 'SCRIPT'\n{DS_SCRIPT}SCRIPT\ntorchrun --nproc_per_node=2 --master_port=29502 /tmp/ds_train.py"
    out = run_in_container(cmd, gpu=True, shm_size="1g", timeout=300)
    assert "ok" in out
