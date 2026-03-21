"""DeepSpeed ZeRO-2 training + checkpoint on 2 GPUs — launched by torchrun.

Usage (workflow runs this):
    torchrun --nproc_per_node=2 --master_port=29502 test/pytorch/multi_gpu/test_deepspeed.py
"""

import shutil

import deepspeed
import torch
import torch.nn as nn

DS_CONFIG = {
    "train_batch_size": 8,
    "gradient_accumulation_steps": 1,
    "zero_optimization": {"stage": 2},
    "fp16": {"enabled": True},
}


def main():
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 1))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    engine, opt, _, _ = deepspeed.initialize(model=model, optimizer=opt, config=DS_CONFIG)

    x = torch.randn(4, 32, device=engine.device, dtype=torch.float16)
    y = torch.ones(4, 1, device=engine.device, dtype=torch.float16)

    # --- Test 1: training loss decreases ---
    losses = []
    for _ in range(20):
        loss = nn.functional.mse_loss(engine(x), y)
        losses.append(loss.item())
        engine.backward(loss)
        engine.step()

    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]} -> {losses[-1]}"

    # --- Test 2: checkpoint round-trip ---
    ckpt_dir = "/tmp/ds_ckpt"
    engine.save_checkpoint(ckpt_dir, tag="test")

    with torch.no_grad():
        out_before = engine(x).clone()

    # Reload into a fresh engine
    torch.manual_seed(99)
    model2 = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 1))
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-2)
    engine2, _, _, _ = deepspeed.initialize(model=model2, optimizer=opt2, config=DS_CONFIG)
    engine2.load_checkpoint(ckpt_dir, tag="test")

    with torch.no_grad():
        out_after = engine2(x)

    assert torch.allclose(out_before, out_after, atol=1e-4), (
        "Checkpoint round-trip changed model output"
    )

    if engine.local_rank == 0:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        print("ok")

    deepspeed.comm.destroy_process_group()


if __name__ == "__main__":
    main()
