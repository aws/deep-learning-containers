"""DeepSpeed ZeRO-2 training on 2 GPUs — launched by torchrun.

Usage (workflow runs this):
    torchrun --nproc_per_node=2 --master_port=29502 test/pytorch/multi_gpu/test_deepspeed.py
"""

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

    losses = []
    for _ in range(20):
        loss = nn.functional.mse_loss(engine(x), y)
        losses.append(loss.item())
        engine.backward(loss)
        engine.step()

    if engine.local_rank == 0:
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]} -> {losses[-1]}"
        print("ok")

    deepspeed.comm.destroy_process_group()


if __name__ == "__main__":
    main()
