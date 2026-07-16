"""Smoke test: Ray Train TorchTrainer trains on 1 GPU, loss decreases, checkpoint round-trips.

Exercises the real customer path (ray.train.torch.TorchTrainer + ScalingConfig +
prepare_model) on a single GPU worker with synthetic, seeded data — deterministic,
no dataset download.
"""

import tempfile

import pytest
import torch


def _train_func(config):
    import os

    import ray.train
    import ray.train.torch
    import torch.nn as nn

    torch.manual_seed(0)
    model = ray.train.torch.prepare_model(nn.Linear(32, 1))
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    device = next(model.parameters()).device
    x = torch.randn(128, 32, device=device)
    y = torch.randn(128, 1, device=device)

    first = last = None
    for step in range(20):
        loss = nn.functional.mse_loss(model(x), y)
        if step == 0:
            first = loss.item()
        last = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()

    with tempfile.TemporaryDirectory() as tmp:
        torch.save(model.state_dict(), os.path.join(tmp, "model.pt"))
        ckpt = ray.train.Checkpoint.from_directory(tmp)
        ray.train.report(
            {
                "first_loss": first,
                "last_loss": last,
                "world_size": ray.train.get_context().get_world_size(),
            },
            checkpoint=ckpt,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_torchtrainer_single_worker_converges():
    import ray
    from ray.train import ScalingConfig
    from ray.train.torch import TorchTrainer

    ray.init(ignore_reinit_error=True)
    try:
        trainer = TorchTrainer(
            _train_func,
            scaling_config=ScalingConfig(num_workers=1, use_gpu=True),
        )
        result = trainer.fit()
    finally:
        ray.shutdown()

    metrics = result.metrics
    assert metrics["world_size"] == 1
    assert metrics["last_loss"] < metrics["first_loss"], "training loss did not decrease"
    assert result.checkpoint is not None, "no checkpoint reported"
