"""Smoke test: train a small MLP, verify loss decreases, checkpoint round-trip."""

import os

import torch
import torch.nn as nn


def test_training_loss_decreases():
    torch.manual_seed(42)
    m = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 1)).cuda()
    opt = torch.optim.Adam(m.parameters())
    x = torch.randn(128, 32, device="cuda")
    y = torch.randn(128, 1, device="cuda")
    first = None
    for i in range(10):
        loss = nn.functional.mse_loss(m(x), y)
        if i == 0:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first


def test_checkpoint_roundtrip():
    m = nn.Linear(16, 16).cuda()
    path = "/tmp/ckpt.pt"
    torch.save(m.state_dict(), path)
    m2 = nn.Linear(16, 16).cuda()
    m2.load_state_dict(torch.load(path, weights_only=True))
    for p1, p2 in zip(m.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)
    os.remove(path)
