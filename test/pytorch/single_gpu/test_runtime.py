"""Validate DataLoader multi-worker, GPU memory management, AMP GradScaler, ONNX export."""

import gc
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def test_dataloader_multiworker():
    """Multi-worker DataLoader with pin_memory — validates /dev/shm is large enough."""
    ds = TensorDataset(torch.randn(256, 32), torch.randn(256, 1))
    dl = DataLoader(ds, batch_size=32, num_workers=2, pin_memory=True)
    count = 0
    for x, y in dl:
        assert x.is_pinned()
        count += x.shape[0]
    assert count == 256


def test_gpu_memory_cleanup():
    """Allocate GPU memory, delete tensors, verify memory is freed."""
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    x = torch.randn(1024, 1024, device="cuda")
    peak = torch.cuda.memory_allocated()
    assert peak > baseline
    del x
    gc.collect()
    torch.cuda.empty_cache()
    freed = torch.cuda.memory_allocated()
    # Allow up to 1MB slack for CUDA allocator fragmentation
    assert freed <= baseline + 1024 * 1024, f"Memory not freed: baseline={baseline}, after={freed}"


def test_amp_grad_scaler():
    """Verify AMP GradScaler training loop produces finite gradients."""
    m = nn.Linear(64, 1).cuda()
    opt = torch.optim.SGD(m.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler()
    x = torch.randn(32, 64, device="cuda")
    y = torch.randn(32, 1, device="cuda")
    for _ in range(5):
        opt.zero_grad()
        with torch.amp.autocast("cuda"):
            loss = nn.functional.mse_loss(m(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_onnx_export():
    """Verify torch.onnx.export produces a valid ONNX file."""
    m = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1)).cuda()
    x = torch.randn(1, 16, device="cuda")
    path = "/tmp/model.onnx"
    torch.onnx.export(m, x, path, input_names=["input"], output_names=["output"])
    assert os.path.getsize(path) > 0
    os.remove(path)
