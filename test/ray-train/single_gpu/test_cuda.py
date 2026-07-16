"""Validate CUDA availability, NCCL, and basic tensor ops on a single GPU."""

import torch


def test_cuda_available():
    assert torch.cuda.is_available()


def test_nccl_library_loadable():
    assert torch.cuda.nccl.version() is not None


def test_tensor_matmul_on_gpu():
    a = torch.randn(64, 64, device="cuda")
    b = torch.randn(64, 64, device="cuda")
    out = a @ b
    assert out.shape == (64, 64)
    assert torch.isfinite(out).all()


def test_amp_autocast():
    x = torch.randn(32, 32, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        y = x @ x
    assert torch.isfinite(y.float()).all()
