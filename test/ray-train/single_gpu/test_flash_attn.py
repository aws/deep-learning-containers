"""Validate Flash Attention 2 on a single GPU."""

import torch
from flash_attn import flash_attn_func


def test_flash_attn_func():
    B, S, H, D = 2, 128, 8, 64
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    out = flash_attn_func(q, k, v)
    assert out.shape == (B, S, H, D)
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()
