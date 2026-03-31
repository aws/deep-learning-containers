"""Validate torch.compile with scaled dot-product attention on a single GPU."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttnBlock(nn.Module):
    def __init__(self, dim=64, heads=4):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)

    def forward(self, x):
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.heads, D // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        return F.scaled_dot_product_attention(q, k, v)


def test_compile_sdpa():
    m = torch.compile(SelfAttnBlock().cuda().bfloat16())
    x = torch.randn(2, 32, 64, device="cuda", dtype=torch.bfloat16)
    out = m(x)
    assert out.shape == (2, 4, 32, 16)
    assert torch.isfinite(out).all()
