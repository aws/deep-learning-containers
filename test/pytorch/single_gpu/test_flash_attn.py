"""Validate Flash Attention 2 and PyTorch SDPA on a single GPU."""

import torch
from flash_attn import flash_attn_func


def test_flash_attn_func():
    """Verify flash_attn_func produces correct-shaped, finite output."""
    B, S, H, D = 2, 128, 8, 64
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    out = flash_attn_func(q, k, v)
    assert out.shape == (B, S, H, D)
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all(), "Output contains NaN/Inf"


def test_flash_attn_backward():
    """Verify flash_attn_func supports backward pass (gradient flow)."""
    B, S, H, D = 2, 64, 4, 32
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16, requires_grad=True)
    out = flash_attn_func(q, k, v)
    out.sum().backward()
    assert q.grad is not None, "No gradient on q"
    assert torch.isfinite(q.grad).all(), "Gradient contains NaN/Inf"


def test_sdpa_flash_backend():
    """Verify SDPA uses an efficient backend, not math fallback."""
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    B, H, S, D = 2, 8, 128, 64
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        out = F.scaled_dot_product_attention(q, k, v)
    assert out.shape == (B, H, S, D)
    assert torch.isfinite(out).all()
