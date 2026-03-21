"""Validate NVIDIA Transformer Engine on a single GPU."""

import torch
import transformer_engine.pytorch as te


def test_te_linear_forward_backward():
    """te.Linear forward + backward pass with BF16."""
    m = te.Linear(64, 64).cuda().bfloat16()
    x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = m(x)
    assert out.shape == (4, 64)
    assert torch.isfinite(out).all(), "Forward output contains NaN/Inf"
    out.sum().backward()
    assert x.grad is not None, "No gradient on input"
    assert torch.isfinite(x.grad).all(), "Input gradient contains NaN/Inf"
    assert any(p.grad is not None for p in m.parameters()), "No gradient on weights"
