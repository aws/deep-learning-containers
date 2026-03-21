"""Validate NVIDIA Transformer Engine on a single GPU."""


def test_te_linear_forward_backward(run_in_container):
    """te.Linear forward + backward pass with BF16 (works on Ampere and Hopper)."""
    script = (
        "import torch\n"
        "import transformer_engine.pytorch as te\n"
        "m = te.Linear(64, 64).cuda().bfloat16()\n"
        "x = torch.randn(4, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)\n"
        "out = m(x)\n"
        "assert out.shape == (4, 64)\n"
        "assert torch.isfinite(out).all(), 'Forward output contains NaN/Inf'\n"
        "out.sum().backward()\n"
        "assert x.grad is not None, 'No gradient on input'\n"
        "assert torch.isfinite(x.grad).all(), 'Input gradient contains NaN/Inf'\n"
        "assert any(p.grad is not None for p in m.parameters()), 'No gradient on weights'\n"
        "print('ok')"
    )
    run_in_container(f'python -c "{script}"', gpu=True)
