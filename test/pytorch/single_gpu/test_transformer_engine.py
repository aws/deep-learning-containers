"""Validate NVIDIA Transformer Engine on a single GPU."""


def test_te_linear(run_in_container):
    """te.Linear forward pass with BF16 (works on Ampere and Hopper)."""
    run_in_container(
        'python -c "'
        "import torch; "
        "import transformer_engine.pytorch as te; "
        "m = te.Linear(64, 64).cuda(); "
        "x = torch.randn(4, 64, device='cuda', dtype=torch.bfloat16); "
        "out = m(x); "
        "assert out.shape == (4, 64); "
        "print('ok')\"",
        gpu=True,
    )
