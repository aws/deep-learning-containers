"""Validate NVIDIA Transformer Engine on a single GPU."""


def test_te_linear(run_in_container):
    """te.Linear forward pass with BF16 (works on Ampere and Hopper)."""
    script = (
        "import torch\n"
        "import transformer_engine.pytorch as te\n"
        "m = te.Linear(64, 64).cuda().bfloat16()\n"
        "x = torch.randn(4, 64, device='cuda', dtype=torch.bfloat16)\n"
        "out = m(x)\n"
        "assert out.shape == (4, 64)\n"
        "print('ok')"
    )
    run_in_container(f'python -c "{script}"', gpu=True)
