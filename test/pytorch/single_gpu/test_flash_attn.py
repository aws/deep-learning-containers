"""Validate Flash Attention 2 and PyTorch SDPA on a single GPU."""


def test_flash_attn_func(run_in_container):
    """Verify flash_attn_func produces correct-shaped, finite output."""
    run_in_container(
        'python -c "'
        "import torch; "
        "from flash_attn import flash_attn_func; "
        "B, S, H, D = 2, 128, 8, 64; "
        "q = torch.randn(B, S, H, D, device='cuda', dtype=torch.float16); "
        "k = torch.randn(B, S, H, D, device='cuda', dtype=torch.float16); "
        "v = torch.randn(B, S, H, D, device='cuda', dtype=torch.float16); "
        "out = flash_attn_func(q, k, v); "
        "assert out.shape == (B, S, H, D); "
        "assert out.dtype == torch.float16; "
        "assert torch.isfinite(out).all(), 'Output contains NaN/Inf'; "
        "print('ok')\"",
        gpu=True,
    )


def test_flash_attn_backward(run_in_container):
    """Verify flash_attn_func supports backward pass (gradient flow)."""
    script = (
        "import torch\n"
        "from flash_attn import flash_attn_func\n"
        "B, S, H, D = 2, 64, 4, 32\n"
        "q = torch.randn(B, S, H, D, device='cuda', dtype=torch.float16, requires_grad=True)\n"
        "k = torch.randn(B, S, H, D, device='cuda', dtype=torch.float16, requires_grad=True)\n"
        "v = torch.randn(B, S, H, D, device='cuda', dtype=torch.float16, requires_grad=True)\n"
        "out = flash_attn_func(q, k, v)\n"
        "out.sum().backward()\n"
        "assert q.grad is not None, 'No gradient on q'\n"
        "assert torch.isfinite(q.grad).all(), 'Gradient contains NaN/Inf'\n"
        "print('ok')"
    )
    run_in_container(f'python -c "{script}"', gpu=True)


def test_sdpa_flash_backend(run_in_container):
    """Verify SDPA uses an efficient backend (flash or mem_efficient), not math fallback."""
    script = (
        "import torch\n"
        "import torch.nn.functional as F\n"
        "B, H, S, D = 2, 8, 128, 64\n"
        "q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)\n"
        "k = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)\n"
        "v = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)\n"
        "out = F.scaled_dot_product_attention(q, k, v)\n"
        "assert out.shape == (B, H, S, D)\n"
        "assert torch.isfinite(out).all(), 'Output contains NaN/Inf'\n"
        "# Verify an efficient backend is available (not just math fallback)\n"
        "from torch.nn.attention import SDPBackend, sdpa_kernel\n"
        "with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):\n"
        "    out2 = F.scaled_dot_product_attention(q, k, v)\n"
        "assert torch.isfinite(out2).all()\n"
        "print('ok')"
    )
    run_in_container(f'python -c "{script}"', gpu=True)
