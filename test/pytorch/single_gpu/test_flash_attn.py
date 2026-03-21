"""Validate Flash Attention 2 and PyTorch SDPA on a single GPU."""


def test_flash_attn_func(run_in_container):
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
        "print('ok')\"",
        gpu=True,
    )


def test_sdpa(run_in_container):
    run_in_container(
        'python -c "'
        "import torch; "
        "import torch.nn.functional as F; "
        "B, H, S, D = 2, 8, 128, 64; "
        "q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16); "
        "k = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16); "
        "v = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16); "
        "out = F.scaled_dot_product_attention(q, k, v); "
        "assert out.shape == (B, H, S, D); "
        "print('ok')\"",
        gpu=True,
    )
