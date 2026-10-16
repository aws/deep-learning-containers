"""Verify the attention, normalization and quantization kernels Miles trains and serves
with match reference implementations on the GPU."""

import importlib

import pytest
import torch
import torch.nn.functional as F
import transformer_engine.pytorch as te


def relative_error(actual, expected):
    actual, expected = actual.float(), expected.float()
    return ((actual - expected).norm() / expected.norm()).item()


def reference_attention(q, k, v, causal):
    """Attention in float32 for (batch, seq, heads, head_dim) inputs; kv heads may be fewer."""
    q, k, v = (t.float().transpose(1, 2) for t in (q, k, v))
    k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    scores = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
    if causal:
        mask = torch.ones(scores.shape[-2:], dtype=torch.bool, device=q.device).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
    return (scores.softmax(dim=-1) @ v).transpose(1, 2)


def attention_inputs(batch=2, seq=256, heads=8, head_dim=64):
    torch.manual_seed(0)
    return [
        torch.randn(
            batch, seq, heads, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        for _ in range(3)
    ]


def test_flash_attention_matches_reference_forward_and_backward():
    from flash_attn import flash_attn_func

    q, k, v = attention_inputs()
    out = flash_attn_func(q, k, v, causal=True)
    expected = reference_attention(q, k, v, causal=True)
    assert relative_error(out, expected) < 2e-2

    grad = torch.randn_like(out)
    grads = torch.autograd.grad(out, (q, k, v), grad)
    expected_grads = torch.autograd.grad(expected, (q, k, v), grad.float())
    for actual_grad, expected_grad in zip(grads, expected_grads):
        assert relative_error(actual_grad, expected_grad) < 5e-2


def test_transformer_engine_attention_matches_reference_forward_and_backward():
    # TransformerEngine picks the attention backend per GPU; training needs its backward too.
    q, k, v = attention_inputs()
    attention = te.DotProductAttention(
        num_attention_heads=8, kv_channels=64, attn_mask_type="causal", qkv_format="bshd"
    )
    out = attention(q, k, v)
    expected = reference_attention(q, k, v, causal=True).reshape(out.shape)
    assert relative_error(out, expected) < 2e-2

    grad = torch.randn_like(out)
    grads = torch.autograd.grad(out, (q, k, v), grad)
    expected_grads = torch.autograd.grad(expected, (q, k, v), grad.float())
    for actual_grad, expected_grad in zip(grads, expected_grads):
        assert relative_error(actual_grad, expected_grad) < 5e-2


def test_transformer_engine_linear_matches_torch_forward_and_backward():
    torch.manual_seed(0)
    layer = te.Linear(256, 512, params_dtype=torch.bfloat16)
    reference = torch.nn.Linear(256, 512, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        reference.weight.copy_(layer.weight)
        reference.bias.copy_(layer.bias)
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    grad = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)

    out = layer(x)
    expected = reference(x)
    assert relative_error(out, expected) < 1e-2
    (input_grad,) = torch.autograd.grad(out, x, grad)
    (expected_input_grad,) = torch.autograd.grad(expected, x, grad)
    assert relative_error(input_grad, expected_input_grad) < 1e-2


@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability() < (8, 9),
    reason="FP8 needs sm_89 or newer",
)
def test_transformer_engine_fp8_linear_tracks_bf16():
    torch.manual_seed(0)
    layer = te.Linear(256, 512, params_dtype=torch.bfloat16)
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
    expected = layer(x)
    with te.fp8_autocast(enabled=True):
        out = layer(x)
    assert torch.isfinite(out).all()
    assert relative_error(out, expected) < 0.1


@pytest.mark.parametrize("module", ["flashinfer", "sgl_kernel"])
def test_rmsnorm_matches_reference(module):
    rmsnorm = importlib.import_module(module).rmsnorm
    torch.manual_seed(0)
    x = torch.randn(128, 4096, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    expected = (
        x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6) * weight.float()
    )
    assert relative_error(rmsnorm(x, weight, 1e-6), expected) < 1e-2


@pytest.mark.parametrize("module", ["flashinfer", "sgl_kernel"])
def test_silu_and_mul_matches_reference(module):
    silu_and_mul = importlib.import_module(module).silu_and_mul
    torch.manual_seed(0)
    x = torch.randn(128, 2 * 2048, device="cuda", dtype=torch.bfloat16)
    gate, up = x.float().chunk(2, dim=-1)
    assert relative_error(silu_and_mul(x), F.silu(gate) * up) < 1e-2


def test_flashinfer_decode_attention_matches_reference():
    import flashinfer

    torch.manual_seed(0)
    q = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1024, 8, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1024, 8, 128, device="cuda", dtype=torch.bfloat16)
    out = flashinfer.single_decode_with_kv_cache(q, k, v)
    expected = reference_attention(q[None, None], k[None], v[None], causal=False)[0, 0]
    assert relative_error(out, expected) < 2e-2


@pytest.mark.parametrize("sym", [True, False])
def test_fake_int4_quantization_matches_reference(sym):
    from fake_int4_quant_cuda import fake_int4_quant_cuda

    torch.manual_seed(0)
    x = torch.randn(64, 512, device="cuda")
    out, scale, zero = fake_int4_quant_cuda(x, [1, 128], sym)
    blocks = x.view(64, 4, 128)
    if sym:
        expected_scale = (blocks.abs().amax(dim=-1) * (1 / 7)).clamp_min(1e-5)
        expected = torch.round(blocks / expected_scale[..., None])
        assert out.abs().max() <= 7
    else:
        low, high = blocks.amin(dim=-1), blocks.amax(dim=-1)
        expected_scale = ((high - low) * (1 / 15)).clamp_min(1e-5)
        expected_zero = (-torch.round(low / expected_scale)).clamp(0, 15)
        expected = torch.round(blocks / expected_scale[..., None]) + expected_zero[..., None]
        torch.testing.assert_close(zero, expected_zero, rtol=0, atol=0)
    torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(out.view(64, 4, 128), expected, rtol=0, atol=0)
