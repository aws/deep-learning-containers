"""Validate CUDA availability, tensor ops, AMP, and torch.compile on a single GPU."""

import pytest
import torch


def test_cuda_available():
    assert torch.cuda.is_available()


def test_device_count():
    assert torch.cuda.device_count() >= 1


def test_gpu_tensor_matmul():
    a = torch.randn(256, 256, device="cuda")
    b = torch.randn(256, 256, device="cuda")
    c = a @ b
    assert c.shape == (256, 256)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_amp_autocast(dtype):
    with torch.amp.autocast("cuda", dtype=dtype):
        x = torch.randn(64, 64, device="cuda")
        y = x @ x
    assert y.dtype == dtype


def test_torch_compile():
    """Verify torch.compile runs a compiled graph, not eager fallback."""
    import torch._dynamo as dynamo

    m = torch.nn.Linear(32, 32).cuda()
    m_compiled = torch.compile(m)
    x = torch.randn(4, 32, device="cuda")
    out = m_compiled(x)
    assert out.shape == (4, 32)
    explanation = dynamo.explain(m)(x)
    assert explanation.graph_count >= 1, f"No graphs captured: {explanation}"
