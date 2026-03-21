"""Validate CUDA availability, tensor ops, AMP, and torch.compile on a single GPU."""

import pytest


def test_cuda_available(run_in_container):
    out = run_in_container("python -c 'import torch; print(torch.cuda.is_available())'", gpu=True)
    assert out == "True"


def test_device_count(run_in_container):
    out = run_in_container("python -c 'import torch; print(torch.cuda.device_count())'", gpu=True)
    assert int(out) >= 1


def test_gpu_tensor_matmul(run_in_container):
    run_in_container(
        'python -c "'
        "import torch; "
        "a = torch.randn(256, 256, device='cuda'); "
        "b = torch.randn(256, 256, device='cuda'); "
        "c = a @ b; "
        "assert c.shape == (256, 256); "
        "print('ok')\"",
        gpu=True,
    )


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_amp_autocast(run_in_container, dtype):
    script = (
        "import torch\n"
        f"with torch.amp.autocast('cuda', dtype=torch.{dtype}):\n"
        "    x = torch.randn(64, 64, device='cuda')\n"
        "    y = x @ x\n"
        f"assert y.dtype == torch.{dtype}\n"
        "print('ok')"
    )
    run_in_container(f'python -c "{script}"', gpu=True)


def test_torch_compile(run_in_container):
    """Verify torch.compile runs a compiled graph, not eager fallback."""
    script = (
        "import torch\n"
        "import torch._dynamo as dynamo\n"
        "m = torch.nn.Linear(32, 32).cuda()\n"
        "m_compiled = torch.compile(m)\n"
        "x = torch.randn(4, 32, device='cuda')\n"
        "out = m_compiled(x)\n"
        "assert out.shape == (4, 32)\n"
        "# Verify dynamo actually captured the graph\n"
        "explanation = dynamo.explain(m)(x)\n"
        "assert explanation.graph_count >= 1, f'No graphs captured: {explanation}'\n"
        "print('ok')"
    )
    run_in_container(f'python -c "{script}"', gpu=True)
