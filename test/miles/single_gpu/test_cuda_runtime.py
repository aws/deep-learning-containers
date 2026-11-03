"""Verify the CUDA stack does real work on the GPU: libraries, compilers and JIT paths."""

import json
import os
import subprocess

import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline

ADD_ONE_SOURCE = r"""
#include <torch/extension.h>

__global__ void add_one(float* values, int64_t n) {
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i < n) values[i] += 1.0f;
}

void run(torch::Tensor values) {
  int64_t n = values.numel();
  add_one<<<(n + 255) / 256, 256>>>(values.data_ptr<float>(), n);
}
"""


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def test_gpu_is_visible():
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() >= 1


def test_gpu_architecture_is_one_the_image_builds_for():
    major, minor = torch.cuda.get_device_capability()
    assert f"{major}.{minor}" in os.environ["MILES_SUPPORTED_CUDA_ARCHES"].split(",")


def test_nvtop_reports_the_gpu():
    # nvtop reads the GPU through NVML, from the driver the container runtime mounts.
    snapshot = subprocess.run(
        ["nvtop", "--snapshot"], check=True, capture_output=True, text=True
    ).stdout
    assert snapshot.lstrip().startswith("["), snapshot
    names = [device["device_name"] for device in json.loads(snapshot)]
    assert torch.cuda.get_device_name(0) in names, snapshot


def test_cublas_matmul_matches_cpu():
    torch.manual_seed(0)
    a = torch.randn(512, 512)
    b = torch.randn(512, 512)
    torch.testing.assert_close((a.cuda() @ b.cuda()).cpu(), a @ b, rtol=1e-4, atol=1e-4)


def test_cudnn_convolution_matches_cpu():
    torch.manual_seed(0)
    conv = torch.nn.Conv2d(8, 16, kernel_size=3, padding=1)
    x = torch.randn(4, 8, 32, 32)
    expected = conv(x)
    with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
        actual = conv.cuda()(x.cuda()).cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_jit_compiled_cuda_kernel_runs(tmp_path):
    major, minor = torch.cuda.get_device_capability()
    module = load_inline(
        name="miles_gpu_add_one",
        cpp_sources="void run(torch::Tensor values);",
        cuda_sources=ADD_ONE_SOURCE,
        functions=["run"],
        extra_cuda_cflags=[f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"],
        build_directory=str(tmp_path),
    )
    values = torch.arange(1000, dtype=torch.float32, device="cuda")
    module.run(values)
    torch.testing.assert_close(values.cpu(), torch.arange(1000, dtype=torch.float32) + 1)


def test_triton_kernel_compiles_and_runs():
    torch.manual_seed(0)
    x = torch.rand(10_000, device="cuda")
    y = torch.rand(10_000, device="cuda")
    out = torch.empty_like(x)
    _add_kernel[(triton.cdiv(x.numel(), 1024),)](x, y, out, x.numel(), BLOCK=1024)
    torch.testing.assert_close(out, x + y)


def test_torch_compile_generates_gpu_kernels(tmp_path, monkeypatch):
    # A fresh cache forces TorchInductor to generate and compile Triton kernels.
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))
    torch._dynamo.reset()

    def fn(x):
        return torch.nn.functional.gelu(x) * x.sum(dim=-1, keepdim=True)

    x = torch.randn(64, 256, device="cuda")
    torch.testing.assert_close(torch.compile(fn)(x), fn(x), rtol=1e-4, atol=1e-4)
