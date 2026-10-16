"""Verify the image's JIT toolchains build kernels without a GPU."""

import os
import subprocess

import torch
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE = r"""
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

__global__ void kernel(float* values) {
  values[threadIdx.x] += 1.0f;
}

void run(torch::Tensor tensor) {
  kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(tensor.data_ptr<float>());
}
"""


def test_cuda_extension_compiles_and_links_nccl(tmp_path):
    module = load_inline(
        name="miles_cuda_jit_smoke",
        cpp_sources="void run(torch::Tensor tensor);",
        cuda_sources=CUDA_SOURCE,
        functions=["run"],
        with_cuda=True,
        extra_cuda_cflags=["-gencode=arch=compute_90,code=sm_90"],
        extra_ldflags=["-lnccl"],
        build_directory=str(tmp_path),
    )
    assert os.path.isfile(module.__file__)


def test_nvcc_builds_a_cuda_program(tmp_path):
    # nvcc links the CUDA runtime statically by default.
    source = tmp_path / "program.cu"
    source.write_text("__global__ void kernel() {}\nint main() { kernel<<<1, 1>>>(); }\n")
    subprocess.run(
        [
            "nvcc",
            "-gencode=arch=compute_90,code=sm_90",
            str(source),
            "-o",
            str(tmp_path / "program"),
        ],
        check=True,
    )
    assert (tmp_path / "program").is_file()


def test_torch_compile_builds_and_runs_a_cpu_kernel(tmp_path, monkeypatch):
    # A fresh cache forces TorchInductor's C++ codegen to build the kernel.
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))
    compiled = torch.compile(lambda values: torch.relu(values).sum())
    assert compiled(torch.tensor([-1.0, 2.0, 3.0])).item() == 5.0
