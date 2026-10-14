#!/usr/bin/env python3
"""Compile and load a minimal CUDA/NCCL extension without executing a kernel."""

from __future__ import annotations

import os
import tempfile

from torch.utils.cpp_extension import load_inline


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="miles-cuda-jit-") as build_directory:
        module = load_inline(
            name="miles_cuda_jit_smoke",
            cpp_sources="void run(torch::Tensor tensor);",
            cuda_sources=r"""
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

__global__ void kernel(float* values) {
  values[threadIdx.x] += 1.0f;
}

void run(torch::Tensor tensor) {
  kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      tensor.data_ptr<float>());
}
""",
            functions=["run"],
            with_cuda=True,
            extra_cuda_cflags=[
                "-gencode=arch=compute_90,code=sm_90",
            ],
            extra_ldflags=["-lnccl"],
            build_directory=build_directory,
            verbose=True,
        )
        assert module is not None
        assert os.path.isfile(module.__file__)
        print(f"CUDA JIT toolchain validation passed: {module.__file__}")


if __name__ == "__main__":
    main()
