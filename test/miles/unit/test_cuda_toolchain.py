"""Verify the image can JIT-compile and link a CUDA extension without a GPU."""

import os

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
