"""Verify numba can compile CUDA kernels — cupy image.

The cupy image advertises Numba, and `numba.cuda` compiles kernels through libNVVM.
The CUDA runtime base does not ship libNVVM, so without it every `@cuda.jit` raises
NvvmSupportError at compile time. `compile_ptx` exercises that compiler path with no
GPU required; the end-to-end kernel launch is covered in single_gpu/test_cupy_cuda.py.
"""

from numba import cuda, types


def test_compile_kernel_to_ptx():
    def increment(x):
        i = cuda.grid(1)
        if i < x.size:
            x[i] += 1

    ptx, _ = cuda.compile_ptx(increment, (types.float32[:],), cc=(8, 0))
    assert ".visible .entry" in ptx
