"""Validate CuPy CUDA operations on a single GPU — cupy image."""

import cupy as cp


def test_cuda_available():
    assert cp.cuda.is_available()


def test_device_count():
    assert cp.cuda.runtime.getDeviceCount() >= 1


def test_gpu_array_ops():
    a = cp.array([1, 2, 3])
    b = cp.array([4, 5, 6])
    c = a + b
    assert cp.array_equal(c, cp.array([5, 7, 9]))


def test_matrix_multiply():
    cp.random.seed(42)
    A = cp.random.randn(1000, 1000, dtype=cp.float32)
    B = cp.random.randn(1000, 1000, dtype=cp.float32)
    C = cp.dot(A, B)
    assert C.shape == (1000, 1000)
    assert cp.isfinite(C).all()


def test_matrix_inversion():
    cp.random.seed(42)
    # Diagonally dominant matrix — guaranteed well-conditioned
    A = cp.eye(500, dtype=cp.float64) * 10 + cp.random.randn(500, 500, dtype=cp.float64)
    A_inv = cp.linalg.inv(A)
    identity = cp.dot(A, A_inv)
    assert cp.allclose(identity, cp.eye(500), atol=1e-5)


def test_svd():
    cp.random.seed(42)
    A = cp.random.randn(500, 300, dtype=cp.float32)
    U, s, Vt = cp.linalg.svd(A, full_matrices=False)
    assert U.shape == (500, 300)
    assert s.shape == (300,)
    assert Vt.shape == (300, 300)
    # Singular values must be non-negative and sorted descending
    assert (s >= 0).all()
    assert (cp.diff(s) <= 0).all()


def test_fft():
    n = 1024
    signal = cp.sin(2 * cp.pi * 5 * cp.linspace(0, 1, n))
    fft_result = cp.fft.fft(signal)
    assert fft_result.shape == (n,)
    assert cp.isfinite(fft_result).all()
