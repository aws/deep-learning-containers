#!/usr/bin/env python3
"""Comprehensive validation for lambda-cupy runtime."""

import os
import sys


def test_python_runtime():
    """Validate Python runtime."""
    version = sys.version_info
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    print(f"  Executable: {sys.executable}")
    print(f"  Prefix: {sys.prefix}")
    assert version >= (3, 12), "Expected Python >= 3.12"


def test_imports():
    """Test all required packages."""
    packages = {
        "cupy": "CuPy",
        "numpy": "NumPy",
        "scipy": "SciPy",
        "pandas": "Pandas",
        "numba": "Numba",
        "cvxpy": "CVXPY",
        "boto3": "boto3",
    }

    for pkg, name in packages.items():
        mod = __import__(pkg)
        version = getattr(mod, "__version__", "unknown")
        print(f"✓ {name}: {version}")


def test_cupy_cuda():
    """Test CuPy CUDA availability and basic operations."""
    import cupy as cp

    cuda_available = cp.cuda.is_available()
    print(f"  CUDA available: {cuda_available}")

    if cuda_available:
        print(f"  CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
        print(f"  Device count: {cp.cuda.runtime.getDeviceCount()}")
        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        print(f"  Device name: {props['name'].decode()}")
        print(f"  Compute capability: {device.compute_capability}")

        # GPU array operations
        a = cp.array([1, 2, 3])
        b = cp.array([4, 5, 6])
        c = a + b
        assert cp.array_equal(c, cp.array([5, 7, 9]))
        print(f"  GPU array ops: {cp.asnumpy(c).tolist()}")
    else:
        print("  ⚠ CUDA not available (CPU fallback)")
        a = cp.array([1, 2, 3])
        assert cp.array_equal(a, cp.array([1, 2, 3]))


def test_matrix_multiplication():
    """Test matrix multiplication (use case 1: Linear Algebra)."""
    import cupy as cp

    n = 1000
    A = cp.random.randn(n, n, dtype=cp.float32)
    B = cp.random.randn(n, n, dtype=cp.float32)
    C = cp.dot(A, B)

    assert C.shape == (n, n)
    print(f"  Matrix multiply {n}x{n}: shape={C.shape}, dtype={C.dtype}")


def test_matrix_inversion():
    """Test matrix inversion."""
    import cupy as cp

    n = 500
    A = cp.random.randn(n, n, dtype=cp.float64)
    A_inv = cp.linalg.inv(A)
    identity = cp.dot(A, A_inv)

    # Check if result is close to identity matrix
    is_identity = cp.allclose(identity, cp.eye(n), atol=1e-5)
    print(f"  Matrix inversion {n}x{n}: valid={is_identity}")
    assert is_identity


def test_svd():
    """Test Singular Value Decomposition."""
    import cupy as cp

    m, n = 500, 300
    A = cp.random.randn(m, n, dtype=cp.float32)
    U, s, Vt = cp.linalg.svd(A, full_matrices=False)

    assert U.shape == (m, n)
    assert s.shape == (n,)
    assert Vt.shape == (n, n)
    print(f"  SVD {m}x{n}: U={U.shape}, s={s.shape}, Vt={Vt.shape}")


def test_monte_carlo():
    """Test Monte Carlo simulation (use case 2)."""
    import cupy as cp

    num_sims = 1_000_000
    num_assets = 50

    # Generate random returns
    returns = cp.random.randn(num_sims, num_assets, dtype=cp.float32) * 0.01
    weights = cp.ones(num_assets, dtype=cp.float32) / num_assets

    # Portfolio returns
    portfolio_returns = cp.dot(returns, weights)

    # Calculate VaR
    var_95 = float(cp.percentile(portfolio_returns, 5))
    var_99 = float(cp.percentile(portfolio_returns, 1))
    mean_return = float(cp.mean(portfolio_returns))

    print(f"  Simulations: {num_sims:,}, Assets: {num_assets}")
    print(f"  Mean return: {mean_return:.6f}")
    print(f"  VaR(95%): {var_95:.6f}, VaR(99%): {var_99:.6f}")


def test_portfolio_optimization():
    """Test portfolio optimization (use case 3: Financial Risk)."""
    import cvxpy as cvx
    import numpy as np

    n_assets = 50

    # Generate sample data
    np.random.seed(42)
    mu = np.random.randn(n_assets) * 0.01 + 0.05
    Sigma = np.eye(n_assets) * 0.01

    # Optimization variables
    w = cvx.Variable(n_assets)

    # Objective: minimize risk
    risk = cvx.quad_form(w, Sigma)

    # Constraints
    constraints = [cvx.sum(w) == 1, w >= 0, mu @ w >= 0.06]

    # Solve
    prob = cvx.Problem(cvx.Minimize(risk), constraints)
    prob.solve()

    assert prob.status == "optimal"
    expected_return = float(mu @ w.value)
    portfolio_risk = float(prob.value)

    print(f"  Assets: {n_assets}, Status: {prob.status}")
    print(f"  Expected return: {expected_return:.4f}")
    print(f"  Portfolio risk: {portfolio_risk:.6f}")


def test_numba_jit():
    """Test Numba JIT compilation."""
    import numpy as np
    from numba import jit

    @jit(nopython=True)
    def monte_carlo_pi(n):
        inside = 0
        for _ in range(n):
            x = np.random.random()
            y = np.random.random()
            if x * x + y * y <= 1.0:
                inside += 1
        return 4.0 * inside / n

    n = 100_000
    pi_estimate = monte_carlo_pi(n)
    error = abs(pi_estimate - 3.14159265)

    print(f"  Monte Carlo π ({n:,} samples): {pi_estimate:.6f}, error: {error:.6f}")
    assert error < 0.1


def test_scipy_optimization():
    """Test SciPy optimization."""
    import numpy as np
    from scipy.optimize import minimize

    # Rosenbrock function
    def rosenbrock(x):
        return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)

    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    result = minimize(rosenbrock, x0, method="BFGS")

    assert result.success
    print(f"  Rosenbrock optimization: success={result.success}, iterations={result.nit}")
    print(f"  Final value: {result.fun:.6e}")


def test_pandas_operations():
    """Test Pandas data operations."""
    import numpy as np
    import pandas as pd

    # Create sample data
    df = pd.DataFrame(
        {"returns": np.random.randn(1000) * 0.01, "volume": np.random.randint(1000, 10000, 1000)}
    )

    # Statistical operations
    mean_return = df["returns"].mean()
    std_return = df["returns"].std()
    df.corr()

    print(f"  DataFrame: {df.shape}, mean={mean_return:.6f}, std={std_return:.6f}")


def test_cupy_fft():
    """Test CuPy FFT operations."""
    import cupy as cp

    n = 1024
    signal = cp.sin(2 * cp.pi * 5 * cp.linspace(0, 1, n))
    fft_result = cp.fft.fft(signal)

    assert fft_result.shape == (n,)
    print(f"  FFT: input={n}, output={fft_result.shape}")


def test_environment():
    """Test environment variables."""
    required = ["PATH", "LD_LIBRARY_PATH", "PYTHONPATH"]
    for var in required:
        val = os.environ.get(var, "NOT SET")
        print(f"  {var}: {val[:60]}...")
        assert var in os.environ

    assert os.environ.get("LAMBDA_TASK_ROOT") == "/var/task"
    assert os.environ.get("LAMBDA_RUNTIME_DIR") == "/var/runtime"
    assert os.environ.get("LANG") == "en_US.UTF-8"
    assert os.environ.get("TZ") == ":/etc/localtime"

    ld_path = os.environ["LD_LIBRARY_PATH"]
    for p in [
        "/var/lang/lib",
        "/lib64",
        "/usr/lib64",
        "/var/runtime",
        "/var/runtime/lib",
        "/var/task",
        "/var/task/lib",
        "/opt/lib",
        "/usr/local/cuda/lib64",
        "/x86_64-bottlerocket-linux-gnu/sys-root/usr/lib/nvidia",
    ]:
        assert p in ld_path, f"LD_LIBRARY_PATH missing {p}"
    print(f"  LD_LIBRARY_PATH entries: {len(ld_path.split(':'))}")
    print("✓ Environment configured")


def main():
    print("=" * 70)
    print("lambda-cupy Runtime Validation")
    print("=" * 70)

    tests = [
        ("Python Runtime", test_python_runtime),
        ("Package Imports", test_imports),
        ("CuPy CUDA", test_cupy_cuda),
        ("Matrix Multiplication", test_matrix_multiplication),
        ("Matrix Inversion", test_matrix_inversion),
        ("SVD Decomposition", test_svd),
        ("Monte Carlo Simulation", test_monte_carlo),
        ("Portfolio Optimization", test_portfolio_optimization),
        ("Numba JIT", test_numba_jit),
        ("SciPy Optimization", test_scipy_optimization),
        ("Pandas Operations", test_pandas_operations),
        ("CuPy FFT", test_cupy_fft),
        ("Environment", test_environment),
    ]

    failed = []
    for name, test_fn in tests:
        try:
            print(f"\n[{name}]")
            test_fn()
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            import traceback

            traceback.print_exc()
            failed.append(name)

    print("\n" + "=" * 70)
    if failed:
        print(f"✗ {len(failed)}/{len(tests)} test(s) failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"✓ All {len(tests)} validations passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
