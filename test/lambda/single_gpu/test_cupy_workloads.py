"""Validate CuPy numerical workloads on a single GPU — cupy image."""

import cupy as cp
import numpy as np


def test_monte_carlo_var():
    """Monte Carlo VaR — result must be negative (loss at tail)."""
    cp.random.seed(42)
    returns = cp.random.randn(1_000_000, 50, dtype=cp.float32) * 0.01
    weights = cp.ones(50, dtype=cp.float32) / 50
    portfolio_returns = cp.dot(returns, weights)
    var_95 = float(cp.percentile(portfolio_returns, 5))
    assert var_95 < 0, "VaR(95%) should be negative"


def test_portfolio_optimization():
    """CVXPY portfolio optimization converges to optimal."""
    import cvxpy as cvx

    np.random.seed(42)
    n = 50
    mu = np.random.randn(n) * 0.01 + 0.05
    Sigma = np.eye(n) * 0.01
    w = cvx.Variable(n)
    prob = cvx.Problem(
        cvx.Minimize(cvx.quad_form(w, Sigma)), [cvx.sum(w) == 1, w >= 0, mu @ w >= 0.06]
    )
    prob.solve()
    assert prob.status == "optimal"
    assert float(mu @ w.value) >= 0.059  # within solver tolerance


def test_numba_jit():
    """Numba JIT compiles and produces correct result."""
    from numba import jit

    @jit(nopython=True)
    def monte_carlo_pi(n, seed):
        np.random.seed(seed)
        inside = 0
        for _ in range(n):
            x = np.random.random()
            y = np.random.random()
            if x * x + y * y <= 1.0:
                inside += 1
        return 4.0 * inside / n

    pi_estimate = monte_carlo_pi(100_000, 42)
    assert abs(pi_estimate - 3.14159265) < 0.05


def test_scipy_optimization():
    """SciPy Rosenbrock minimization converges."""
    from scipy.optimize import minimize

    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    result = minimize(
        lambda x: sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0),
        x0,
        method="BFGS",
    )
    assert result.success
    assert result.fun < 1e-4
