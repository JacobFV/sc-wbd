"""Numerical correctness of the solvers.

These are the tests that make the rest of the module trustworthy: an observed
convergence order, a strong SDE order, an energy-drift comparison, and the
statement that Heun integrates the *Stratonovich* SDE (a modelling fact, not a
numerical detail).
"""

from __future__ import annotations

import math

import pytest
import torch

from conftest import order_estimate
from scwbd.dynamics.integrators import (
    BrownianPath,
    euler_maruyama,
    heun,
    integrate,
    milstein,
    stochastic_rk,
)
from scwbd.dynamics.types import assert_solver_dtype


def _roll(step, f, x0, dt, n, g=None, dW=None):
    x = x0
    t = 0.0
    for k in range(n):
        x = step(f, g, x, t, dt, None if dW is None else dW[k])
        t += dt
    return x


# ---------------------------------------------------------------------------
# Deterministic convergence order
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,expected", [(euler_maruyama, 1.0), (heun, 2.0), (stochastic_rk, 2.0)]
)
def test_deterministic_convergence_order_linear(method, expected, device):
    """Global error vs dt on a linear system with an exact matrix-exponential solution."""
    torch.manual_seed(0)
    A = torch.tensor([[-1.5, 0.7], [-0.4, -2.0]], dtype=torch.float64, device=device)
    x0 = torch.tensor([[1.0, -0.5]], dtype=torch.float64, device=device)
    T = 1.0
    exact = (torch.matrix_exp(A * T) @ x0.T).T

    def f(x, t):
        return x @ A.T

    errs, factors = [], []
    for n in (100, 200, 400, 800):
        dt = T / n
        xf = _roll(method, f, x0, dt, n)
        errs.append(float((xf - exact).abs().max()))
        factors.append(dt)
    p = order_estimate(errs, factors)
    assert p == pytest.approx(expected, abs=0.15), f"observed order {p:.3f}, expected {expected}"


@pytest.mark.parametrize("method,expected", [(euler_maruyama, 1.0), (heun, 2.0)])
def test_deterministic_convergence_order_nonlinear(method, expected, device):
    """Logistic equation: exact solution known, genuinely nonlinear."""
    r, x0v, T = 1.7, 0.1, 2.0
    x0 = torch.tensor([[x0v]], dtype=torch.float64, device=device)
    exact = x0v * math.exp(r * T) / (1 - x0v + x0v * math.exp(r * T))

    def f(x, t):
        return r * x * (1 - x)

    errs, factors = [], []
    for n in (50, 100, 200, 400):
        dt = T / n
        xf = _roll(method, f, x0, dt, n)
        errs.append(abs(float(xf) - exact))
        factors.append(dt)
    p = order_estimate(errs, factors)
    assert p == pytest.approx(expected, abs=0.15), f"observed order {p:.3f}"


# ---------------------------------------------------------------------------
# Strong stochastic order
# ---------------------------------------------------------------------------


def _gbm_setup(device, n_fine, dt_fine, seed=7):
    mu, sigma, x0v = 0.5, 0.6, 1.0
    path = BrownianPath((256, 1), n_fine, dt_fine, seed=seed, device=device, dtype=torch.float64)
    x0 = torch.full((256, 1), x0v, dtype=torch.float64, device=device)
    return mu, sigma, x0, path


def test_euler_maruyama_strong_order_half_multiplicative(device):
    """EM has strong order 0.5 for multiplicative noise (geometric Brownian motion)."""
    T, n_fine, dt_fine = 1.0, 4096, 1.0 / 4096
    mu, sigma, x0, path = _gbm_setup(device, n_fine, dt_fine)
    W_T = path.total()
    exact_ito = x0 * torch.exp((mu - 0.5 * sigma**2) * T + sigma * W_T)

    def f(x, t):
        return mu * x

    def g(x, t):
        return sigma * x

    errs, factors = [], []
    for factor in (32, 64, 128, 256):
        dW, dt = path.coarsen(factor)
        xf = _roll(euler_maruyama, f, x0, dt, dW.shape[0], g, dW)
        errs.append(float((xf - exact_ito).abs().mean()))
        factors.append(dt)
    p = order_estimate(errs, factors)
    assert 0.35 < p < 0.75, f"EM strong order on multiplicative noise = {p:.3f}, expected ~0.5"


def test_milstein_strong_order_one_multiplicative(device):
    """Milstein recovers strong order 1.0 on the same problem."""
    T, n_fine, dt_fine = 1.0, 4096, 1.0 / 4096
    mu, sigma, x0, path = _gbm_setup(device, n_fine, dt_fine)
    exact_ito = x0 * torch.exp((mu - 0.5 * sigma**2) * T + sigma * path.total())

    def f(x, t):
        return mu * x

    def g(x, t):
        return sigma * x

    errs, factors = [], []
    for factor in (32, 64, 128, 256):
        dW, dt = path.coarsen(factor)
        xf = _roll(milstein, f, x0, dt, dW.shape[0], g, dW)
        errs.append(float((xf - exact_ito).abs().mean()))
        factors.append(dt)
    p = order_estimate(errs, factors)
    assert p > 0.8, f"Milstein strong order = {p:.3f}, expected ~1.0"


def test_heun_integrates_the_stratonovich_sde(device):
    """Heun converges to the **Stratonovich** solution, not the Itô one.

    This is a modelling fact with consequences: mixing Heun with an Itô-specified
    diffusion silently changes the model by the Itô–Stratonovich drift
    correction ``0.5 g g'``.
    """
    T, n_fine, dt_fine = 1.0, 4096, 1.0 / 4096
    mu, sigma, x0, path = _gbm_setup(device, n_fine, dt_fine)
    W_T = path.total()
    exact_ito = x0 * torch.exp((mu - 0.5 * sigma**2) * T + sigma * W_T)
    exact_strat = x0 * torch.exp(mu * T + sigma * W_T)

    def f(x, t):
        return mu * x

    def g(x, t):
        return sigma * x

    dW, dt = path.coarsen(32)
    xf = _roll(heun, f, x0, dt, dW.shape[0], g, dW)
    err_strat = float((xf - exact_strat).abs().mean())
    err_ito = float((xf - exact_ito).abs().mean())
    assert err_strat < 0.25 * err_ito, (
        f"Heun should track the Stratonovich solution (err={err_strat:.4g}) "
        f"rather than the Ito one (err={err_ito:.4g})"
    )


def test_additive_noise_all_schemes_agree(device):
    """With additive noise Itô and Stratonovich coincide; every scheme converges."""
    T, n_fine, dt_fine = 1.0, 2048, 1.0 / 2048
    theta, sigma = 2.0, 0.5
    path = BrownianPath((128, 1), n_fine, dt_fine, seed=11, device=device, dtype=torch.float64)
    x0 = torch.zeros(128, 1, dtype=torch.float64, device=device)

    def f(x, t):
        return -theta * x

    def g(x, t):
        return torch.full_like(x, sigma)

    ref = _roll(euler_maruyama, f, x0, dt_fine, n_fine, g, path.dW)
    dW, dt = path.coarsen(16)
    outs = {
        name: _roll(m, f, x0, dt, dW.shape[0], g, dW)
        for name, m in [("euler", euler_maruyama), ("heun", heun), ("milstein", milstein), ("srk", stochastic_rk)]
    }
    for name, xf in outs.items():
        err = float((xf - ref).abs().mean())
        assert err < 5e-3, f"{name} additive-noise error {err:.4g} too large"


# ---------------------------------------------------------------------------
# Stability, conservation, determinism
# ---------------------------------------------------------------------------


def test_energy_drift_harmonic_oscillator(device):
    """Neither scheme is symplectic, but Heun's energy drift is O(dt^2) smaller."""
    omega, T = 2 * math.pi, 4.0
    x0 = torch.tensor([[1.0, 0.0]], dtype=torch.float64, device=device)

    def f(x, t):
        q, p = x[..., 0:1], x[..., 1:2]
        return torch.cat([p, -(omega**2) * q], dim=-1)

    def energy(x):
        return float(0.5 * (x[..., 1] ** 2 + omega**2 * x[..., 0] ** 2))

    e0 = energy(x0)
    n = 2000
    dt = T / n
    de_euler = abs(energy(_roll(euler_maruyama, f, x0, dt, n)) - e0) / e0
    de_heun = abs(energy(_roll(heun, f, x0, dt, n)) - e0) / e0
    assert de_heun < de_euler / 10, f"heun drift {de_heun:.3e} vs euler {de_euler:.3e}"
    assert de_heun < 1e-3


def test_stability_stiff_decay_bounded(device):
    """Inside the stability region the solution decays; outside, Euler blows up."""
    lam = -50.0
    x0 = torch.ones(1, 1, dtype=torch.float64, device=device)

    def f(x, t):
        return lam * x

    stable = _roll(euler_maruyama, f, x0, 0.01, 200)  # |1 + lam dt| = 0.5
    unstable = _roll(euler_maruyama, f, x0, 0.05, 40)  # |1 + lam dt| = 1.5
    assert float(stable.abs()) < 1e-6
    assert float(unstable.abs()) > 1.0


def test_seed_determinism_and_independence(device):
    """Same seed -> bit-identical; different seed -> different."""
    x0 = torch.zeros(4, 3, dtype=torch.float32, device=device)

    def f(x, t):
        return -x

    def g(x, t):
        return torch.full_like(x, 0.3)

    a, _ = integrate(f, g, x0, dt=1e-3, n_steps=50, method="heun", seed=123)
    b, _ = integrate(f, g, x0, dt=1e-3, n_steps=50, method="heun", seed=123)
    c, _ = integrate(f, g, x0, dt=1e-3, n_steps=50, method="heun", seed=124)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_stochastic_integration_requires_explicit_seed(device):
    x0 = torch.zeros(2, 2, device=device)
    with pytest.raises(ValueError, match="explicit seed"):
        integrate(lambda x, t: -x, lambda x, t: torch.full_like(x, 0.1), x0, dt=1e-3, n_steps=5)


def test_solver_refuses_bfloat16(device):
    """bf16 is permitted inside learned operators, never in a solver (§3)."""
    x = torch.zeros(2, 2, dtype=torch.bfloat16, device=device)
    with pytest.raises(TypeError, match="float32"):
        assert_solver_dtype(x)
    with pytest.raises(TypeError):
        euler_maruyama(lambda xx, t: xx, None, x, 0.0, 1e-3)


def test_brownian_refinement_property(device):
    """Coarsened increments sum correctly — the property the order tests rely on."""
    path = BrownianPath((5,), 64, 1e-3, seed=3, device=device, dtype=torch.float64)
    dW2, dt2 = path.coarsen(2)
    assert dt2 == pytest.approx(2e-3)
    assert torch.allclose(dW2[0], path.dW[0] + path.dW[1])
    assert torch.allclose(dW2.sum(0), path.dW.sum(0))
