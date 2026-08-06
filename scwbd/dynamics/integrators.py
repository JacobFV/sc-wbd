"""SDE/ODE integrators, the semigroup residual, and the R06 adaptive-step guard.

Contents
--------
* ``euler_maruyama`` (strong order 0.5 multiplicative / 1.0 additive; drift order 1)
* ``heun`` (deterministic order 2; Stratonovich stochastic Heun, strong order 1
  for additive noise)
* ``milstein`` (strong order 1.0 for diagonal noise, using the exact derivative
  of ``g`` via autograd-free finite differences on the diagonal)
* ``stochastic_rk`` — Rößler-style SRK, weak order 2 / strong order 1 for
  diagonal noise
* ``BrownianPath`` — a refinable Brownian path so strong convergence order can
  actually be *measured* (the same path at several step sizes)
* ``AdaptiveStepper`` — embedded Heun/Euler PI controller that **refuses** to run
  for a learned propagator without a passing semigroup certificate (R06)
* ``semigroup_residual`` / ``certify_semigroup`` — thesis §4.5,
  ``eps_sg(d1,d2;x) = ||Phi^{d1+d2}(x) - Phi^{d2}(Phi^{d1}(x))||_W``

Solvers stay in fp32/fp64 (ARCHITECTURE.md §3); every entry point checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import torch
from torch import Tensor

from .types import (
    DTYPE,
    GuardViolation,
    NumericalBudget,
    SemigroupRefusal,
    assert_solver_dtype,
    make_generator,
)

__all__ = [
    "DriftFn",
    "DiffusionFn",
    "euler_maruyama",
    "heun",
    "milstein",
    "stochastic_rk",
    "INTEGRATORS",
    "get_integrator",
    "BrownianPath",
    "integrate",
    "AdaptiveStepper",
    "semigroup_residual",
    "certify_semigroup",
    "SemigroupCertificate",
    "SemigroupGuard",
]


class DriftFn(Protocol):
    def __call__(self, x: Tensor, t: float) -> Tensor: ...


class DiffusionFn(Protocol):
    def __call__(self, x: Tensor, t: float) -> Tensor: ...


# ---------------------------------------------------------------------------
# Steppers
# ---------------------------------------------------------------------------


def euler_maruyama(
    f: DriftFn, g: DiffusionFn | None, x: Tensor, t: float, dt: float, dW: Tensor | None = None
) -> Tensor:
    """``x + f dt + g dW``.  Deterministic order 1; strong order 0.5 (1.0 additive)."""
    assert_solver_dtype(x)
    out = x + f(x, t) * dt
    if g is not None and dW is not None:
        out = out + g(x, t) * dW
    return out


def heun(
    f: DriftFn, g: DiffusionFn | None, x: Tensor, t: float, dt: float, dW: Tensor | None = None
) -> Tensor:
    """Predictor–corrector.  Deterministic order 2; Stratonovich stochastic Heun.

    For **additive** noise (``g`` independent of ``x``) Stratonovich and Itô
    coincide and this is strong order 1.0.  For state-dependent ``g`` the result
    converges to the *Stratonovich* solution — a different SDE from the Itô one
    Euler–Maruyama integrates.  That is a modelling decision, not a numerical
    detail, so it is stated here rather than buried.
    """
    assert_solver_dtype(x)
    f0 = f(x, t)
    if g is None or dW is None:
        x_pred = x + f0 * dt
        return x + 0.5 * dt * (f0 + f(x_pred, t + dt))
    g0 = g(x, t)
    x_pred = x + f0 * dt + g0 * dW
    f1 = f(x_pred, t + dt)
    g1 = g(x_pred, t + dt)
    return x + 0.5 * dt * (f0 + f1) + 0.5 * (g0 + g1) * dW


def milstein(
    f: DriftFn,
    g: DiffusionFn | None,
    x: Tensor,
    t: float,
    dt: float,
    dW: Tensor | None = None,
    *,
    fd_eps: float = 1e-4,
) -> Tensor:
    """Itô–Milstein for **diagonal** noise: strong order 1.0.

    The Milstein correction ``0.5 g g' (dW^2 - dt)`` needs ``dg/dx`` on the
    diagonal; it is obtained by a central finite difference so the scheme works
    for non-differentiable/learned ``g`` too.  With additive noise ``g' = 0`` and
    this reduces to Euler–Maruyama, as it should.
    """
    assert_solver_dtype(x)
    out = x + f(x, t) * dt
    if g is None or dW is None:
        return out
    g0 = g(x, t)
    out = out + g0 * dW
    h = fd_eps * (1.0 + x.abs())
    gp = (g(x + h, t) - g(x - h, t)) / (2.0 * h)
    return out + 0.5 * g0 * gp * (dW * dW - dt)


def stochastic_rk(
    f: DriftFn, g: DiffusionFn | None, x: Tensor, t: float, dt: float, dW: Tensor | None = None
) -> Tensor:
    """Rößler SRI-type scheme for diagonal noise (weak order 2, strong order 1).

    Deterministic part is the classical RK2 midpoint (order 2); the stochastic
    part uses the two-stage diagonal construction with the supporting value
    ``x +/- g sqrt(dt)``.
    """
    assert_solver_dtype(x)
    if g is None or dW is None:
        k1 = f(x, t)
        k2 = f(x + 0.5 * dt * k1, t + 0.5 * dt)
        return x + dt * k2
    sdt = math.sqrt(dt)
    g0 = g(x, t)
    k1 = f(x, t)
    x_mid = x + 0.5 * dt * k1 + 0.5 * g0 * dW
    k2 = f(x_mid, t + 0.5 * dt)
    x_plus = x + k1 * dt + g0 * sdt
    x_minus = x + k1 * dt - g0 * sdt
    g_plus, g_minus = g(x_plus, t + dt), g(x_minus, t + dt)
    stoch = g0 * dW + 0.25 * (g_plus + g_minus - 2.0 * g0) * dW
    stoch = stoch + 0.25 * (g_plus - g_minus) * (dW * dW - dt) / sdt
    return x + dt * k2 + stoch


INTEGRATORS: dict[str, Callable[..., Tensor]] = {
    "euler_maruyama": euler_maruyama,
    "euler": euler_maruyama,
    "heun": heun,
    "milstein": milstein,
    "srk": stochastic_rk,
    "stochastic_rk": stochastic_rk,
}

#: nominal orders, used by the adaptive controller and asserted in the tests
INTEGRATOR_ORDERS: dict[str, tuple[float, float]] = {
    # name -> (deterministic order, strong SDE order for additive noise)
    "euler_maruyama": (1.0, 1.0),
    "heun": (2.0, 1.0),
    "milstein": (1.0, 1.0),
    "stochastic_rk": (2.0, 1.0),
}


def get_integrator(name: str) -> Callable[..., Tensor]:
    if name not in INTEGRATORS:
        raise KeyError(f"unknown integrator {name!r}; available: {sorted(INTEGRATORS)}")
    return INTEGRATORS[name]


# ---------------------------------------------------------------------------
# Brownian paths
# ---------------------------------------------------------------------------


class BrownianPath:
    """A fixed Brownian path that can be read at several step sizes.

    Strong convergence order is only measurable if the *same* driving noise is
    used at every step size.  This class stores increments at the finest level
    and sums consecutive increments for coarser levels, which is exactly the
    Brownian refinement property ``W(t+2h) - W(t) = dW_1 + dW_2``.
    """

    def __init__(
        self,
        shape: Sequence[int],
        n_steps: int,
        dt: float,
        *,
        seed: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = DTYPE,
    ):
        g = make_generator(seed, device)
        self.dt = float(dt)
        self.n_steps = int(n_steps)
        self.shape = tuple(shape)
        self.dW = math.sqrt(dt) * torch.randn(
            (n_steps, *shape), generator=g, device=g.device, dtype=dtype
        )

    def coarsen(self, factor: int) -> tuple[Tensor, float]:
        """Increments and step size at ``factor``x coarser resolution."""
        if self.n_steps % factor:
            raise ValueError(f"n_steps={self.n_steps} not divisible by factor={factor}")
        n = self.n_steps // factor
        dW = self.dW.reshape(n, factor, *self.shape).sum(dim=1)
        return dW, self.dt * factor

    def total(self) -> Tensor:
        return self.dW.sum(dim=0)


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


def integrate(
    f: DriftFn,
    g: DiffusionFn | None,
    x0: Tensor,
    *,
    dt: float,
    n_steps: int,
    method: str = "heun",
    seed: int | None = None,
    dW: Tensor | None = None,
    t0: float = 0.0,
    record_every: int = 1,
    post_step: Callable[[Tensor, int, float], Tensor] | None = None,
    store: bool = True,
) -> tuple[Tensor, Tensor]:
    """Fixed-step rollout.  Returns ``(trajectory, x_final)``.

    ``trajectory`` has shape ``(n_recorded, *x0.shape)`` (empty if ``store=False``).
    ``post_step`` is where the coupling buffer push / state clamping happens.
    """
    assert_solver_dtype(x0)
    step = get_integrator(method)
    x = x0
    if g is not None and dW is None:
        if seed is None:
            raise ValueError("stochastic integration requires an explicit seed (ARCHITECTURE.md §3)")
        path = BrownianPath(tuple(x0.shape), n_steps, dt, seed=seed, device=x0.device, dtype=x0.dtype)
        dW = path.dW
    out: list[Tensor] = []
    t = t0
    for k in range(n_steps):
        x = step(f, g, x, t, dt, None if dW is None else dW[k])
        t += dt
        if post_step is not None:
            x = post_step(x, k, t)
        if store and (k % record_every == 0):
            out.append(x)
    traj = torch.stack(out) if out else x.new_zeros((0, *x.shape))
    return traj, x


# ---------------------------------------------------------------------------
# Semigroup residual (thesis §4.5, refusal R06)
# ---------------------------------------------------------------------------


def semigroup_residual(
    propagator: Callable[[Tensor, float], Tensor],
    x: Tensor,
    d1: float,
    d2: float,
    *,
    W: Tensor | None = None,
    relative: bool = True,
    eps: float = 1e-12,
) -> Tensor:
    """``eps_sg(d1,d2;x) = || Phi^{d1+d2}(x) - Phi^{d2}(Phi^{d1}(x)) ||_W``.

    Returned per sample ``(B,)``.  With ``relative=True`` the residual is
    normalised by the norm of the composed update's displacement, which makes
    the tolerance interpretable across state scales.
    """
    direct = propagator(x, d1 + d2)
    composed = propagator(propagator(x, d1), d2)
    diff = direct - composed
    if W is not None:
        diff = diff * W
    B = x.shape[0]
    num = diff.reshape(B, -1).norm(dim=-1)
    if not relative:
        return num
    scale = (composed - x).reshape(B, -1).norm(dim=-1)
    return num / (scale + eps)


@dataclass
class SemigroupCertificate:
    """The evidence that a propagator may be stepped adaptively (R06).

    ``pairs`` are the permitted ``(d1, d2)`` step pairs actually tested;
    ``residuals`` holds their per-pair summary statistic.  ``ok`` is the
    per-pair verdict: adaptive stepping and coarse-for-fine substitution are
    permitted **only** for pairs that passed.
    """

    pairs: list[tuple[float, float]]
    residual_max: list[float]
    residual_mean: list[float]
    tolerance: float
    ok: list[bool]
    propagator: str = ""
    n_samples: int = 0

    @property
    def passed(self) -> bool:
        return bool(self.ok) and all(self.ok)

    @property
    def worst(self) -> float:
        return max(self.residual_max) if self.residual_max else 0.0

    def permits(self, d1: float, d2: float, *, rtol: float = 1e-6) -> bool:
        for (a, b), good in zip(self.pairs, self.ok):
            if abs(a - d1) <= rtol * max(a, 1.0) and abs(b - d2) <= rtol * max(b, 1.0):
                return good
        return False

    def failing_pairs(self) -> list[tuple[float, float]]:
        return [p for p, good in zip(self.pairs, self.ok) if not good]

    def budget_contribution(self) -> float:
        """Below tolerance, the residual distribution enters the numerical budget.

        Reported as a variance (mean squared residual over passing pairs), per
        thesis §4.5: "below tolerance, its empirical distribution enters the
        numerical and model-discrepancy budget".
        """
        vals = [m for m, good in zip(self.residual_mean, self.ok) if good]
        return float(sum(v * v for v in vals) / max(len(vals), 1))

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": "R06",
            "propagator": self.propagator,
            "tolerance": self.tolerance,
            "pairs": [list(p) for p in self.pairs],
            "residual_max": list(self.residual_max),
            "residual_mean": list(self.residual_mean),
            "ok": list(self.ok),
            "passed": self.passed,
            "worst": self.worst,
            "n_samples": self.n_samples,
            "numerical_budget_variance": self.budget_contribution(),
        }


def certify_semigroup(
    propagator: Callable[[Tensor, float], Tensor],
    x: Tensor,
    steps: Sequence[float],
    *,
    tolerance: float = 1e-2,
    W: Tensor | None = None,
    pairs: Sequence[tuple[float, float]] | None = None,
    relative: bool = True,
    name: str = "",
    reduce: str = "max",
) -> SemigroupCertificate:
    """Measure the semigroup residual at **every permitted step pair**.

    ``steps`` is the set of admissible step sizes; by default all ordered pairs
    are tested (that is what "every permitted pair of steps" means in §4.5).
    """
    test_pairs = list(pairs) if pairs is not None else [(a, b) for a in steps for b in steps]
    res_max: list[float] = []
    res_mean: list[float] = []
    ok: list[bool] = []
    with torch.no_grad():
        for d1, d2 in test_pairs:
            eps = semigroup_residual(propagator, x, d1, d2, W=W, relative=relative)
            res_max.append(float(eps.max()))
            res_mean.append(float(eps.mean()))
            stat = res_max[-1] if reduce == "max" else res_mean[-1]
            ok.append(stat <= tolerance)
    return SemigroupCertificate(
        pairs=test_pairs,
        residual_max=res_max,
        residual_mean=res_mean,
        tolerance=float(tolerance),
        ok=ok,
        propagator=name,
        n_samples=int(x.shape[0]),
    )


class SemigroupGuard:
    """Gatekeeper for adaptive stepping and coarse-for-fine substitution (R06).

    A propagator declared ``learned=True`` may not be stepped adaptively, and a
    coarse step may not replace several fine steps, unless a
    :class:`SemigroupCertificate` covering that step pair has passed.
    """

    def __init__(
        self,
        certificate: SemigroupCertificate | None = None,
        *,
        learned: bool = True,
        owner: str = "propagator",
        budget: NumericalBudget | None = None,
    ):
        self.certificate = certificate
        self.learned = learned
        self.owner = owner
        self.budget = budget
        self.violations: list[GuardViolation] = []
        if certificate is not None and budget is not None:
            budget.add(
                "semigroup_residual",
                certificate.budget_contribution(),
                f"semigroup residual of {owner} below tolerance {certificate.tolerance:g}",
            )

    def _refuse(self, detail: str, value: float, tol: float) -> None:
        v = GuardViolation(
            code="R06",
            detail=detail,
            value=value,
            tolerance=tol,
            offending_object=self.owner,
            remedy=(
                "restrict the scheduler to certified step pairs, refine the step, or carry the "
                "discrepancy in the prediction interval"
            ),
        )
        self.violations.append(v)
        raise SemigroupRefusal(v)

    def check_adaptive(self) -> None:
        if not self.learned:
            return
        if self.certificate is None:
            self._refuse(
                "adaptive stepping requested for a learned propagator with no semigroup certificate",
                float("inf"),
                0.0,
            )
        if not self.certificate.passed:
            self._refuse(
                "semigroup residual above tolerance on "
                f"{len(self.certificate.failing_pairs())} step pair(s); adaptive stepping disabled",
                self.certificate.worst,
                self.certificate.tolerance,
            )

    def check_substitution(self, coarse: float, fine_steps: Sequence[float]) -> None:
        """Permit replacing several fine steps by one coarse step?"""
        if not self.learned:
            return
        if self.certificate is None:
            self._refuse(
                "coarse-for-fine substitution requested for a learned propagator without a certificate",
                float("inf"),
                0.0,
            )
        if not fine_steps:
            raise ValueError("fine_steps must be non-empty")
        if abs(sum(fine_steps) - coarse) > 1e-9 * max(coarse, 1.0):
            raise ValueError(
                f"the fine steps {list(fine_steps)} do not sum to the coarse step {coarse:g}"
            )
        # every prefix split of the coarse step must be certified: substituting
        # one coarse update for k fine updates is exactly a chain of semigroup
        # compositions, so a single certified pair is not sufficient.
        acc = 0.0
        for s in fine_steps[:-1]:
            acc += s
            d1, d2 = acc, coarse - acc
            if not self.certificate.permits(d1, d2):
                self._refuse(
                    f"step pair ({d1:g}, {d2:g}) is not certified; "
                    "coarse-for-fine substitution disabled",
                    self.certificate.worst,
                    self.certificate.tolerance,
                )

    def permitted_steps(self) -> list[float]:
        if self.certificate is None:
            return []
        good = {p[0] for p, o in zip(self.certificate.pairs, self.certificate.ok) if o}
        return sorted(good)


# ---------------------------------------------------------------------------
# Adaptive stepping
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveResult:
    x: Tensor
    t: float
    n_steps: int
    n_rejected: int
    dt_final: float
    dt_min_used: float
    dt_max_used: float
    error_history: list[float] = field(default_factory=list)


class AdaptiveStepper:
    """Embedded Heun/Euler adaptive stepper with a PI controller.

    Deterministic (ODE) mode only: adaptive step-size control under Brownian
    noise requires a refinable path and changes the law of the discretisation,
    so it is refused rather than silently approximated.  Set
    ``allow_stochastic=True`` only with a Brownian tree, which this module does
    not provide.

    R06: if ``guard`` is supplied and the propagator is learned, adaptivity is
    checked against the semigroup certificate *before the first step*.
    """

    def __init__(
        self,
        rtol: float = 1e-4,
        atol: float = 1e-6,
        *,
        dt_min: float = 1e-6,
        dt_max: float = 1e-1,
        safety: float = 0.9,
        guard: SemigroupGuard | None = None,
        max_rejects: int = 32,
    ):
        self.rtol, self.atol = float(rtol), float(atol)
        self.dt_min, self.dt_max = float(dt_min), float(dt_max)
        self.safety = float(safety)
        self.guard = guard
        self.max_rejects = int(max_rejects)

    def _error(self, x_low: Tensor, x_high: Tensor) -> float:
        scale = self.atol + self.rtol * torch.maximum(x_low.abs(), x_high.abs())
        return float(((x_high - x_low) / scale).pow(2).mean().sqrt())

    def run(
        self,
        f: DriftFn,
        x0: Tensor,
        *,
        t_end: float,
        dt_init: float = 1e-3,
        t0: float = 0.0,
    ) -> AdaptiveResult:
        assert_solver_dtype(x0)
        if self.guard is not None:
            self.guard.check_adaptive()  # R06 fires here, before any integration
        x, t, dt = x0, float(t0), float(dt_init)
        n, rej = 0, 0
        errs: list[float] = []
        dt_lo, dt_hi = dt, dt
        while t < t_end - 1e-15:
            dt = min(dt, t_end - t)
            x_low = euler_maruyama(f, None, x, t, dt)
            x_high = heun(f, None, x, t, dt)
            err = self._error(x_low, x_high)
            errs.append(err)
            if err <= 1.0 or dt <= self.dt_min * (1 + 1e-12):
                x, t = x_high, t + dt
                n += 1
                dt_lo, dt_hi = min(dt_lo, dt), max(dt_hi, dt)
            else:
                rej += 1
                if rej > self.max_rejects * max(n, 1):
                    raise RuntimeError("adaptive stepper failed to converge; the system is likely stiff")
            factor = self.safety * (1.0 / max(err, 1e-16)) ** 0.5  # Heun/Euler pair: p = 2
            dt = float(min(self.dt_max, max(self.dt_min, dt * min(5.0, max(0.2, factor)))))
        return AdaptiveResult(x, t, n, rej, dt, dt_lo, dt_hi, errs)
