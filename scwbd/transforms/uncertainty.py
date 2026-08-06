"""First-order (T5), Monte-Carlo and interval propagation of measurement error.

Equation (T5) of ``thesis_contract.tex`` §0.4, restated in Appendix C:

.. math::

    \\Sigma_z \\approx J_x \\Sigma_x J_x^\\top + J_c \\Sigma_c J_c^\\top
                     + J_x \\Sigma_{xc} J_c^\\top + J_c \\Sigma_{cx} J_x^\\top,
    \\qquad
    b_z \\approx J_x b_x + J_c b_c + \\delta_f .

``x`` is the per-observation quantity, ``c`` the calibration coefficients.  The
cross terms are **mandatory** whenever calibration error is shared across
samples in a session; dropping them asserts independence and understates
aggregate uncertainty (ARCHITECTURE.md §3: "dropping ``J_x Sigma_xc J_c^T`` is
a bug, not an optimization").  :func:`independence_understatement` measures how
much, so the understatement is a reported number rather than a slogan.

Bias never merges into variance: :class:`Propagated` keeps ``bias`` (a
systematic offset, propagated linearly and carrying ``delta_f`` model
discrepancy) separate from ``cov``.

For nonlinear maps -- nonlinear registration, thresholded tractography,
acoustic focusing -- T5 is invalid.  :func:`monte_carlo_propagate` and
:func:`interval_propagate` replace it, and :func:`linearization_error` tells you
when you must switch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import torch

from .errors import CovarianceError, LinearizationInvalidError, TransformError
from .se3 import DTYPE, Pose, adjoint, exp_se3, log_se3

Array = torch.Tensor


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _as_cov(S: Any, n: int, name: str, *, symmetric: bool = True) -> Array:
    M = torch.as_tensor(S, dtype=DTYPE)
    if M.shape != (n, n):
        raise CovarianceError(
            f"{name} must be {n}x{n}, got {tuple(M.shape)}",
            remedy="Match the covariance block to its variable's dimension.",
            offending_object=tuple(M.shape),
        )
    if symmetric:
        asym = float(torch.linalg.norm(M - M.T))
        scale = float(torch.linalg.norm(M)) + 1e-300
        if asym / scale > 1e-8:
            raise CovarianceError(
                f"{name} is not symmetric (asymmetry {asym:.3e})",
                remedy="Symmetrize deliberately, or fix the estimator.",
                offending_object=name,
            )
        M = 0.5 * (M + M.T)
        eig = torch.linalg.eigvalsh(M)
        if float(eig.min()) < -1e-9 * max(1.0, float(eig.max())):
            raise CovarianceError(
                f"{name} is not positive semi-definite "
                f"(min eigenvalue {float(eig.min()):.3e})",
                remedy="A negative-variance direction is an estimator bug.",
                offending_object=name,
            )
    return M


def _jacobian(f: Callable[..., Array], *args: Array, argnums: Sequence[int]) -> list[Array]:
    """Numerically exact (autograd) Jacobians of ``f`` w.r.t. selected args."""
    args = tuple(a.detach().clone().to(DTYPE).requires_grad_(True) for a in args)
    out = f(*args)
    if out.ndim == 0:
        out = out.reshape(1)
    rows = []
    for i in range(out.numel()):
        grads = torch.autograd.grad(
            out.reshape(-1)[i], args, retain_graph=True, allow_unused=True
        )
        rows.append([
            torch.zeros_like(args[k]) if grads[k] is None else grads[k] for k in range(len(args))
        ])
    Js = []
    for k in argnums:
        Js.append(torch.stack([r[k].reshape(-1) for r in rows]))
    return Js


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Propagated:
    """Result of a propagation.  Bias and variance never collapse (thesis §2.7)."""

    value: Array
    cov: Array
    bias: Array
    terms: dict[str, Array] = field(default_factory=dict)
    method: str = "first_order_T5"
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def total_variance(self) -> float:
        return float(torch.trace(self.cov))

    @property
    def cross_term(self) -> Array:
        """``J_x Sigma_xc J_c^T + J_c Sigma_cx J_x^T`` -- the mandatory part."""
        return self.terms.get("cross", torch.zeros_like(self.cov))

    def sd(self) -> Array:
        return torch.sqrt(torch.clamp(torch.diagonal(self.cov), min=0.0))

    def summary(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "value": self.value.tolist(),
            "sd": self.sd().tolist(),
            "bias": self.bias.tolist(),
            "total_variance": self.total_variance,
            "cross_term_trace": float(torch.trace(self.cross_term)),
        }


# --------------------------------------------------------------------------
# T5
# --------------------------------------------------------------------------


def propagate_first_order(
    f: Callable[[Array, Array], Array],
    x: Any,
    c: Any,
    *,
    Sx: Any,
    Sc: Any,
    Sxc: Any | None = None,
    bx: Any | None = None,
    bc: Any | None = None,
    delta_f: Any | None = None,
    include_cross: bool = True,
) -> Propagated:
    """Equation (T5), cross terms included.

    Parameters
    ----------
    f:
        ``z = f(x, c)``, differentiable, ``c`` the calibration coefficients.
    Sxc:
        ``Sigma_xc`` (``n_x x n_c``).  ``None`` means *declared* independence;
        it is not a default to reach for when the session shares a calibration.
    delta_f:
        model discrepancy added to the bias, per T5.  It is *not* variance.
    include_cross:
        Exposed only so :func:`independence_understatement` can construct the
        wrong answer on purpose for comparison.  Production callers leave it
        ``True``; setting it ``False`` is recorded in provenance.
    """
    xv = torch.as_tensor(x, dtype=DTYPE).reshape(-1)
    cv = torch.as_tensor(c, dtype=DTYPE).reshape(-1)
    nx, nc = xv.numel(), cv.numel()
    Sx_ = _as_cov(Sx, nx, "Sigma_x")
    Sc_ = _as_cov(Sc, nc, "Sigma_c")
    Sxc_ = (
        torch.zeros((nx, nc), dtype=DTYPE)
        if Sxc is None
        else torch.as_tensor(Sxc, dtype=DTYPE).reshape(nx, nc)
    )

    Jx, Jc = _jacobian(lambda a, b: f(a, b), xv, cv, argnums=(0, 1))
    z = torch.as_tensor(f(xv, cv), dtype=DTYPE).reshape(-1)

    term_x = Jx @ Sx_ @ Jx.T
    term_c = Jc @ Sc_ @ Jc.T
    cross_half = Jx @ Sxc_ @ Jc.T
    cross = cross_half + cross_half.T
    cov = term_x + term_c + (cross if include_cross else torch.zeros_like(cross))

    bx_ = torch.zeros(nx, dtype=DTYPE) if bx is None else torch.as_tensor(bx, dtype=DTYPE).reshape(-1)
    bc_ = torch.zeros(nc, dtype=DTYPE) if bc is None else torch.as_tensor(bc, dtype=DTYPE).reshape(-1)
    df_ = (
        torch.zeros(z.numel(), dtype=DTYPE)
        if delta_f is None
        else torch.as_tensor(delta_f, dtype=DTYPE).reshape(-1)
    )
    bias = Jx @ bx_ + Jc @ bc_ + df_

    # Guard against the classic silent failure: an indefinite joint covariance
    # (a declared Sigma_xc inconsistent with Sigma_x, Sigma_c).
    joint = torch.zeros((nx + nc, nx + nc), dtype=DTYPE)
    joint[:nx, :nx] = Sx_
    joint[nx:, nx:] = Sc_
    joint[:nx, nx:] = Sxc_
    joint[nx:, :nx] = Sxc_.T
    eig = torch.linalg.eigvalsh(0.5 * (joint + joint.T))
    if float(eig.min()) < -1e-8 * max(1.0, float(eig.max())):
        raise CovarianceError(
            "the declared joint covariance [[Sx, Sxc], [Sxc^T, Sc]] is not PSD "
            f"(min eigenvalue {float(eig.min()):.3e}); Sigma_xc is inconsistent "
            "with its marginals",
            remedy=(
                "Estimate Sigma_xc from the shared calibration model "
                "(e.g. Sxc = A Sigma_cal B^T) instead of asserting a correlation."
            ),
            offending_object="Sigma_xc",
        )

    return Propagated(
        value=z,
        cov=0.5 * (cov + cov.T),
        bias=bias,
        terms={"Jx_Sx_JxT": term_x, "Jc_Sc_JcT": term_c, "cross": cross, "Jx": Jx, "Jc": Jc},
        method="first_order_T5" if include_cross else "first_order_T5_NO_CROSS(invalid)",
        provenance={"include_cross": include_cross},
    )


def independence_understatement(
    f: Callable[[Array, Array], Array],
    x: Any,
    c: Any,
    *,
    Sx: Any,
    Sc: Any,
    Sxc: Any,
    **kw: Any,
) -> dict[str, float]:
    """Quantify what dropping the T5 cross terms costs.

    Returns the trace / generalized-variance ratios between the correct T5
    result and the (wrong) independence assumption.  A ratio > 1 means the
    independence assumption *understates* uncertainty by that factor.
    """
    full = propagate_first_order(f, x, c, Sx=Sx, Sc=Sc, Sxc=Sxc, include_cross=True, **kw)
    indep = propagate_first_order(f, x, c, Sx=Sx, Sc=Sc, Sxc=Sxc, include_cross=False, **kw)
    tf, ti = full.total_variance, indep.total_variance
    out = {
        "trace_with_cross": tf,
        "trace_independent": ti,
        "trace_ratio": tf / ti if ti > 0 else math.inf,
        "cross_term_trace": float(torch.trace(full.cross_term)),
        "sd_ratio": math.sqrt(tf / ti) if ti > 0 else math.inf,
    }
    if full.cov.shape[0] > 1:
        det_f = float(torch.linalg.det(full.cov))
        det_i = float(torch.linalg.det(indep.cov))
        out["logdet_with_cross"] = math.log(det_f) if det_f > 0 else -math.inf
        out["logdet_independent"] = math.log(det_i) if det_i > 0 else -math.inf
    out["understated"] = out["trace_ratio"] > 1.0
    return out


def linearization_error(
    f: Callable[[Array, Array], Array],
    x: Any,
    c: Any,
    *,
    Sx: Any,
    Sc: Any,
    Sxc: Any | None = None,
    seed: int = 0,
    n: int = 4096,
    tol: float = 0.1,
    raise_on_invalid: bool = False,
) -> dict[str, float]:
    """Compare T5 against Monte Carlo; flag maps where linearization is invalid.

    ``tol`` is the maximum acceptable relative discrepancy in total variance.
    ``raise_on_invalid`` turns the diagnostic into a refusal, which is what a
    compiler should do before admitting a first-order ledger for a nonlinear
    registration or an acoustic focusing map.
    """
    fo = propagate_first_order(f, x, c, Sx=Sx, Sc=Sc, Sxc=Sxc)
    mc = monte_carlo_propagate(f, x, c, Sx=Sx, Sc=Sc, Sxc=Sxc, seed=seed, n=n)
    rel_var = abs(mc.total_variance - fo.total_variance) / max(mc.total_variance, 1e-300)
    mean_shift = float(torch.linalg.norm(mc.value - fo.value))
    scale = float(torch.sqrt(torch.clamp(torch.diagonal(mc.cov), min=0.0)).mean()) + 1e-300
    report = {
        "relative_variance_error": rel_var,
        "mean_shift": mean_shift,
        "mean_shift_in_sd": mean_shift / scale,
        "first_order_variance": fo.total_variance,
        "monte_carlo_variance": mc.total_variance,
        "tolerance": tol,
        "linearization_valid": rel_var <= tol and mean_shift / scale <= tol,
    }
    if raise_on_invalid and not report["linearization_valid"]:
        raise LinearizationInvalidError(
            "first-order (T5) propagation is invalid for this map: relative "
            f"variance error {rel_var:.3f}, mean shift {report['mean_shift_in_sd']:.3f} sd "
            f"(tolerance {tol})",
            remedy=(
                "Use monte_carlo_propagate or interval_propagate for nonlinear "
                "registration, thresholded tractography and acoustic focusing."
            ),
            offending_object=report,
        )
    return report


# --------------------------------------------------------------------------
# Monte Carlo
# --------------------------------------------------------------------------


def joint_gaussian_sampler(
    x: Any, c: Any, Sx: Any, Sc: Any, Sxc: Any | None, *, seed: int, n: int
) -> tuple[Array, Array]:
    """Draw ``n`` correlated ``(x, c)`` pairs from the declared joint Gaussian."""
    xv = torch.as_tensor(x, dtype=DTYPE).reshape(-1)
    cv = torch.as_tensor(c, dtype=DTYPE).reshape(-1)
    nx, nc = xv.numel(), cv.numel()
    Sx_ = _as_cov(Sx, nx, "Sigma_x")
    Sc_ = _as_cov(Sc, nc, "Sigma_c")
    Sxc_ = (
        torch.zeros((nx, nc), dtype=DTYPE)
        if Sxc is None
        else torch.as_tensor(Sxc, dtype=DTYPE).reshape(nx, nc)
    )
    joint = torch.zeros((nx + nc, nx + nc), dtype=DTYPE)
    joint[:nx, :nx] = Sx_
    joint[nx:, nx:] = Sc_
    joint[:nx, nx:] = Sxc_
    joint[nx:, :nx] = Sxc_.T
    joint = 0.5 * (joint + joint.T)
    evals, evecs = torch.linalg.eigh(joint)
    L = evecs @ torch.diag(torch.sqrt(torch.clamp(evals, min=0.0)))
    g = torch.Generator().manual_seed(int(seed))
    z = torch.randn((n, nx + nc), dtype=DTYPE, generator=g)
    draws = z @ L.T
    return xv + draws[:, :nx], cv + draws[:, nx:]


def monte_carlo_propagate(
    f: Callable[[Array, Array], Array],
    x: Any,
    c: Any,
    *,
    Sx: Any,
    Sc: Any,
    Sxc: Any | None = None,
    bx: Any | None = None,
    bc: Any | None = None,
    delta_f: Any | None = None,
    n: int = 4096,
    seed: int = 0,
) -> Propagated:
    """Monte-Carlo propagation for maps where T5 does not hold.

    The sampler honours ``Sigma_xc``, so a shared session calibration stays
    shared in the samples too.  Bias terms are propagated *through* the map
    separately (evaluating ``f`` at the biased input) rather than being
    linearized, and reported as ``bias``.
    """
    xs, cs = joint_gaussian_sampler(x, c, Sx, Sc, Sxc, seed=seed, n=n)
    vals = torch.stack([torch.as_tensor(f(xs[i], cs[i]), dtype=DTYPE).reshape(-1) for i in range(n)])
    mean = vals.mean(dim=0)
    centered = vals - mean
    cov = (centered.T @ centered) / (n - 1)

    xv = torch.as_tensor(x, dtype=DTYPE).reshape(-1)
    cv = torch.as_tensor(c, dtype=DTYPE).reshape(-1)
    bx_ = torch.zeros_like(xv) if bx is None else torch.as_tensor(bx, dtype=DTYPE).reshape(-1)
    bc_ = torch.zeros_like(cv) if bc is None else torch.as_tensor(bc, dtype=DTYPE).reshape(-1)
    z0 = torch.as_tensor(f(xv, cv), dtype=DTYPE).reshape(-1)
    z_biased = torch.as_tensor(f(xv + bx_, cv + bc_), dtype=DTYPE).reshape(-1)
    df_ = torch.zeros_like(z0) if delta_f is None else torch.as_tensor(delta_f, dtype=DTYPE).reshape(-1)
    bias = (z_biased - z0) + df_

    return Propagated(
        value=mean,
        cov=0.5 * (cov + cov.T),
        bias=bias,
        terms={"samples": vals},
        method="monte_carlo",
        provenance={"n": n, "seed": int(seed), "nonlinear_mean_shift": float(torch.linalg.norm(mean - z0))},
    )


# --------------------------------------------------------------------------
# intervals
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class IntervalBox:
    """Axis-aligned box ``[lo, hi]``.  Used for bounded (not Gaussian) error."""

    lo: Array
    hi: Array

    def __post_init__(self) -> None:
        lo = torch.as_tensor(self.lo, dtype=DTYPE).reshape(-1)
        hi = torch.as_tensor(self.hi, dtype=DTYPE).reshape(-1)
        if lo.shape != hi.shape:
            raise TransformError(
                f"interval bounds disagree in shape: {tuple(lo.shape)} vs {tuple(hi.shape)}",
                remedy="Supply matching lower and upper bounds.",
                offending_object=(tuple(lo.shape), tuple(hi.shape)),
            )
        if bool((hi < lo).any()):
            raise TransformError(
                "interval upper bound below lower bound",
                remedy="Order the bounds.",
                offending_object=(lo.tolist(), hi.tolist()),
            )
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)

    @staticmethod
    def around(center: Any, half_width: Any) -> "IntervalBox":
        c = torch.as_tensor(center, dtype=DTYPE).reshape(-1)
        h = torch.as_tensor(half_width, dtype=DTYPE).reshape(-1).expand_as(c)
        return IntervalBox(c - h, c + h)

    @property
    def center(self) -> Array:
        return 0.5 * (self.lo + self.hi)

    @property
    def radius(self) -> Array:
        return 0.5 * (self.hi - self.lo)

    def corners(self) -> Array:
        n = self.lo.numel()
        if n > 16:
            raise TransformError(
                f"refusing to enumerate 2^{n} corners",
                remedy="Use method='lipschitz' or Monte Carlo for high dimension.",
                offending_object=n,
            )
        idx = torch.arange(2**n)
        bits = ((idx[:, None] >> torch.arange(n)[None, :]) & 1).to(DTYPE)
        return self.lo + bits * (self.hi - self.lo)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"IntervalBox(lo={self.lo.tolist()}, hi={self.hi.tolist()})"


@dataclass(frozen=True)
class IntervalResult:
    box: IntervalBox
    method: str
    rigorous: bool
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def lo(self) -> Array:
        return self.box.lo

    @property
    def hi(self) -> Array:
        return self.box.hi


def interval_propagate(
    f: Callable[[Array, Array], Array],
    x_box: IntervalBox,
    c_box: IntervalBox,
    *,
    method: str = "lipschitz",
    n_probe: int = 256,
    seed: int = 0,
) -> IntervalResult:
    """Outer bound on ``f`` over a box of inputs.

    ``method="corners"``  -- exact for maps monotone in each argument (affine
    maps included); flagged ``rigorous=True`` only in the affine case, which is
    checked numerically.

    ``method="lipschitz"`` -- centre value expanded by a Jacobian bound sampled
    over the box: ``|f(u) - f(u0)| <= sup|J| . radius``.  The supremum is
    estimated by sampling, so the bound is an *estimate*, reported with
    ``rigorous=False``.  Honesty here matters more than a tight number: an
    interval advertised as guaranteed when it is sampled is exactly the kind of
    quiet overconfidence the ledger exists to prevent.
    """
    xc, cc = x_box.center, c_box.center
    z0 = torch.as_tensor(f(xc, cc), dtype=DTYPE).reshape(-1)

    if method == "corners":
        vals = []
        for xv in x_box.corners():
            for cv in c_box.corners():
                vals.append(torch.as_tensor(f(xv, cv), dtype=DTYPE).reshape(-1))
        V = torch.stack(vals)
        lo, hi = V.min(dim=0).values, V.max(dim=0).values
        # affine check: midpoint value equals mean of opposite corners
        affine = bool(torch.allclose(0.5 * (V[0] + V[-1]), z0, atol=1e-8, rtol=1e-8))
        return IntervalResult(
            IntervalBox(torch.minimum(lo, z0), torch.maximum(hi, z0)),
            "corners",
            rigorous=affine,
            provenance={"affine_detected": affine, "n_corners": int(V.shape[0])},
        )

    if method != "lipschitz":
        raise TransformError(
            f"unknown interval propagation method {method!r}",
            remedy="Use 'corners' or 'lipschitz'.",
            offending_object=method,
        )

    g = torch.Generator().manual_seed(int(seed))
    nx, nc = x_box.lo.numel(), c_box.lo.numel()
    ux = torch.rand((n_probe, nx), dtype=DTYPE, generator=g)
    uc = torch.rand((n_probe, nc), dtype=DTYPE, generator=g)
    xs = x_box.lo + ux * (x_box.hi - x_box.lo)
    cs = c_box.lo + uc * (c_box.hi - c_box.lo)
    sup_x = torch.zeros((z0.numel(), nx), dtype=DTYPE)
    sup_c = torch.zeros((z0.numel(), nc), dtype=DTYPE)
    for i in range(n_probe):
        Jx, Jc = _jacobian(lambda a, b: f(a, b), xs[i], cs[i], argnums=(0, 1))
        sup_x = torch.maximum(sup_x, Jx.abs())
        sup_c = torch.maximum(sup_c, Jc.abs())
    spread = sup_x @ x_box.radius + sup_c @ c_box.radius
    # also fold in the observed sample range so a strongly nonlinear map cannot
    # produce an interval that fails to contain its own probe values
    probe_vals = torch.stack(
        [torch.as_tensor(f(xs[i], cs[i]), dtype=DTYPE).reshape(-1) for i in range(n_probe)]
    )
    lo = torch.minimum(z0 - spread, probe_vals.min(dim=0).values)
    hi = torch.maximum(z0 + spread, probe_vals.max(dim=0).values)
    return IntervalResult(
        IntervalBox(lo, hi),
        "lipschitz",
        rigorous=False,
        provenance={
            "n_probe": n_probe,
            "seed": int(seed),
            "note": "Jacobian supremum estimated by sampling; not a certified bound.",
        },
    )


# --------------------------------------------------------------------------
# pose chains: adjoint propagation (thesis §2.8)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PoseUncertainty:
    """Twist covariance + a *separate* systematic twist bias for one pose.

    §2.8: "For small perturbations, pose covariance is propagated through the
    chain with the relevant Jacobians or adjoint maps; systematic offsets are
    propagated as a separate twist bias."

    ``calibration_source`` / ``sensitivity`` express the shared part: an edge
    whose error partly comes from calibration parameter vector ``c`` (e.g. the
    optical tracker calibration used by *both* head<-tracker and
    tracker<-device) declares ``xi_edge = eta_edge + S c`` with
    ``eta_edge ~ N(0, cov)`` independent.  That is what makes ``Sigma_xc``
    nonzero downstream, and it is the physical reason T5's cross terms exist.
    """

    cov: Array  # 6x6 independent part, right-perturbation convention
    bias: Array = field(default_factory=lambda: torch.zeros(6, dtype=DTYPE))
    calibration_source: str | None = None
    sensitivity: Array | None = None  # 6 x k

    def __post_init__(self) -> None:
        object.__setattr__(self, "cov", _as_cov(self.cov, 6, "pose covariance"))
        object.__setattr__(self, "bias", torch.as_tensor(self.bias, dtype=DTYPE).reshape(6))
        if self.sensitivity is not None:
            S = torch.as_tensor(self.sensitivity, dtype=DTYPE)
            if S.ndim != 2 or S.shape[0] != 6:
                raise CovarianceError(
                    f"sensitivity must be 6xk, got {tuple(S.shape)}",
                    remedy="Map the shared calibration vector into a twist.",
                    offending_object=tuple(S.shape),
                )
            object.__setattr__(self, "sensitivity", S)
            if self.calibration_source is None:
                raise TransformError(
                    "a sensitivity matrix was given without naming the calibration source",
                    remedy="Name the shared calibration so cross terms can be built.",
                    offending_object=self,
                )

    @staticmethod
    def isotropic(sigma_t: float, sigma_r: float, **kw: Any) -> "PoseUncertainty":
        cov = torch.diag(
            torch.tensor(
                [sigma_t**2] * 3 + [sigma_r**2] * 3,
                dtype=DTYPE,
            )
        )
        return PoseUncertainty(cov, **kw)


@dataclass(frozen=True)
class ChainUncertainty:
    """Propagated uncertainty of a composed pose chain."""

    cov: Array
    bias: Array
    cov_independent_only: Array
    jacobians: list[Array]
    shared_sources: dict[str, Array]
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def translation_sd(self) -> Array:
        return torch.sqrt(torch.clamp(torch.diagonal(self.cov)[:3], min=0.0))

    @property
    def rotation_sd_rad(self) -> Array:
        return torch.sqrt(torch.clamp(torch.diagonal(self.cov)[3:], min=0.0))

    def understatement(self) -> dict[str, float]:
        """How much an independence assumption would understate the total."""
        full = float(torch.trace(self.cov))
        indep = float(torch.trace(self.cov_independent_only))
        return {
            "trace_with_shared_cross_terms": full,
            "trace_assuming_independence": indep,
            "trace_ratio": full / indep if indep > 0 else math.inf,
            "sd_ratio": math.sqrt(full / indep) if indep > 0 else math.inf,
        }


def chain_jacobians(poses: Sequence[Pose]) -> list[Array]:
    """``J_i = Ad((T_{i+1} ... T_n)^{-1})`` for right-perturbed factors.

    With ``T_i = Tbar_i exp(xi_i)``, the composed pose satisfies
    ``T = Tbar_1...Tbar_n exp(sum_i J_i xi_i)`` to first order.
    """
    n = len(poses)
    Js: list[Array] = []
    for i in range(n):
        suffix = torch.eye(4, dtype=DTYPE)
        for j in range(i + 1, n):
            suffix = suffix @ poses[j].matrix
        Js.append(adjoint(torch.linalg.inv(suffix)))
    return Js


def propagate_chain(
    poses: Sequence[Pose],
    uncertainties: Sequence[PoseUncertainty],
    *,
    shared_covariances: Mapping[str, Any] | None = None,
    cross_covariances: Mapping[tuple[int, int], Any] | None = None,
) -> ChainUncertainty:
    """Propagate pose covariance and twist bias through an SE(3) chain.

    This is T5 in SE(3) clothing.  Writing ``x`` for the stack of per-edge
    independent twists and ``c`` for the shared calibration vectors:

    * ``J_x = [J_1 ... J_n]``  (block row of adjoints)
    * ``J_c = sum_i J_i S_i``  (edges that share a calibration source)
    * ``Sigma_z = J_x Sigma_x J_x^T + J_c Sigma_c J_c^T + cross``

    where the ``cross`` block includes both the shared-source coupling
    ``J_i S_i Sigma_c S_j^T J_j^T`` for ``i != j`` and any explicitly declared
    ``Sigma_{x_i x_j}``.  ``cov_independent_only`` is the (wrong) answer you get
    by dropping every off-diagonal, retained so callers can report the
    understatement instead of asserting it.
    """
    if len(poses) != len(uncertainties):
        raise TransformError(
            f"chain has {len(poses)} poses but {len(uncertainties)} uncertainty records",
            remedy="Every edge on a path carries its own ledger.",
            offending_object=(len(poses), len(uncertainties)),
        )
    Js = chain_jacobians(poses)
    n = len(poses)
    shared = dict(shared_covariances or {})

    cov = torch.zeros((6, 6), dtype=DTYPE)
    cov_indep = torch.zeros((6, 6), dtype=DTYPE)
    bias = torch.zeros(6, dtype=DTYPE)

    for i in range(n):
        Ji, Ui = Js[i], uncertainties[i]
        block = Ji @ Ui.cov @ Ji.T
        cov = cov + block
        cov_indep = cov_indep + block
        bias = bias + Ji @ Ui.bias

    # shared-calibration contributions (diagonal and cross)
    used_sources: dict[str, Array] = {}
    for i in range(n):
        Ui = uncertainties[i]
        if Ui.sensitivity is None:
            continue
        src = Ui.calibration_source
        if src not in shared:
            raise TransformError(
                f"edge {poses[i].label} declares calibration source {src!r} but no "
                "covariance was supplied for it",
                remedy=(
                    "Pass shared_covariances={source: Sigma_c}. An unquantified "
                    "shared calibration is exactly the term that must not vanish."
                ),
                offending_object=src,
            )
        Sc = _as_cov(shared[src], Ui.sensitivity.shape[1], f"Sigma_c[{src}]")
        used_sources[src] = Sc
        for j in range(n):
            Uj = uncertainties[j]
            if Uj.sensitivity is None or Uj.calibration_source != src:
                continue
            block = Js[i] @ Ui.sensitivity @ Sc @ Uj.sensitivity.T @ Js[j].T
            cov = cov + block
            if i == j:
                cov_indep = cov_indep + block

    for (i, j), S in (cross_covariances or {}).items():
        if i == j:
            raise TransformError(
                f"cross_covariances key ({i}, {j}) is a diagonal block",
                remedy="Put the marginal covariance in the edge's own ledger.",
                offending_object=(i, j),
            )
        Sij = torch.as_tensor(S, dtype=DTYPE).reshape(6, 6)
        block = Js[i] @ Sij @ Js[j].T
        cov = cov + block + block.T

    cov = 0.5 * (cov + cov.T)
    eig = torch.linalg.eigvalsh(cov)
    if float(eig.min()) < -1e-9 * max(1.0, float(eig.max())):
        raise CovarianceError(
            "propagated chain covariance is not PSD; a declared cross-covariance "
            "is inconsistent with its marginals "
            f"(min eigenvalue {float(eig.min()):.3e})",
            remedy="Derive cross terms from a shared calibration model.",
            offending_object="chain covariance",
        )

    return ChainUncertainty(
        cov=cov,
        bias=bias,
        cov_independent_only=0.5 * (cov_indep + cov_indep.T),
        jacobians=Js,
        shared_sources=used_sources,
        provenance={
            "n_edges": n,
            "edges": [p.label for p in poses],
            "shared_calibration_sources": sorted(used_sources),
            "convention": "right perturbation, xi = (rho, phi)",
        },
    )


def sample_chain(
    poses: Sequence[Pose],
    uncertainties: Sequence[PoseUncertainty],
    *,
    shared_covariances: Mapping[str, Any] | None = None,
    n: int = 2048,
    seed: int = 0,
) -> Array:
    """Monte-Carlo draws of the composed chain's twist about its mean pose.

    Used to validate :func:`propagate_chain` (and to replace it where the
    rotations are large enough that first order is not good enough).
    """
    g = torch.Generator().manual_seed(int(seed))
    shared = dict(shared_covariances or {})
    src_draws: dict[str, Array] = {}
    for src, Sc in shared.items():
        M = torch.as_tensor(Sc, dtype=DTYPE)
        evals, evecs = torch.linalg.eigh(0.5 * (M + M.T))
        L = evecs @ torch.diag(torch.sqrt(torch.clamp(evals, min=0.0)))
        src_draws[src] = torch.randn((n, M.shape[0]), dtype=DTYPE, generator=g) @ L.T

    mean = torch.eye(4, dtype=DTYPE)
    for p in poses:
        mean = mean @ p.matrix
    mean_inv = torch.linalg.inv(mean)

    per_edge = []
    for U in uncertainties:
        evals, evecs = torch.linalg.eigh(U.cov)
        L = evecs @ torch.diag(torch.sqrt(torch.clamp(evals, min=0.0)))
        per_edge.append(torch.randn((n, 6), dtype=DTYPE, generator=g) @ L.T)

    out = torch.zeros((n, 6), dtype=DTYPE)
    for k in range(n):
        T = torch.eye(4, dtype=DTYPE)
        for i, (p, U) in enumerate(zip(poses, uncertainties)):
            xi = per_edge[i][k].clone()
            if U.sensitivity is not None:
                xi = xi + U.sensitivity @ src_draws[U.calibration_source][k]
            T = T @ p.matrix @ exp_se3(xi)
        out[k] = log_se3(mean_inv @ T)
    return out


__all__ = [
    "Propagated",
    "propagate_first_order",
    "independence_understatement",
    "linearization_error",
    "monte_carlo_propagate",
    "joint_gaussian_sampler",
    "IntervalBox",
    "IntervalResult",
    "interval_propagate",
    "PoseUncertainty",
    "ChainUncertainty",
    "chain_jacobians",
    "propagate_chain",
    "sample_chain",
]
