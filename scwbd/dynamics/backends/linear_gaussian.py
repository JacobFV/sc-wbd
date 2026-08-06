"""Linear–Gaussian backend: the T1 analytically tractable reference.

This is the baseline every other backend must beat.  It is a multivariate
Ornstein–Uhlenbeck process on the connectome,

    dx = (-x/tau + G W x + I) dt + sigma dW,

for which the mean, the transient covariance and the *stationary* covariance
(hence the model-implied functional connectivity) are available in closed form.
That makes it (i) the reference against which the integrators' convergence and
the coupling module's correctness are checked, and (ii) the baseline in every
model comparison: a mechanistic backend that does not beat a fitted OU process
on the relevant observable has not earned its label.

Closed forms provided:

``stationary_covariance``   solves ``A S + S A^T + Q = 0`` by eigendecomposition
``propagate_moments``       exact ``(m_t, S_t)`` for a finite horizon
``spectrum``                analytic cross-spectral density at given frequencies
"""

from __future__ import annotations

from typing import ClassVar, Mapping

import torch
from torch import Tensor

from ..base import BackendInfo, DynamicsBackend, register_backend
from ..types import DTYPE, ParamPack, Prior, default_device, make_generator

__all__ = ["LinearGaussian", "ou_stationary_covariance", "ou_propagate_moments"]


@register_backend
class LinearGaussian(DynamicsBackend):
    """Ornstein–Uhlenbeck network dynamics.  ``state_dim`` is configurable."""

    info: ClassVar[BackendInfo] = BackendInfo(
        name="linear_gaussian",
        family="flow_ode",
        mechanistic_status="effective",
        state_names=("x",),
        units=("dimensionless",),
        reference="Ornstein–Uhlenbeck / linearised neural mass; T1 reference system",
        falsifier=(
            "This backend makes no mechanistic claim. It is disabled as a *baseline* only by "
            "being beaten; if it is not beaten, the nonlinear backend's claim fails, not this one."
        ),
    )
    n_coupling_channels: ClassVar[int] = 1

    defaults: ClassVar[Mapping[str, float]] = {
        "tau": 0.020,  # s
        "G": 1.0,
        "I": 0.0,
        "sigma": 0.1,
        "self_gain": 0.0,  # extra local self-excitation beyond -1/tau
    }
    param_priors: ClassVar[Mapping[str, Prior]] = {
        "tau": Prior("tau", 0.020, 0.010, "uniform", "s", low=0.005, high=0.060),
        "G": Prior("G", 0.5, 0.5, "uniform", "dimensionless", low=0.0, high=1.5),
        "sigma": Prior("sigma", 0.1, 0.05, "uniform", "dimensionless", low=0.01, high=0.3),
    }
    regional_params: ClassVar[tuple[str, ...]] = ("tau",)

    def __init__(self, state_dim: int = 1):
        super().__init__()
        self._state_dim = int(state_dim)

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def state_names(self) -> tuple[str, ...]:
        return tuple(f"x{i}" for i in range(self._state_dim))

    def init_state(
        self,
        batch: int,
        n_regions: int,
        *,
        seed: int,
        device: str | torch.device | None = None,
        dtype: torch.dtype = DTYPE,
        theta: ParamPack | None = None,
    ) -> Tensor:
        dev = default_device(device)
        g = make_generator(seed, dev)
        return 0.1 * torch.randn((batch, n_regions, self._state_dim), generator=g, device=dev, dtype=dtype)

    def drift(
        self,
        x: Tensor,
        coupling_input: Tensor,
        theta: ParamPack,
        u: Tensor | None = None,
        t: float = 0.0,
    ) -> Tensor:
        self.check_state(x)
        d = (-1.0 / theta.get("tau") + theta.get("self_gain")) * x + theta.get("I")
        d = d + theta.get("G") * coupling_input[..., 0:1]
        if u is not None:
            d = d + u
        return d

    def diffusion(self, x: Tensor, theta: ParamPack) -> Tensor:
        return theta.get("sigma").expand_as(x)

    def observables(self, x: Tensor) -> dict[str, Tensor]:
        return {"activity": x[..., 0], "x": x[..., 0]}

    # -- closed forms ------------------------------------------------------
    def system_matrix(self, weights: Tensor, theta: ParamPack) -> Tensor:
        """Dense ``(B, N, N)`` system matrix ``A = diag(-1/tau + self_gain) + G W``.

        Densifying is legitimate *here* — the analytic reference is exactly the
        declared dense control (G2).  Nothing else in the dynamics core forms a
        dense N x N operator without an explicit request.
        """
        W = weights if weights.ndim == 3 else weights.unsqueeze(0)
        B = max(theta.batch, W.shape[0])
        N = W.shape[-1]
        diag = (-1.0 / theta.get("tau") + theta.get("self_gain")).reshape(theta.batch, -1)
        diag = diag.expand(B, N) if diag.shape[1] == 1 else diag
        G = theta.get("G").reshape(theta.batch, -1, 1)
        A = G * W.expand(B, N, N)
        A = A + torch.diag_embed(diag)
        return A

    def stationary_covariance(self, weights: Tensor, theta: ParamPack) -> Tensor:
        A = self.system_matrix(weights, theta)
        B, N, _ = A.shape
        sigma = theta.get("sigma").reshape(theta.batch, -1)
        sigma = sigma.expand(B, N) if sigma.shape[1] == 1 else sigma
        Q = torch.diag_embed(sigma**2)
        return ou_stationary_covariance(A, Q)

    def propagate_moments(
        self, weights: Tensor, theta: ParamPack, m0: Tensor, S0: Tensor, t: float
    ) -> tuple[Tensor, Tensor]:
        A = self.system_matrix(weights, theta)
        sigma = theta.get("sigma").reshape(theta.batch, -1)
        sigma = sigma.expand(A.shape[0], A.shape[1]) if sigma.shape[1] == 1 else sigma
        Q = torch.diag_embed(sigma**2)
        return ou_propagate_moments(A, Q, m0, S0, t)


def ou_stationary_covariance(A: Tensor, Q: Tensor, *, max_doublings: int = 60, tol: float = 1e-12) -> Tensor:
    """Solve the continuous Lyapunov equation ``A S + S A^T + Q = 0``.

    Uses the **doubling** (squaring) form of the integral
    ``S = int_0^inf e^{As} Q e^{A^T s} ds``:

        Phi_0 = e^{A h},  S_0 = int_0^h e^{As} Q e^{A^T s} ds   (Van Loan)
        S_{k+1} = S_k + Phi_k S_k Phi_k^T,   Phi_{k+1} = Phi_k^2

    after ``k`` doublings the integral covers ``2^k h``.  This avoids inverting
    the eigenvector matrix, which is what makes the textbook eigendecomposition
    method fail on whole-brain systems: a leak-dominated ``A = -I/tau + G W`` has
    nearly degenerate eigenvalues, ``V`` is ill-conditioned, and the "analytic"
    answer comes back with 1e10 entries.  Batched, on device, O(B N^3 log(1/eps)).

    Requires ``A`` Hurwitz; otherwise there is no stationary covariance and this
    raises rather than returning a number.
    """
    work = torch.float64 if A.dtype in (torch.float32, torch.float64) else A.dtype
    Ad, Qd = A.to(work), Q.to(work)
    lam = torch.linalg.eigvals(Ad)
    if bool((lam.real >= -1e-9).any()):
        raise ValueError(
            "system matrix is not Hurwitz: no stationary covariance exists "
            f"(max Re(lambda) = {float(lam.real.max()):.4g}). Reduce G or increase leak."
        )
    N = Ad.shape[-1]
    scale = torch.linalg.matrix_norm(Ad, ord="fro").reshape(-1, 1, 1).clamp_min(1e-12)
    h = (0.5 / scale).clamp(max=1.0)
    # Van Loan: one matrix exponential gives both Phi and the finite-horizon integral
    top = torch.cat([-Ad, Qd], dim=-1)
    bot = torch.cat([torch.zeros_like(Ad), Ad.transpose(-1, -2)], dim=-1)
    E = torch.linalg.matrix_exp(torch.cat([top, bot], dim=-2) * h)
    Phi = E[:, N:, N:].transpose(-1, -2)
    S = Phi @ E[:, :N, N:]
    for _ in range(max_doublings):
        S = S + Phi @ S @ Phi.transpose(-1, -2)
        Phi = Phi @ Phi
        if float(torch.linalg.matrix_norm(Phi, ord="fro").max()) < tol:
            break
    S = 0.5 * (S + S.transpose(-1, -2))
    return S.to(A.dtype)


def ou_propagate_moments(
    A: Tensor, Q: Tensor, m0: Tensor, S0: Tensor, t: float
) -> tuple[Tensor, Tensor]:
    """Exact OU moments at horizon ``t``.

    ``m_t = e^{At} m_0`` and ``S_t = e^{At} S_0 e^{A^T t} + int_0^t e^{As} Q e^{A^T s} ds``.
    The integral is evaluated with the standard matrix-fraction (Van Loan)
    construction so no stationarity assumption is needed.
    """
    N = A.shape[-1]
    B = A.shape[0]
    dtype = A.dtype
    top = torch.cat([-A, Q], dim=-1)
    bot = torch.cat([torch.zeros_like(A), A.transpose(-1, -2)], dim=-1)
    M = torch.cat([top, bot], dim=-2) * t
    E = torch.linalg.matrix_exp(M)
    Phi = E[:, N:, N:].transpose(-1, -2)
    Sig = Phi @ E[:, :N, N:]
    m_t = (Phi @ m0.reshape(B, N, 1)).reshape(m0.shape)
    S_t = Phi @ S0 @ Phi.transpose(-1, -2) + Sig
    return m_t.to(dtype), (0.5 * (S_t + S_t.transpose(-1, -2))).to(dtype)
