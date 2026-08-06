"""Oscillatory backends: Stuart–Landau (Hopf) and Kuramoto.

These are the metastability workhorses.  The Hopf normal form (Deco et al.
2017) lets a region sit sub- or super-critically and is the standard model for
resting-state amplitude/frequency structure; Kuramoto strips the amplitude away
and keeps phase only, which is the right control for asking whether an
explanation needs amplitude dynamics at all.

Stuart–Landau (real coordinates, diffusive coupling):

    dx = [(a - x^2 - y^2) x - omega y] dt + G sum_j w_ij (x_j(t-tau) - x_i) dt + s dW
    dy = [(a - x^2 - y^2) y + omega x] dt + G sum_j w_ij (y_j(t-tau) - y_i) dt + s dW

Kuramoto (phase only, coupling is a *pairwise nonlinearity*, hence
``coupling_kind = "phase_difference"``):

    dtheta_i = [omega_i + K sum_j w_ij sin(theta_j(t-tau) - theta_i(t))] dt + s dW
"""

from __future__ import annotations

import math
from typing import ClassVar, Mapping

import torch
from torch import Tensor

from ..base import BackendInfo, DynamicsBackend, register_backend
from ..types import DTYPE, ParamPack, Prior, default_device, make_generator

__all__ = ["StuartLandau", "Kuramoto", "kuramoto_order_parameter", "metastability"]


@register_backend
class StuartLandau(DynamicsBackend):
    """Hopf normal form, 2 states (x, y) per region, diffusive coupling."""

    info: ClassVar[BackendInfo] = BackendInfo(
        name="stuart_landau",
        family="flow_ode",
        mechanistic_status="effective",
        state_names=("x", "y"),
        units=("dimensionless", "dimensionless"),
        reference="Deco et al. 2017 (Hopf whole-brain model)",
        falsifier=(
            "Disabled if the bifurcation parameter a carries no regional information beyond a "
            "fitted band-limited AR process at matched capacity, i.e. if 'sub- vs supercritical' "
            "is not recoverable from data that the AR surrogate also fits."
        ),
    )
    #: diffusive coupling needs both quadrature components
    n_coupling_channels: ClassVar[int] = 2
    coupling_kind: ClassVar[str] = "additive"

    defaults: ClassVar[Mapping[str, float]] = {
        "a": -0.02,  # bifurcation parameter: <0 noisy focus, >0 limit cycle
        "f": 10.0,  # Hz, intrinsic frequency
        "G": 1.0,
        "sigma": 0.02,
        "diffusive": 1.0,  # 1 = (x_j - x_i), 0 = plain sum x_j
    }
    param_priors: ClassVar[Mapping[str, Prior]] = {
        "a": Prior("a", 0.0, 0.1, "uniform", "dimensionless", low=-0.2, high=0.2),
        "f": Prior("f", 10.0, 6.0, "uniform", "Hz", low=2.0, high=30.0),
        "G": Prior("G", 0.5, 0.5, "uniform", "dimensionless", low=0.0, high=2.0),
        "sigma": Prior("sigma", 0.02, 0.02, "uniform", "dimensionless", low=0.0, high=0.08),
    }
    regional_params: ClassVar[tuple[str, ...]] = ("a", "f")

    @property
    def state_dim(self) -> int:
        return 2

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
        return 0.1 * torch.randn((batch, n_regions, 2), generator=g, device=dev, dtype=dtype)

    def drift(
        self,
        x: Tensor,
        coupling_input: Tensor,
        theta: ParamPack,
        u: Tensor | None = None,
        t: float = 0.0,
    ) -> Tensor:
        self.check_state(x)
        xr, yi = x[..., 0:1], x[..., 1:2]
        a = theta.get("a")
        omega = 2.0 * math.pi * theta.get("f")
        G = theta.get("G")
        amp2 = xr * xr + yi * yi
        cx = coupling_input[..., 0:1]
        cy = coupling_input[..., 1:2]
        diff = theta.get("diffusive")
        # diffusive form subtracts the local value times the row sum, which the
        # coupling module supplies as channel-wise (sum_j w_ij x_j); the local
        # subtraction uses the row sum carried in theta["row_sum"] when present.
        row = theta.get("row_sum", 0.0)
        dx = (a - amp2) * xr - omega * yi + G * (cx - diff * row * xr)
        dy = (a - amp2) * yi + omega * xr + G * (cy - diff * row * yi)
        if u is not None:
            dx = dx + u[..., 0:1]
            if u.shape[-1] > 1:
                dy = dy + u[..., 1:2]
        return torch.cat([dx, dy], dim=-1)

    def diffusion(self, x: Tensor, theta: ParamPack) -> Tensor:
        return theta.get("sigma").expand_as(x)

    def observables(self, x: Tensor) -> dict[str, Tensor]:
        xr, yi = x[..., 0], x[..., 1]
        return {
            "activity": xr,
            "amplitude": torch.sqrt(xr * xr + yi * yi),
            "phase": torch.atan2(yi, xr),
        }

    def coupling_variable(self, x: Tensor, theta: ParamPack | None = None) -> Tensor:
        return x  # both quadratures travel


@register_backend
class Kuramoto(DynamicsBackend):
    """Phase-only oscillator network.  Coupling is pairwise (phase difference)."""

    info: ClassVar[BackendInfo] = BackendInfo(
        name="kuramoto",
        family="flow_ode",
        mechanistic_status="functional",
        state_names=("phase",),
        units=("rad",),
        reference="Kuramoto 1984; Cabral et al. 2011 (delayed Kuramoto connectome)",
        falsifier=(
            "Disabled as a mechanistic account if amplitude-carrying observations "
            "(power-envelope dynamics, evoked amplitude changes) are predicted better by "
            "Stuart-Landau at matched capacity: phase-only is then a lossy summary, not a mechanism."
        ),
    )
    n_coupling_channels: ClassVar[int] = 1
    coupling_kind: ClassVar[str] = "phase_difference"

    defaults: ClassVar[Mapping[str, float]] = {
        "f": 40.0,  # Hz
        "K": 1.0,  # global coupling strength
        "alpha": 0.0,  # phase frustration
        "sigma": 0.1,  # rad/sqrt(s)
    }
    param_priors: ClassVar[Mapping[str, Prior]] = {
        "f": Prior("f", 40.0, 2.0, "normal", "Hz", low=1.0),
        "K": Prior("K", 5.0, 5.0, "uniform", "dimensionless", low=0.0, high=20.0),
        "sigma": Prior("sigma", 0.1, 0.1, "uniform", "rad/s^0.5", low=0.0, high=0.5),
    }
    regional_params: ClassVar[tuple[str, ...]] = ("f",)

    @property
    def state_dim(self) -> int:
        return 1

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
        return 2.0 * math.pi * torch.rand((batch, n_regions, 1), generator=g, device=dev, dtype=dtype)

    def drift(
        self,
        x: Tensor,
        coupling_input: Tensor,
        theta: ParamPack,
        u: Tensor | None = None,
        t: float = 0.0,
    ) -> Tensor:
        self.check_state(x)
        omega = 2.0 * math.pi * theta.get("f")
        K = theta.get("K")
        d = omega + K * coupling_input[..., 0:1]
        if u is not None:
            d = d + u[..., 0:1]
        return d

    def diffusion(self, x: Tensor, theta: ParamPack) -> Tensor:
        return theta.get("sigma").expand_as(x)

    def observables(self, x: Tensor) -> dict[str, Tensor]:
        ph = x[..., 0]
        return {"activity": torch.sin(ph), "phase": ph, "cos": torch.cos(ph)}

    def coupling_variable(self, x: Tensor, theta: ParamPack | None = None) -> Tensor:
        return x  # the phase itself; the coupling module forms sin(theta_j - theta_i)

    def wrap(self, x: Tensor) -> Tensor:
        return torch.remainder(x, 2.0 * math.pi)


def kuramoto_order_parameter(phase: Tensor) -> Tensor:
    """``R(t) = |mean_i exp(i phi_i)|``.  ``phase`` is ``(..., N)`` -> ``(...)``."""
    return torch.abs(torch.exp(1j * phase.to(torch.complex64)).mean(dim=-1))


def metastability(phase: Tensor, time_dim: int = 0) -> Tensor:
    """SD over time of the Kuramoto order parameter — the standard metastability index."""
    r = kuramoto_order_parameter(phase)
    return r.std(dim=time_dim)
