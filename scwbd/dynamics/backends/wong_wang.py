"""Reduced Wong–Wang backends — the standard whole-brain resting-state model.

Two variants, both batched over parameter sets:

``ReducedWongWang``
    Excitatory/inhibitory version of Deco et al. (2014) with **feedback
    inhibition control (FIC)**: the inhibitory-to-excitatory weight ``J_i`` is
    tuned per region so that the excitatory input current sits at
    ``I_E - b_E/a_E = -0.026 nA`` (excitatory rate ~3.06 Hz) whatever the global
    coupling G is.  FIC is what keeps the working point comparable across G and
    across subjects; without it, increasing G just saturates the network.

``ReducedWongWangSingle``
    One-population reduction of Deco et al. (2013).  This is the variant in
    which the classic result lives: similarity between simulated and empirical
    functional connectivity peaks just below the bifurcation of the global
    coupling G.  It is cheaper (1 state/region) and is used for the criticality
    validation in ``tests/dynamics/test_wong_wang_criticality.py``.

Units: currents in nA, rates in Hz, time in s, synaptic gating dimensionless.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, ClassVar, Mapping

import torch
from torch import Tensor

from ..base import BackendInfo, DynamicsBackend, register_backend
from ..types import DTYPE, ParamPack, Prior, default_device, make_generator

__all__ = ["ReducedWongWang", "ReducedWongWangSingle", "FICResult", "tune_fic"]


def _transfer(x: Tensor, a: Tensor, b: Tensor, d: Tensor) -> Tensor:
    """Abbott–Chance f–I curve ``(a I - b) / (1 - exp(-d (a I - b)))``.

    The removable singularity at ``a I = b`` is handled with a series expansion
    so that gradients stay finite (the naive form yields 0/0 -> NaN, which is a
    real and frequently-shipped bug in whole-brain codebases).
    """
    z = a * x - b
    dz = d * z
    small = dz.abs() < 1e-4
    safe = torch.where(small, torch.ones_like(dz), dz)
    dense = z / (1.0 - torch.exp(-safe))
    # series: z / (1 - exp(-d z)) = 1/d + z/2 + d z^2/12 + O(z^3)
    series = 1.0 / d + z / 2.0 + d * z * z / 12.0
    return torch.where(small, series, dense)


@register_backend
class ReducedWongWang(DynamicsBackend):
    """Excitatory/inhibitory reduced Wong–Wang with feedback inhibition control."""

    info: ClassVar[BackendInfo] = BackendInfo(
        name="wong_wang",
        family="flow_ode",
        mechanistic_status="mechanistic",
        state_names=("S_E", "S_I"),
        units=("dimensionless", "dimensionless"),
        reference="Wong & Wang 2006; Deco et al. 2013, 2014",
        falsifier=(
            "Disabled if FIC-tuned regional inhibition confers no predictive advantage over a "
            "capacity-matched surrogate on held-out perturbation responses, or if the "
            "FC-vs-G peak near the bifurcation fails to appear on the target connectome."
        ),
    )
    n_coupling_channels: ClassVar[int] = 1

    defaults: ClassVar[Mapping[str, float]] = {
        "a_E": 310.0,  # 1/(nC)
        "b_E": 125.0,  # Hz
        "d_E": 0.16,  # s
        "a_I": 615.0,
        "b_I": 177.0,
        "d_I": 0.087,
        "tau_E": 0.100,  # s (NMDA)
        "tau_I": 0.010,  # s (GABA)
        "gamma": 0.641,
        "W_E": 1.0,
        "W_I": 0.7,
        "I_0": 0.382,  # nA
        "w_plus": 1.4,
        "J_NMDA": 0.15,  # nA
        "J_i": 1.0,  # nA, FIC-tuned per region
        "G": 1.0,  # global coupling
        "sigma": 0.01,  # nA
        "I_ext": 0.0,  # nA
        # NB: a gain on the *inhibitory* current (``-J_i * ei_ratio * S_I``), so
        # it varies INVERSELY with a conventional excitation/inhibition ratio:
        # >1 means more inhibition.  ``theta_from_prior`` inverts the anatomy
        # prior's E/I ratio before writing it here; do not assign that prior to
        # this key directly.
        "ei_ratio": 1.0,
    }
    param_priors: ClassVar[Mapping[str, Prior]] = {
        "G": Prior("G", 1.5, 1.5, "uniform", "dimensionless", low=0.0, high=3.0),
        "w_plus": Prior("w_plus", 1.4, 0.2, "normal", "dimensionless", low=0.8),
        "J_NMDA": Prior("J_NMDA", 0.15, 0.02, "normal", "nA", low=0.05),
        "sigma": Prior("sigma", 0.01, 0.005, "uniform", "nA", low=0.0, high=0.03),
        "I_0": Prior("I_0", 0.382, 0.02, "normal", "nA", low=0.2),
    }
    regional_params: ClassVar[tuple[str, ...]] = ("w_plus",)
    #: NMDA and GABA decay constants.  Kept out of ``param_priors`` so that
    #: ``sample_theta`` continues to hold them at their literature values; these
    #: bounds only constrain values pushed in from an external prior.
    param_support: ClassVar[Mapping[str, tuple[float | None, float | None]]] = {
        "tau_E": (0.050, 0.200),
        "tau_I": (0.005, 0.030),
    }

    #: FIC target for ``a_E I_E - b_E`` in Hz-equivalent units (Deco 2014)
    FIC_TARGET_CURRENT_OFFSET: ClassVar[float] = -0.026  # nA relative to b_E/a_E

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
        s_e = 0.14 + 0.02 * torch.rand((batch, n_regions, 1), generator=g, device=dev, dtype=dtype)
        s_i = 0.08 + 0.02 * torch.rand((batch, n_regions, 1), generator=g, device=dev, dtype=dtype)
        return torch.cat([s_e, s_i], dim=-1)

    def currents(self, x: Tensor, coupling_input: Tensor, theta: ParamPack, u: Tensor | None = None):
        S_E = x[..., 0:1].clamp(0.0, 1.0)
        S_I = x[..., 1:2].clamp_min(0.0)
        J = theta.get("J_NMDA")
        I_ext = theta.get("I_ext") + (u[..., 0:1] if u is not None else 0.0)
        I_E = (
            theta.get("W_E") * theta.get("I_0")
            + theta.get("w_plus") * J * S_E
            + theta.get("G") * J * coupling_input[..., 0:1]
            - theta.get("J_i") * theta.get("ei_ratio") * S_I
            + I_ext
        )
        I_I = theta.get("W_I") * theta.get("I_0") + J * S_E - S_I
        r_E = _transfer(I_E, theta.get("a_E"), theta.get("b_E"), theta.get("d_E"))
        r_I = _transfer(I_I, theta.get("a_I"), theta.get("b_I"), theta.get("d_I"))
        return I_E, I_I, r_E, r_I

    def drift(
        self,
        x: Tensor,
        coupling_input: Tensor,
        theta: ParamPack,
        u: Tensor | None = None,
        t: float = 0.0,
    ) -> Tensor:
        self.check_state(x)
        S_E = x[..., 0:1].clamp(0.0, 1.0)
        S_I = x[..., 1:2].clamp_min(0.0)
        _, _, r_E, r_I = self.currents(x, coupling_input, theta, u)
        dS_E = -S_E / theta.get("tau_E") + (1.0 - S_E) * theta.get("gamma") * r_E
        dS_I = -S_I / theta.get("tau_I") + r_I
        return torch.cat([dS_E, dS_I], dim=-1)

    def diffusion(self, x: Tensor, theta: ParamPack) -> Tensor:
        return theta.get("sigma").expand_as(x)

    def observables(self, x: Tensor) -> dict[str, Tensor]:
        S_E, S_I = x[..., 0], x[..., 1]
        return {"activity": S_E, "S_E": S_E, "S_I": S_I, "ei_balance": S_E - S_I}


@register_backend
class ReducedWongWangSingle(DynamicsBackend):
    """One-population reduced Wong–Wang (Deco et al. 2013).  1 state per region."""

    info: ClassVar[BackendInfo] = BackendInfo(
        name="wong_wang_single",
        family="flow_ode",
        mechanistic_status="mechanistic",
        state_names=("S",),
        units=("dimensionless",),
        reference="Deco et al. 2013 (J Neurosci 33:11239)",
        falsifier=(
            "Disabled if simulated-vs-empirical FC similarity does not peak near the "
            "bifurcation of G, i.e. if the critical-working-point claim fails on the "
            "target connectome."
        ),
    )
    n_coupling_channels: ClassVar[int] = 1

    defaults: ClassVar[Mapping[str, float]] = {
        "a": 270.0,  # 1/(nC)
        "b": 108.0,  # Hz
        "d": 0.154,  # s
        "tau_s": 0.100,  # s
        "gamma": 0.641,
        "w": 0.9,  # local recurrent weight
        "J_N": 0.2609,  # nA
        "I_0": 0.3,  # nA
        "G": 1.0,
        "sigma": 0.001,
        "I_ext": 0.0,
    }
    param_priors: ClassVar[Mapping[str, Prior]] = {
        "G": Prior("G", 1.0, 1.0, "uniform", "dimensionless", low=0.0, high=3.0),
        "w": Prior("w", 0.9, 0.1, "normal", "dimensionless", low=0.4),
        "sigma": Prior("sigma", 0.001, 0.001, "uniform", "nA", low=0.0, high=0.005),
    }

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
        return 0.05 + 0.05 * torch.rand((batch, n_regions, 1), generator=g, device=dev, dtype=dtype)

    def drift(
        self,
        x: Tensor,
        coupling_input: Tensor,
        theta: ParamPack,
        u: Tensor | None = None,
        t: float = 0.0,
    ) -> Tensor:
        self.check_state(x)
        S = x.clamp(0.0, 1.0)
        J = theta.get("J_N")
        I_ext = theta.get("I_ext") + (u[..., 0:1] if u is not None else 0.0)
        I = (
            theta.get("w") * J * S
            + theta.get("G") * J * coupling_input[..., 0:1]
            + theta.get("I_0")
            + I_ext
        )
        r = _transfer(I, theta.get("a"), theta.get("b"), theta.get("d"))
        return -S / theta.get("tau_s") + (1.0 - S) * theta.get("gamma") * r

    def diffusion(self, x: Tensor, theta: ParamPack) -> Tensor:
        return theta.get("sigma").expand_as(x)

    def observables(self, x: Tensor) -> dict[str, Tensor]:
        S = x[..., 0]
        return {"activity": S, "S": S}


# ---------------------------------------------------------------------------
# Feedback inhibition control
# ---------------------------------------------------------------------------


@dataclass
class FICResult:
    """Outcome of FIC tuning — reported, never silently applied."""

    J_i: Tensor  # (B, N)
    converged: Tensor  # (B,) bool
    final_offset: Tensor  # (B, N) nA, I_E - b_E/a_E
    iterations: int
    target: float

    def report(self) -> dict[str, float]:
        return {
            "iterations": float(self.iterations),
            "target_nA": float(self.target),
            "max_abs_offset_error_nA": float((self.final_offset - self.target).abs().max()),
            "frac_converged": float(self.converged.float().mean()),
        }


@torch.no_grad()
def tune_fic(
    backend: ReducedWongWang,
    theta: ParamPack,
    coupling_fn: Callable[[Tensor], Tensor],
    *,
    x0: Tensor,
    dt: float = 1e-3,
    n_steps: int = 3000,
    n_rounds: int = 12,
    lr: float = 0.5,
    tol: float = 0.005,
    seed: int = 0,
) -> FICResult:
    """Deco-style feedback inhibition control, batched over parameter sets.

    Runs a short noise-free relaxation, measures the mean excitatory input
    current offset ``I_E - b_E/a_E`` per region, and adjusts ``J_i`` towards the
    target ``-0.026 nA``.  Everything stays on device; the only Python loop is
    over tuning rounds (12), never over regions.

    ``coupling_fn`` maps the coupling variable ``(B, N, 1)`` to the coupling
    input ``(B, N, 1)`` — normally ``lambda s: connectome(s)`` with delays
    ignored, which is the standard (and declared) approximation in FIC.
    """
    target = float(backend.FIC_TARGET_CURRENT_OFFSET)
    x = x0.clone()
    B, N, _ = x.shape
    J_i = theta.get("J_i").expand(B, N, 1).clone()
    theta = theta.with_(J_i=J_i.squeeze(-1))
    converged = torch.zeros(B, dtype=torch.bool, device=x.device)
    offset = torch.zeros(B, N, device=x.device, dtype=x.dtype)
    for it in range(n_rounds):
        x = x0.clone()
        acc = torch.zeros(B, N, 1, device=x.device, dtype=x.dtype)
        n_acc = 0
        burn = n_steps // 3
        for k in range(n_steps):
            c = coupling_fn(backend.coupling_variable(x, theta))
            I_E, _, _, _ = backend.currents(x, c, theta)
            x = x + dt * backend.drift(x, c, theta)
            x = x.clamp(0.0, 1.0)
            if k >= burn:
                acc = acc + I_E
                n_acc += 1
        mean_I_E = acc / max(n_acc, 1)
        thr = theta.get("b_E") / theta.get("a_E")
        offset3 = mean_I_E - thr
        err = offset3 - target
        offset = offset3.squeeze(-1)
        converged = (err.abs().amax(dim=(1, 2)) < tol)
        if bool(converged.all()):
            break
        J_i = (J_i + lr * err).clamp_min(0.0)
        theta = theta.with_(J_i=J_i.squeeze(-1))
    return FICResult(J_i.squeeze(-1), converged, offset, it + 1, target)
