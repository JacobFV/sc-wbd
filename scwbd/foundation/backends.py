"""Mechanistic dynamical backends + delayed coupling, for corpus generation.

Agent E owns ``scwbd.dynamics``.  This module **prefers agent E's registry** and
only falls back to the local implementations below when a backend is not yet
registered.  Everything here subclasses agent E's
:class:`scwbd.dynamics.base.DynamicsBackend`, so the fallbacks are drop-in
compatible in both directions and can be deleted without touching call sites.

Numerical contract (ARCHITECTURE.md §3): solvers run in **float32**; bf16 is
permitted only inside learned operators, never here.

Backends implemented: Wilson-Cowan, Jansen-Rit, reduced Wong-Wang,
Stuart-Landau, linear Gaussian.  The learned neural operator lives in
:mod:`scwbd.foundation.model` because it is simultaneously a backend and the
equal-capacity control for every mechanistic claim (Appendix D, "Operator /
mechanism claim").
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import ClassVar, Mapping

import torch
from torch import Tensor

from scwbd.dynamics.base import BackendInfo, DynamicsBackend, register_backend, sigmoid_offset
from scwbd.dynamics.types import DTYPE, ParamPack, Prior, assert_solver_dtype, make_generator

__all__ = [
    "FALLBACK_BACKENDS",
    "resolve_backend",
    "list_available_backends",
    "DelayedCoupling",
    "integrate_sde",
    "WilsonCowanFallback",
    "JansenRitFallback",
    "ReducedWongWangFallback",
    "StuartLandauFallback",
    "LinearGaussianFallback",
]


def _P(name, loc, scale, dist="normal", **kw) -> Prior:
    return Prior(name=name, loc=loc, scale=scale, dist=dist, **kw)


# ======================================================================
# Wilson-Cowan
# ======================================================================
class WilsonCowanFallback(DynamicsBackend):
    """Wilson-Cowan excitatory/inhibitory mean field (Wilson & Cowan 1972).

    .. math::
        \\tau_e \\dot E = -E + (1 - r_e E)\\,S_e(c_{ee}E - c_{ei}I + P + G u)\\\\
        \\tau_i \\dot I = -I + (1 - r_i I)\\,S_i(c_{ie}E - c_{ii}I + Q)
    """

    info: ClassVar[BackendInfo] = BackendInfo(
        name="wilson_cowan",
        family="flow_ode",
        mechanistic_status="mechanistic",
        state_names=("E", "I"),
        units=("dimensionless", "dimensionless"),
        reference="Wilson & Cowan 1972; Deco et al. 2009",
        falsifier=(
            "Regional spectra whose peak frequency does not shift with the E->I "
            "coupling implied by receptor maps, at any admissible parameterisation."
        ),
    )
    n_coupling_channels: ClassVar[int] = 1
    defaults: ClassVar[Mapping[str, float]] = {
        "tau_e": 0.010,
        "tau_i": 0.020,
        "c_ee": 16.0,
        "c_ei": 12.0,
        "c_ie": 15.0,
        "c_ii": 3.0,
        "a_e": 1.3,
        "b_e": 4.0,
        "a_i": 2.0,
        "b_i": 3.7,
        "r_e": 1.0,
        "r_i": 1.0,
        "P": 1.25,
        "Q": 0.0,
        "G": 0.6,
        "sigma": 0.01,
        "ei_ratio": 1.0,
    }
    param_priors: ClassVar[Mapping[str, Prior]] = {
        "G": _P("G", 0.6, 0.55, "uniform", low=0.02, high=2.5),
        "P": _P("P", 1.25, 0.45, "uniform", low=0.4, high=2.0),
        "sigma": _P("sigma", -4.0, 0.7, "lognormal", low=1e-4, high=0.12),
        "ei_ratio": _P("ei_ratio", 1.0, 0.22, "normal", low=0.5, high=1.8),
    }
    regional_params: ClassVar[tuple[str, ...]] = ("ei_ratio", "P")

    @property
    def state_dim(self) -> int:
        return 2

    def init_state(self, batch, n_regions, *, seed, device=None, dtype=DTYPE, theta=None):
        g = make_generator(seed, device)
        x = 0.1 + 0.02 * torch.randn(batch, n_regions, 2, generator=g, device=g.device, dtype=dtype)
        return x.clamp(0.0, 0.9)

    def drift(self, x, coupling_input, theta, u=None, t=0.0):
        E = x[..., 0:1]
        I = x[..., 1:2]
        ei = theta.get("ei_ratio")
        c_ee, c_ei = theta.get("c_ee") * ei, theta.get("c_ei")
        c_ie, c_ii = theta.get("c_ie"), theta.get("c_ii")
        G = theta.get("G")
        cpl = coupling_input[..., 0:1]
        drive_e = c_ee * E - c_ei * I + theta.get("P") + G * cpl
        drive_i = c_ie * E - c_ii * I + theta.get("Q")
        if u is not None:
            drive_e = drive_e + u[..., 0:1]
        se = sigmoid_offset(drive_e, theta.get("a_e"), theta.get("b_e"))
        si = sigmoid_offset(drive_i, theta.get("a_i"), theta.get("b_i"))
        dE = (-E + (1 - theta.get("r_e") * E) * se) / theta.get("tau_e")
        dI = (-I + (1 - theta.get("r_i") * I) * si) / theta.get("tau_i")
        return torch.cat([dE, dI], dim=-1)

    def diffusion(self, x, theta):
        s = theta.get("sigma")
        return torch.cat([s.expand_as(x[..., 0:1]), 0.5 * s.expand_as(x[..., 1:2])], dim=-1)

    def observables(self, x):
        return {"activity": x[..., 0], "rate_e": x[..., 0], "rate_i": x[..., 1], "eeg_source": x[..., 0] - x[..., 1]}


# ======================================================================
# Jansen-Rit
# ======================================================================
class JansenRitFallback(DynamicsBackend):
    """Jansen-Rit cortical column (Jansen & Rit 1995).  State (y0..y2, y3..y5)."""

    info: ClassVar[BackendInfo] = BackendInfo(
        name="jansen_rit",
        family="flow_ode",
        mechanistic_status="mechanistic",
        state_names=("y0", "y1", "y2", "y3", "y4", "y5"),
        units=("mV",) * 6,
        reference="Jansen & Rit 1995; David & Friston 2003",
        falsifier=(
            "Evoked responses whose polarity/latency ordering contradicts the "
            "pyramidal-interneuron PSP decomposition under matched drive."
        ),
    )
    defaults: ClassVar[Mapping[str, float]] = {
        "A": 3.25,
        "B": 22.0,
        "a": 100.0,
        "b": 50.0,
        "C": 135.0,
        "v0": 6.0,
        "e0": 2.5,
        "r": 0.56,
        "p_mean": 120.0,
        "G": 0.5,
        "sigma": 2.0,
        "ei_ratio": 1.0,
    }
    param_priors: ClassVar[Mapping[str, Prior]] = {
        "G": _P("G", 0.5, 0.5, "uniform", low=0.0, high=2.0),
        "A": _P("A", 3.25, 0.6, "uniform", low=2.6, high=4.5),
        "B": _P("B", 22.0, 6.0, "uniform", low=17.0, high=30.0),
        "p_mean": _P("p_mean", 120.0, 40.0, "uniform", low=60.0, high=200.0),
        "sigma": _P("sigma", 0.8, 0.5, "lognormal", low=0.2, high=12.0),
        "ei_ratio": _P("ei_ratio", 1.0, 0.15, "normal", low=0.6, high=1.6),
    }
    regional_params: ClassVar[tuple[str, ...]] = ("ei_ratio", "A")

    @property
    def state_dim(self) -> int:
        return 6

    def _sigm(self, v, theta):
        e0, r, v0 = theta.get("e0"), theta.get("r"), theta.get("v0")
        return 2 * e0 / (1 + torch.exp(r * (v0 - v)))

    def init_state(self, batch, n_regions, *, seed, device=None, dtype=DTYPE, theta=None):
        g = make_generator(seed, device)
        x = 0.05 * torch.randn(batch, n_regions, 6, generator=g, device=g.device, dtype=dtype)
        x[..., 1] += 0.15
        return x

    def drift(self, x, coupling_input, theta, u=None, t=0.0):
        y0, y1, y2, y3, y4, y5 = (x[..., i : i + 1] for i in range(6))
        A, B, a, b = theta.get("A"), theta.get("B"), theta.get("a"), theta.get("b")
        C = theta.get("C") * theta.get("ei_ratio")
        C1, C2, C3, C4 = C, 0.8 * C, 0.25 * C, 0.25 * C
        p = theta.get("p_mean") + theta.get("G") * coupling_input[..., 0:1] * 100.0
        if u is not None:
            p = p + u[..., 0:1]
        dy0, dy1, dy2 = y3, y4, y5
        dy3 = A * a * self._sigm(y1 - y2, theta) - 2 * a * y3 - a * a * y0
        dy4 = A * a * (p * 1e-3 * 2 * theta.get("e0") + C2 * self._sigm(C1 * y0, theta)) - 2 * a * y4 - a * a * y1
        dy5 = B * b * C4 * self._sigm(C3 * y0, theta) - 2 * b * y5 - b * b * y2
        return torch.cat([dy0, dy1, dy2, dy3, dy4, dy5], dim=-1)

    def diffusion(self, x, theta):
        s = theta.get("sigma")
        z = torch.zeros_like(x[..., 0:1])
        return torch.cat([z, z, z, z, s.expand_as(z), z], dim=-1)

    def observables(self, x):
        v = x[..., 1] - x[..., 2]
        return {"activity": v, "eeg_source": v, "rate_e": x[..., 1], "rate_i": x[..., 2]}

    def coupling_variable(self, x, theta=None):
        v = (x[..., 1] - x[..., 2]).unsqueeze(-1)
        return torch.sigmoid(0.56 * (v - 6.0)) if theta is None else 2 * theta.get("e0")[..., 0:1] / (
            1 + torch.exp(theta.get("r")[..., 0:1] * (theta.get("v0")[..., 0:1] - v))
        )


# ======================================================================
# reduced Wong-Wang
# ======================================================================
class ReducedWongWangFallback(DynamicsBackend):
    """Reduced Wong-Wang with feedback inhibition (Deco et al. 2014).  State (S_E, S_I)."""

    info: ClassVar[BackendInfo] = BackendInfo(
        name="reduced_wong_wang",
        family="flow_ode",
        mechanistic_status="mechanistic",
        state_names=("S_E", "S_I"),
        units=("dimensionless", "dimensionless"),
        reference="Deco et al. 2014 J Neurosci",
        falsifier=(
            "Failure of the feedback-inhibition control to hold regional firing near "
            "its target under any admissible J_i when structural coupling is scaled."
        ),
    )
    defaults: ClassVar[Mapping[str, float]] = {
        "a_E": 310.0,
        "b_E": 125.0,
        "d_E": 0.16,
        "a_I": 615.0,
        "b_I": 177.0,
        "d_I": 0.087,
        "tau_E": 0.100,
        "tau_I": 0.010,
        "gamma": 0.641,
        "W_E": 1.0,
        "W_I": 0.7,
        "I0": 0.382,
        "w_plus": 1.4,
        "J_NMDA": 0.15,
        "J_i": 1.0,
        "G": 1.0,
        "sigma": 0.01,
        "ei_ratio": 1.0,
    }
    param_priors: ClassVar[Mapping[str, Prior]] = {
        "G": _P("G", 1.5, 1.0, "uniform", low=0.05, high=3.5),
        "J_i": _P("J_i", 1.0, 0.4, "uniform", low=0.3, high=2.0),
        "w_plus": _P("w_plus", 1.4, 0.3, "uniform", low=1.0, high=1.9),
        "sigma": _P("sigma", -4.6, 0.6, "lognormal", low=1e-4, high=0.05),
        "ei_ratio": _P("ei_ratio", 1.0, 0.2, "normal", low=0.5, high=1.7),
    }
    regional_params: ClassVar[tuple[str, ...]] = ("ei_ratio", "J_i")

    @property
    def state_dim(self) -> int:
        return 2

    @staticmethod
    def _rate(I, a, b, d):
        """``(aI-b)/(1-exp(-d(aI-b)))`` with the removable singularity at 0 handled.

        Clamping the denominator away from zero (the obvious shortcut) flips the
        sign for negative arguments and makes the model diverge; the limit
        ``1/d + x/2`` is used inside a small band instead.
        """
        x = a * I - b
        dx = d * x
        safe = torch.where(dx.abs() < 1e-3, torch.ones_like(dx), dx)
        big = x / (1 - torch.exp(-safe))
        small = 1.0 / d + 0.5 * x
        return torch.where(dx.abs() < 1e-3, small, big)

    def init_state(self, batch, n_regions, *, seed, device=None, dtype=DTYPE, theta=None):
        g = make_generator(seed, device)
        s = 0.15 + 0.03 * torch.randn(batch, n_regions, 2, generator=g, device=g.device, dtype=dtype)
        return s.clamp(0.001, 0.9)

    def drift(self, x, coupling_input, theta, u=None, t=0.0):
        SE = x[..., 0:1].clamp(0.0, 1.0)
        SI = x[..., 1:2].clamp(0.0, 1.0)
        I0, W_E, W_I = theta.get("I0"), theta.get("W_E"), theta.get("W_I")
        J = theta.get("J_NMDA")
        wp = theta.get("w_plus") * theta.get("ei_ratio")
        IE = W_E * I0 + wp * J * SE + theta.get("G") * J * coupling_input[..., 0:1] - theta.get("J_i") * SI
        II = W_I * I0 + J * SE - SI
        if u is not None:
            IE = IE + u[..., 0:1]
        rE = self._rate(IE, theta.get("a_E"), theta.get("b_E"), theta.get("d_E"))
        rI = self._rate(II, theta.get("a_I"), theta.get("b_I"), theta.get("d_I"))
        dSE = -SE / theta.get("tau_E") + (1 - SE) * theta.get("gamma") * rE
        dSI = -SI / theta.get("tau_I") + rI
        return torch.cat([dSE, dSI], dim=-1)

    def diffusion(self, x, theta):
        s = theta.get("sigma")
        return torch.cat([s.expand_as(x[..., 0:1]), s.expand_as(x[..., 1:2])], dim=-1)

    def observables(self, x):
        return {"activity": x[..., 0], "rate_e": x[..., 0], "rate_i": x[..., 1], "eeg_source": x[..., 0] - x[..., 1]}


# ======================================================================
# Stuart-Landau
# ======================================================================
class StuartLandauFallback(DynamicsBackend):
    """Stuart-Landau (Hopf) normal form -- the canonical *effective* oscillator.

    Its ``mechanistic_status`` is deliberately ``effective``: it reproduces
    spectra and functional connectivity without committing to a circuit.
    """

    info: ClassVar[BackendInfo] = BackendInfo(
        name="stuart_landau",
        family="flow_ode",
        mechanistic_status="effective",
        state_names=("x", "y"),
        units=("dimensionless", "dimensionless"),
        reference="Deco et al. 2017 Sci Rep (Hopf whole-brain model)",
        falsifier=(
            "Perturbational responses whose shape depends on drive amplitude in a way "
            "a normal form near a supercritical Hopf bifurcation cannot express."
        ),
    )
    defaults: ClassVar[Mapping[str, float]] = {
        "a": -0.02,
        "omega": 62.8,
        "G": 0.5,
        "sigma": 0.04,
        "ei_ratio": 1.0,
    }
    param_priors: ClassVar[Mapping[str, Prior]] = {
        "G": _P("G", 0.5, 0.5, "uniform", low=0.0, high=2.0),
        "a": _P("a", 0.0, 0.1, "uniform", low=-0.2, high=0.1),
        "omega": _P("omega", 62.8, 30.0, "uniform", low=12.6, high=125.7),
        "sigma": _P("sigma", -3.2, 0.5, "lognormal", low=0.005, high=0.2),
    }
    regional_params: ClassVar[tuple[str, ...]] = ("a", "omega")

    @property
    def state_dim(self) -> int:
        return 2

    def init_state(self, batch, n_regions, *, seed, device=None, dtype=DTYPE, theta=None):
        g = make_generator(seed, device)
        return 0.1 * torch.randn(batch, n_regions, 2, generator=g, device=g.device, dtype=dtype)

    def drift(self, x, coupling_input, theta, u=None, t=0.0):
        xr, yi = x[..., 0:1], x[..., 1:2]
        r2 = xr * xr + yi * yi
        a, w, G = theta.get("a"), theta.get("omega"), theta.get("G")
        cpl = G * coupling_input[..., 0:1]
        dx = (a - r2) * xr - w * yi + cpl
        dy = (a - r2) * yi + w * xr
        if u is not None:
            dx = dx + u[..., 0:1]
        return torch.cat([dx, dy], dim=-1)

    def diffusion(self, x, theta):
        s = theta.get("sigma")
        return s.expand_as(x)

    def observables(self, x):
        return {"activity": x[..., 0], "rate_e": x[..., 0].clamp_min(0), "rate_i": (-x[..., 0]).clamp_min(0), "eeg_source": x[..., 0]}


# ======================================================================
# linear Gaussian
# ======================================================================
class LinearGaussianFallback(DynamicsBackend):
    """Linear (Ornstein-Uhlenbeck) network -- the honest null dynamics model.

    Retained because a large share of resting-state FC is reproducible by a
    linear model; any nonlinear backend that does not beat this one has not
    earned its extra structure.
    """

    info: ClassVar[BackendInfo] = BackendInfo(
        name="linear_gaussian",
        family="delayed_ssm",
        mechanistic_status="effective",
        state_names=("x",),
        units=("dimensionless",),
        reference="Galan 2008; Saggio et al. 2016",
        falsifier="Any observable requiring amplitude-dependent frequency or waveform asymmetry.",
    )
    defaults: ClassVar[Mapping[str, float]] = {"tau": 0.020, "G": 0.5, "sigma": 0.05, "ei_ratio": 1.0}
    param_priors: ClassVar[Mapping[str, Prior]] = {
        "G": _P("G", 0.4, 0.35, "uniform", low=0.0, high=0.95),
        "tau": _P("tau", 0.02, 0.015, "uniform", low=0.006, high=0.12),
        "sigma": _P("sigma", -3.0, 0.5, "lognormal", low=0.005, high=0.3),
    }
    regional_params: ClassVar[tuple[str, ...]] = ("tau",)

    @property
    def state_dim(self) -> int:
        return 1

    def init_state(self, batch, n_regions, *, seed, device=None, dtype=DTYPE, theta=None):
        g = make_generator(seed, device)
        return 0.05 * torch.randn(batch, n_regions, 1, generator=g, device=g.device, dtype=dtype)

    def drift(self, x, coupling_input, theta, u=None, t=0.0):
        d = -x / theta.get("tau") + theta.get("G") * coupling_input[..., 0:1]
        return d if u is None else d + u[..., 0:1]

    def diffusion(self, x, theta):
        return theta.get("sigma").expand_as(x)

    def observables(self, x):
        return {"activity": x[..., 0], "rate_e": x[..., 0].clamp_min(0), "rate_i": (-x[..., 0]).clamp_min(0), "eeg_source": x[..., 0]}


#: Local stand-ins, used **only** for names ``scwbd.dynamics`` does not provide.
#: They are deliberately not auto-registered under the canonical names: a
#: fallback that shadows the real implementation because of import order is a
#: worse failure than no fallback at all.
FALLBACK_BACKENDS: dict[str, type[DynamicsBackend]] = {
    "wilson_cowan": WilsonCowanFallback,
    "jansen_rit": JansenRitFallback,
    "reduced_wong_wang": ReducedWongWangFallback,
    "stuart_landau": StuartLandauFallback,
    "linear_gaussian": LinearGaussianFallback,
}


def _import_agent_e() -> None:
    """Import agent E's concrete backends if they exist, so they self-register."""
    for name in ("scwbd.dynamics", "scwbd.dynamics.models", "scwbd.dynamics.backends", "scwbd.dynamics.neural_mass"):
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001
            pass


def resolve_backend(name: str, **kw) -> DynamicsBackend:
    """Instantiate ``name``, preferring agent E's implementation.

    Resolution is explicit rather than registration-order dependent: agent E's
    registry is consulted first, and the local fallback is used only for a name
    agent E does not provide.  (Relying on which module imported last is how a
    fallback silently shadows the real implementation.)  ``origin`` on the
    returned object records which one actually ran, and it is written into every
    corpus shard.
    """
    _import_agent_e()
    from scwbd.dynamics.base import get_backend

    try:
        cls = get_backend(name)
        origin = "scwbd.dynamics (agent E)"
    except KeyError:
        if name not in FALLBACK_BACKENDS:
            raise
        cls = FALLBACK_BACKENDS[name]
        origin = "fallback(agent I)"
    if cls in FALLBACK_BACKENDS.values():
        origin = "fallback(agent I)"
    obj = cls(**kw)
    obj.origin = origin  # type: ignore[attr-defined]
    return obj


def list_available_backends() -> list[str]:
    _import_agent_e()
    from scwbd.dynamics.base import list_backends

    return list_backends()


# ======================================================================
# delayed, connectome-masked coupling
# ======================================================================
@dataclass
class DelayedCoupling:
    """Delay-binned block-sparse coupling  ``c_i(t) = sum_j W_ij x_j(t - d_ij)``.

    Exact per-edge delays would require an ``(B,N,N)`` gather every step.  We
    instead quantise delays into ``n_bins`` lags and precompute one weight matrix
    per lag, so the whole delayed sum is a **single GEMM** against a stacked
    history buffer.  The quantisation error is not hidden: :attr:`delay_rmse_s`
    reports its RMS in seconds and callers add it to the numerical budget
    (thesis §4.5 -- coarsening is a reported uncertainty contribution).
    """

    weights: Tensor  # (N,N) or (B,N,N)
    delay_s: Tensor  # (N,N) or (B,N,N) seconds
    dt: float
    n_bins: int = 16
    device: torch.device | None = None

    def __post_init__(self) -> None:
        W = self.weights
        D = self.delay_s
        if W.ndim == 2:
            W = W.unsqueeze(0)
        if D.ndim == 2:
            D = D.unsqueeze(0)
        self.batched = W.shape[0] > 1 or D.shape[0] > 1
        dev = self.device or W.device
        W = W.to(dev, torch.float32)
        D = D.to(dev, torch.float32)
        self.n_regions = W.shape[-1]
        steps = torch.round(D / self.dt).clamp_min(0)
        present = W > 0
        maxstep = int(steps[present].max().item()) if present.any() else 0
        self.max_lag = max(int(maxstep), 1)
        # Quantile bin edges over the *realised* edge delays: for a connectome the
        # delay distribution is strongly peaked, so equal-count bins cut the
        # quantisation RMSE several-fold versus uniform bins at the same cost.
        Bw = int(max(W.shape[0], D.shape[0]))
        W = W.expand(Bw, -1, -1).contiguous()
        steps = steps.expand(Bw, -1, -1).contiguous()
        present = present.expand(Bw, -1, -1)
        nb_req = int(min(self.n_bins, self.max_lag + 1))
        if present.any():
            q = torch.linspace(0, 1, nb_req + 1, device=dev)
            edges = torch.quantile(steps[present].float(), q)
            edges[0], edges[-1] = -0.5, float(self.max_lag) + 0.5
            edges = torch.unique(torch.cummax(edges, 0).values)
        else:
            edges = torch.linspace(-0.5, self.max_lag + 0.5, nb_req + 1, device=dev)
        nb = int(edges.numel() - 1)
        idx = torch.bucketize(steps, edges[1:-1].contiguous(), right=False).clamp(0, nb - 1)
        # representative lag per bin = mean realised lag in that bin (minimises RMSE)
        w_edge = present.float()
        cnt = torch.zeros(nb, device=dev).scatter_add_(0, idx.reshape(-1), w_edge.reshape(-1))
        tot = torch.zeros(nb, device=dev).scatter_add_(0, idx.reshape(-1), (steps * w_edge).reshape(-1))
        self.bin_lag = torch.where(cnt > 0, torch.round(tot / cnt.clamp_min(1)), torch.round(0.5 * (edges[:-1] + edges[1:]))).long().clamp(0, self.max_lag)
        self.n_bins_used = nb
        N = self.n_regions
        # scatter W into per-bin planes without a Python loop over edges
        Wb = torch.zeros(Bw, nb, N, N, device=dev, dtype=torch.float32)
        flat = Wb.reshape(Bw, nb, N * N)
        flat.scatter_(1, idx.reshape(Bw, 1, N * N), W.reshape(Bw, 1, N * N))
        # (B, N, nb*N) laid out so coupling = hist_flat @ Wcat^T
        self.Wcat = Wb.permute(0, 2, 1, 3).reshape(Bw, N, nb * N).contiguous()
        D = D.expand(Bw, -1, -1)
        realised = self.bin_lag.to(dev)[idx] * self.dt
        err = (realised - D)[present]
        self.delay_rmse_s = float(torch.sqrt((err**2).mean()).item()) if err.numel() else 0.0
        self.delay_max_abs_err_s = float(err.abs().max().item()) if err.numel() else 0.0
        self._dev = dev

    # -- history buffer ---------------------------------------------------
    def new_history(self, batch: int, n_channels: int = 1, *, fill: float = 0.0) -> Tensor:
        """Ring buffer ``(B, L, N, C)`` with ``L = max_lag + 1``."""
        return torch.full(
            (batch, self.max_lag + 1, self.n_regions, n_channels), fill, device=self._dev, dtype=torch.float32
        )

    def gather(self, history: Tensor, t: int) -> Tensor:
        """Stack the ``n_bins_used`` delayed slices: ``(B, nb, N, C)``."""
        L = history.shape[1]
        pos = (t - self.bin_lag) % L
        return history.index_select(1, pos)

    def __call__(self, history: Tensor, t: int) -> Tensor:
        """Coupling input ``(B, N, C)``."""
        assert_solver_dtype(history, "coupling history")
        h = self.gather(history, t)  # (B, nb, N, C)
        B, nb, N, C = h.shape
        W = self.Wcat
        hf = h.permute(0, 3, 1, 2).reshape(B, C, nb * N)  # (B, C, nb*N)
        if W.shape[0] == 1:
            out = hf.reshape(B * C, nb * N) @ W[0].transpose(0, 1)  # (B*C, N)
            out = out.reshape(B, C, N)
        else:
            out = torch.bmm(hf, W.transpose(1, 2))  # (B, C, N)
        return out.permute(0, 2, 1).contiguous()

    def budget(self) -> dict[str, float]:
        return {
            "delay_bins": float(self.n_bins_used),
            "delay_quantisation_rmse_s": self.delay_rmse_s,
            "delay_quantisation_max_abs_err_s": self.delay_max_abs_err_s,
            "max_lag_steps": float(self.max_lag),
        }


@torch.no_grad()
def integrate_sde(
    backend: DynamicsBackend,
    coupling: DelayedCoupling,
    theta: ParamPack,
    *,
    n_steps: int,
    dt: float,
    seed: int,
    n_regions: int,
    batch: int,
    store_every: int = 10,
    warmup_steps: int = 0,
    u_fn=None,
    device: torch.device | None = None,
    record: tuple[str, ...] = ("activity",),
) -> dict[str, Tensor]:
    """Euler-Maruyama integration of a delayed network SDE, fp32, on device.

    Returns ``{name: (B, T_stored, N)}`` for each requested observable plus
    ``'final_state'``.  Deterministic given ``seed``.
    """
    dev = device or coupling._dev
    x = backend.init_state(batch, n_regions, seed=seed, device=dev, theta=theta)
    hist = coupling.new_history(batch, backend.n_coupling_channels)
    L = hist.shape[1]
    g = make_generator(seed + 7919, dev)
    sqdt = dt**0.5
    total = warmup_steps + n_steps
    n_store = n_steps // store_every
    out = {k: torch.empty(batch, n_store, n_regions, device=dev, dtype=torch.float32) for k in record}
    cv = backend.coupling_variable(x, theta)
    hist[:, 0] = cv
    si = 0
    for t in range(total):
        c = coupling(hist, t)
        f = backend.drift(x, c, theta, u_fn(t * dt) if u_fn is not None else None, t * dt)
        gdiff = backend.diffusion(x, theta)
        noise = torch.randn(x.shape, generator=g, device=dev, dtype=torch.float32)
        x = x + dt * f + gdiff * sqdt * noise
        x = torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3).clamp(-1e3, 1e3)
        cv = backend.coupling_variable(x, theta)
        hist[:, (t + 1) % L] = cv
        if t >= warmup_steps and ((t - warmup_steps) % store_every == 0) and si < n_store:
            obs = backend.observables(x)
            for k in record:
                out[k][:, si] = obs[k]
            si += 1
    out["final_state"] = x
    return out
