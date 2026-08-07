"""Engineered per-family backends (§5, §5.1), exposed through ``DynamicsBackend``.

``scwbd/dynamics/subcortical.py`` and ``scwbd/dynamics/hippocampus.py`` implement
the thalamic relay/burst switch, the direct/indirect/hyperdirect basal-ganglia
motif, the cerebellar forward model and four hippocampal memory hypotheses.  They
are tested and, before this module, referenced by **zero** foundation code
(``reports/scope_gap.md`` G-3).  §5's whole argument — that these systems
"warrant more engineered regional backends than a generic transformer block" —
had no expression in the artifact.

This module is that expression.  Each class here is a thin, batched, fp32
``DynamicsBackend`` over the *existing* implementations, so that
``scwbd.foundation.families`` can assign one per family and
``scwbd.foundation.backends.resolve_backend`` can find it by name.

Every backend here holds **no learnable parameters** — the base class's rule for
a mechanistic backend.  Fixed random projections are registered buffers, and the
free parameters live in the batched :class:`ParamPack`.

Epistemic status, stated because it is easy to overstate: these are *engineered*,
not *validated*.  Each carries its own falsifier in ``BackendInfo``, and none of
them has been fitted to data in this repository.
"""

from __future__ import annotations

import math
from typing import ClassVar, Mapping

import torch
from torch import Tensor

from .base import BackendInfo, DynamicsBackend, Prior, register_backend
from .subcortical import BasalGangliaGate, Cerebellum, ThalamicRelay
from .types import DTYPE, ParamPack, make_generator

__all__ = [
    "ThalamicRelayBackend",
    "BasalGangliaBackend",
    "HippocampalCodeBackend",
    "CerebellarForwardBackend",
    "FAMILY_BACKENDS",
]


def _P(name, loc, scale, dist="normal", **kw) -> Prior:
    return Prior(name=name, loc=loc, scale=scale, dist=dist, **kw)


# ======================================================================
# thalamus
# ======================================================================
@register_backend
class ThalamicRelayBackend(DynamicsBackend):
    """Relay/burst switching with TRN inhibition — ``ThalamicRelay`` as a backend.

    State ``(m, h, trn)``: membrane-like relay variable, slow T-type calcium
    de-inactivation, and a reticular inhibitory pool.  The relay *mode* is a
    state, not a switch: hyperpolarisation de-inactivates the T-current, so the
    same cortical input produces tonic relay or a burst depending on history.
    That is the property a generic block cannot express and the reason §5 gives
    the thalamus its own backend.
    """

    info: ClassVar[BackendInfo] = BackendInfo(
        name="thalamic_relay",
        family="flow_ode",
        mechanistic_status="mechanistic",
        state_names=("m", "h", "trn"),
        units=("dimensionless", "dimensionless", "dimensionless"),
        reference="Sherman & Guillery 2006; Jahnsen & Llinas 1984 (T-current burst); scwbd.dynamics.subcortical.ThalamicRelay",
        falsifier=(
            "Thalamic responses whose burst/tonic ratio does not depend on the preceding "
            "hyperpolarisation history at any admissible tau_h — i.e. a relay whose mode is "
            "a free parameter rather than a state."
        ),
    )
    defaults: ClassVar[Mapping[str, float]] = {
        "tau_m": 0.010,
        "tau_h": 0.100,
        "tau_trn": 0.020,
        "burst_threshold": -0.3,
        "burst_gain": 3.0,
        "relay_gain": 1.0,
        "trn_gain": 1.0,
        "trn_drive": 0.6,
        "G": 0.6,
        "sigma": 0.01,
        "ei_ratio": 1.0,
    }
    param_priors: ClassVar[Mapping[str, Prior]] = {
        "G": _P("G", 0.6, 0.5, "uniform", low=0.02, high=2.0),
        "burst_gain": _P("burst_gain", 3.0, 1.0, "uniform", low=1.0, high=6.0),
        "tau_h": _P("tau_h", 0.10, 0.05, "uniform", low=0.03, high=0.30),
        "sigma": _P("sigma", -4.0, 0.7, "lognormal", low=1e-4, high=0.1),
        "ei_ratio": _P("ei_ratio", 1.0, 0.2, "normal", low=0.5, high=1.8),
    }
    regional_params: ClassVar[tuple[str, ...]] = ("ei_ratio", "burst_gain")

    def __init__(self) -> None:
        super().__init__()
        # the reference implementation, used for its output() semantics
        self._ref = ThalamicRelay(1)

    @property
    def state_dim(self) -> int:
        return 3

    def init_state(self, batch, n_regions, *, seed, device=None, dtype=DTYPE, theta=None):
        g = make_generator(seed, device)
        x = torch.zeros(batch, n_regions, 3, device=g.device, dtype=dtype)
        x[..., 1] = 0.2  # partially inactivated T-current at rest
        return x + 0.01 * torch.randn(x.shape, generator=g, device=g.device, dtype=dtype)

    def drift(self, x, coupling_input, theta, u=None, t=0.0):
        m, h, trn = x[..., 0:1], x[..., 1:2], x[..., 2:3]
        drive = theta.get("G") * coupling_input[..., 0:1]
        if u is not None:
            drive = drive + u[..., 0:1]
        inp = drive - theta.get("trn_gain") * trn
        dm = (-m + inp) / theta.get("tau_m")
        # hyperpolarisation de-inactivates the T-current (ThalamicRelay.drift)
        h_inf = torch.sigmoid(-(m - theta.get("burst_threshold")) * 10.0)
        dh = (h_inf - h) / theta.get("tau_h")
        # TRN is driven by relay output and by corticothalamic feedback
        relay = self.relay(x, theta)
        dtrn = (theta.get("trn_drive") * relay * theta.get("ei_ratio") - trn) / theta.get("tau_trn")
        return torch.cat([dm, dh, dtrn], dim=-1)

    def relay(self, x: Tensor, theta: ParamPack) -> Tensor:
        """Tonic + burst relay output, ``(..., 1)`` — the ``relay_out`` port."""
        m, h = x[..., 0:1], x[..., 1:2]
        tonic = torch.relu(m) * theta.get("relay_gain")
        burst = torch.relu(m) * h * theta.get("burst_gain")
        return tonic + burst

    def diffusion(self, x, theta):
        s = theta.get("sigma")
        z = torch.zeros_like(x[..., 0:1])
        return torch.cat([s.expand_as(z), 0.1 * s.expand_as(z), 0.5 * s.expand_as(z)], dim=-1)

    def observables(self, x):
        m, h, trn = x[..., 0], x[..., 1], x[..., 2]
        tonic = torch.relu(m)
        return {
            "activity": tonic,
            "rate_e": tonic,
            "rate_i": trn.clamp_min(0),
            "eeg_source": tonic - trn,
            "burst_readiness": h,
        }


# ======================================================================
# basal ganglia
# ======================================================================
@register_backend
class BasalGangliaBackend(DynamicsBackend):
    """Direct / indirect / hyperdirect gating — ``BasalGangliaGate`` as a backend.

    State ``(GPe, STN, GPi)``; the D1/D2 striatal rates and the disinhibition
    gate are observables of it.  Dopamine enters as a **receptor-typed gain on
    corticostriatal transmission with opposite sign at D1 and D2** (parameter
    ``dopamine``).  It is not a reward signal and nothing here consumes a reward
    — §5, and ``scwbd.dynamics.subcortical.FORBIDDEN_SEMANTICS`` makes the
    equation raise elsewhere in the package.
    """

    info: ClassVar[BackendInfo] = BackendInfo(
        name="basal_ganglia_gate",
        family="flow_ode",
        mechanistic_status="mechanistic",
        state_names=("gpe", "stn", "gpi"),
        units=("dimensionless",) * 3,
        reference="Frank 2006; Gurney/Prescott/Redgrave 2001; scwbd.dynamics.subcortical.BasalGangliaGate",
        falsifier=(
            "Selection behaviour whose dependence on dopamine has the SAME sign through the D1 and "
            "D2 pathways, or a stop signal whose latency is not shorter through STN than through "
            "the striatopallidal route."
        ),
    )
    defaults: ClassVar[Mapping[str, float]] = {
        "w_d1": 1.0,
        "w_d2": 1.0,
        "kappa_d1": 0.6,
        "kappa_d2": 0.6,
        "w_gpe": 1.0,
        "w_gpi": 1.0,
        "w_stn_gpe": 0.8,
        "w_gpe_stn": 1.0,
        "w_stn_gpi": 1.0,
        "w_hyper": 1.0,
        "gate_slope": 4.0,
        "gpi_baseline": 0.5,
        "tau": 0.020,
        "dopamine": 0.0,
        "G": 1.0,
        "sigma": 0.01,
        "ei_ratio": 1.0,
    }
    param_priors: ClassVar[Mapping[str, Prior]] = {
        "G": _P("G", 1.0, 0.6, "uniform", low=0.05, high=2.5),
        "dopamine": _P("dopamine", 0.0, 0.4, "uniform", low=-0.9, high=0.9),
        "gate_slope": _P("gate_slope", 4.0, 1.5, "uniform", low=1.0, high=8.0),
        "sigma": _P("sigma", -4.6, 0.6, "lognormal", low=1e-4, high=0.05),
        "ei_ratio": _P("ei_ratio", 1.0, 0.2, "normal", low=0.5, high=1.7),
    }
    regional_params: ClassVar[tuple[str, ...]] = ("ei_ratio", "dopamine")

    def __init__(self) -> None:
        super().__init__()
        self._ref = BasalGangliaGate(1)

    @property
    def state_dim(self) -> int:
        return 3

    @staticmethod
    def _f(x: Tensor) -> Tensor:
        return torch.relu(torch.tanh(x))

    def init_state(self, batch, n_regions, *, seed, device=None, dtype=DTYPE, theta=None):
        g = make_generator(seed, device)
        x = torch.zeros(batch, n_regions, 3, device=g.device, dtype=dtype)
        x[..., 2] = 0.5
        return x + 0.01 * torch.randn(x.shape, generator=g, device=g.device, dtype=dtype)

    def striatum(self, cortex: Tensor, theta: ParamPack) -> tuple[Tensor, Tensor]:
        """D1 and D2 projection rates.  Opposite-sign dopaminergic gain (§5)."""
        da = theta.get("dopamine")
        d1 = self._f(theta.get("w_d1") * (1.0 + theta.get("kappa_d1") * da) * cortex)
        d2 = self._f(theta.get("w_d2") * (1.0 - theta.get("kappa_d2") * da) * cortex)
        return d1, d2

    def drift(self, x, coupling_input, theta, u=None, t=0.0):
        gpe, stn, gpi = x[..., 0:1], x[..., 1:2], x[..., 2:3]
        cortex = theta.get("G") * coupling_input[..., 0:1] * theta.get("ei_ratio")
        if u is not None:
            cortex = cortex + u[..., 0:1]
        d1, d2 = self.striatum(cortex, theta)
        gpe_t = self._f(theta.get("w_gpe") - d2 - theta.get("w_stn_gpe") * stn)
        stn_t = self._f(theta.get("w_hyper") * cortex - theta.get("w_gpe_stn") * gpe)
        gpi_t = self._f(theta.get("w_gpi") - d1 + theta.get("w_stn_gpi") * stn_t)
        tau = theta.get("tau")
        return torch.cat([(gpe_t - gpe) / tau, (stn_t - stn) / tau, (gpi_t - gpi) / tau], dim=-1)

    def gate(self, x: Tensor, theta: ParamPack) -> Tensor:
        """Disinhibition gate in [0,1], ``(..., 1)`` — the ``gate_out`` port."""
        return torch.sigmoid(-theta.get("gate_slope") * (x[..., 2:3] - theta.get("gpi_baseline")))

    def diffusion(self, x, theta):
        return theta.get("sigma").expand_as(x)

    def observables(self, x):
        stn, gpi = x[..., 1], x[..., 2]
        gate = torch.sigmoid(-4.0 * (gpi - 0.5))
        return {
            "activity": gate,
            "rate_e": stn,
            "rate_i": gpi,
            "eeg_source": stn - gpi,
            "gate": gate,
            "hyperdirect_stop": stn,
        }


# ======================================================================
# hippocampus
# ======================================================================
@register_backend
class HippocampalCodeBackend(DynamicsBackend):
    """``H_t = {k, v, g, c, rho}`` as a regional flow (body.tex §5.1).

    The one family whose *state space* body.tex specifies component by component,
    and therefore the concrete reason a single ``D`` for every parcel is
    non-conformant with §2.1.

    * ``c`` — leaky contextual integrator of the afferent drive (slow).
    * ``g`` — a multiscale relational code: ``c`` advances the phase of
      ``n_modules`` independent 2-D rotors with incommensurate frequencies.  That
      is the Vector-HaSH scaffold claim in continuous time: the scaffold is
      *fixed*, so it does not interfere as items accumulate.
    * ``k`` — the cue, a fixed random projection of the afferent drive and ``g``.
    * ``v`` — bound content, relaxing toward a modern-Hopfield read of ``k``
      against a fixed codebook (softmax similarity, Ramsauer et al. 2020).
    * ``rho`` — retrieval confidence: the max softmax mass, low-pass filtered.

    **What this is not.**  The episodic write/read store
    (``scwbd.dynamics.hippocampus.HippocampalBackend`` and its four subclasses,
    with the capacity / interference / cue-degradation / replay benchmark) is
    *not* driven by this backend.  Those hypotheses are compared offline by
    ``compare_backends``; here the retrieval is against a fixed codebook, so the
    rollout expresses the state *shape* and the confidence channel but not
    episodic storage.  Declared narrowing **`hippocampal-codebook`**.
    """

    info: ClassVar[BackendInfo] = BackendInfo(
        name="hippocampal_code",
        family="attention",
        mechanistic_status="effective",
        state_names=("k", "v", "g", "c", "rho"),
        units=("dimensionless", "dimensionless", "dimensionless", "dimensionless", "probability"),
        reference="body.tex §5.1; Ramsauer et al. 2020; Chandra et al. 2025 (Vector-HaSH); Moser et al. 2015",
        falsifier=(
            "Retrieval confidence that does not fall with cue degradation, or a capacity that "
            "scales with the number of stored items rather than with the scaffold — either kills "
            "the fixed-scaffold reading of §5.1. Status is 'effective', not 'mechanistic': no "
            "cellular claim is made and the algorithmic claim is not yet fitted to data."
        ),
    )
    #: two coupling channels: an afferent cue and a contextual drive
    n_coupling_channels: ClassVar[int] = 2
    defaults: ClassVar[Mapping[str, float]] = {
        "tau_c": 0.250,
        "tau_k": 0.020,
        "tau_v": 0.040,
        "tau_rho": 0.100,
        "beta": 8.0,
        "theta_hz": 7.0,
        "G": 0.8,
        "sigma": 0.01,
        "ei_ratio": 1.0,
    }
    param_priors: ClassVar[Mapping[str, Prior]] = {
        "G": _P("G", 0.8, 0.5, "uniform", low=0.05, high=2.0),
        "beta": _P("beta", 8.0, 4.0, "uniform", low=1.0, high=24.0),
        "theta_hz": _P("theta_hz", 7.0, 1.5, "uniform", low=4.0, high=10.0),
        "tau_c": _P("tau_c", 0.25, 0.15, "uniform", low=0.05, high=1.0),
        "sigma": _P("sigma", -4.6, 0.6, "lognormal", low=1e-4, high=0.05),
    }
    regional_params: ClassVar[tuple[str, ...]] = ("theta_hz", "beta")

    def __init__(
        self,
        *,
        d_key: int = 16,
        d_value: int = 16,
        d_grid: int = 12,
        d_context: int = 4,
        n_patterns: int = 64,
        seed: int = 20260806,
    ) -> None:
        super().__init__()
        if d_grid % 2:
            raise ValueError(f"d_grid must be even (it is n_modules 2-D rotors); got {d_grid}")
        self.d_key, self.d_value = int(d_key), int(d_value)
        self.d_grid, self.d_context = int(d_grid), int(d_context)
        self.n_modules = self.d_grid // 2
        self.n_patterns = int(n_patterns)
        g = make_generator(seed, "cpu")
        # fixed scaffold: incommensurate module frequencies (the "multiscale" in g)
        ratios = torch.tensor([1.0, 1.6180339887, 2.4142135624, 3.3027756377, 4.2360679775, 5.1925824036])
        w = ratios.repeat((self.n_modules // len(ratios)) + 1)[: self.n_modules]
        self.register_buffer("module_ratio", w)
        # fixed random codebook: keys and their bound values (no learnable params)
        K = torch.randn(self.n_patterns, self.d_key, generator=g) / math.sqrt(self.d_key)
        V = torch.randn(self.n_patterns, self.d_value, generator=g) / math.sqrt(self.d_value)
        self.register_buffer("codebook_k", K / K.norm(dim=-1, keepdim=True).clamp_min(1e-8))
        self.register_buffer("codebook_v", V)
        # fixed cue projection: (afferent, context, grid) -> key
        self.register_buffer(
            "W_cue", torch.randn(1 + self.d_context + self.d_grid, self.d_key, generator=g) / math.sqrt(self.d_key)
        )

    # -- layout ------------------------------------------------------------
    @property
    def state_dim(self) -> int:
        return self.d_key + self.d_value + self.d_grid + self.d_context + 1

    def _split(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        a = self.d_key
        b = a + self.d_value
        c = b + self.d_grid
        d = c + self.d_context
        return x[..., :a], x[..., a:b], x[..., b:c], x[..., c:d], x[..., d : d + 1]

    def init_state(self, batch, n_regions, *, seed, device=None, dtype=DTYPE, theta=None):
        gen = make_generator(seed, device)
        x = 0.05 * torch.randn(batch, n_regions, self.state_dim, generator=gen, device=gen.device, dtype=dtype)
        k, v, g, c, rho = self._split(x)
        # grid rotors start on the unit circle so the scaffold is non-degenerate
        g = g.reshape(*g.shape[:-1], self.n_modules, 2)
        g = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.cat([k, v, g.reshape(*x.shape[:-1], self.d_grid), c, rho.sigmoid()], dim=-1)

    # -- retrieval ---------------------------------------------------------
    def read(self, k: Tensor, theta: ParamPack) -> tuple[Tensor, Tensor]:
        """Modern-Hopfield read: ``(v_hat, rho)`` from the fixed codebook."""
        q = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        logits = theta.get("beta") * (q @ self.codebook_k.to(q.dtype).T)
        p = torch.softmax(logits, dim=-1)
        return p @ self.codebook_v.to(q.dtype), p.max(dim=-1, keepdim=True).values

    def drift(self, x, coupling_input, theta, u=None, t=0.0):
        k, v, g, c, rho = self._split(x)
        cue_in = coupling_input[..., 0:1] * theta.get("G")
        ctx_in = coupling_input[..., 1:2] if coupling_input.shape[-1] > 1 else cue_in
        if u is not None:
            cue_in = cue_in + u[..., 0:1]

        # c: leaky contextual integration
        dc = (ctx_in.expand_as(c) - c) / theta.get("tau_c")

        # g: fixed multiscale scaffold. Each module is a rotor whose angular
        # velocity is (theta rhythm) * (module ratio) * (1 + contextual drive).
        rot = g.reshape(*g.shape[:-1], self.n_modules, 2)
        omega = (
            2 * math.pi * theta.get("theta_hz") * (1.0 + 0.25 * c.mean(dim=-1, keepdim=True))
        ).unsqueeze(-1) * self.module_ratio.to(g.dtype).reshape(1, 1, -1, 1)
        # d/dt (cos, sin) = omega * (-sin, cos), plus radial pull back to |g|=1
        drot = torch.stack([-rot[..., 1], rot[..., 0]], dim=-1) * omega
        radial = (1.0 - rot.pow(2).sum(-1, keepdim=True)) * rot * 5.0
        dg = (drot + radial).reshape(*g.shape)

        # k: cue = fixed random projection of (afferent, context, grid)
        k_t = torch.cat([cue_in, c, g], dim=-1) @ self.W_cue.to(x.dtype)
        dk = (k_t - k) / theta.get("tau_k")

        # v, rho: attractor relaxation toward the retrieved content
        v_hat, p_max = self.read(k, theta)
        dv = (v_hat - v) / theta.get("tau_v")
        drho = (p_max - rho) / theta.get("tau_rho")
        return torch.cat([dk, dv, dg, dc, drho], dim=-1)

    def diffusion(self, x, theta):
        s = theta.get("sigma")
        k, v, g, c, rho = self._split(x)
        return torch.cat(
            [
                s.expand_as(k),
                s.expand_as(v),
                torch.zeros_like(g),  # the scaffold is fixed: noise on it is not a hypothesis
                0.5 * s.expand_as(c),
                torch.zeros_like(rho),
            ],
            dim=-1,
        )

    def observables(self, x):
        k, v, g, c, rho = self._split(x)
        act = v.mean(dim=-1)
        return {
            "activity": act,
            "rate_e": act.clamp_min(0),
            "rate_i": (-act).clamp_min(0),
            "eeg_source": act,
            "retrieval_confidence": rho.squeeze(-1),
        }

    def coupling_variable(self, x, theta=None):
        k, v, g, c, rho = self._split(x)
        return torch.cat([v.mean(dim=-1, keepdim=True), c.mean(dim=-1, keepdim=True)], dim=-1)


# ======================================================================
# cerebellum
# ======================================================================
@register_backend
class CerebellarForwardBackend(DynamicsBackend):
    """Forward model + climbing-fibre-timed residual correction (``Cerebellum``).

    State ``(prediction, error, eligibility)``.  The granule expansion is the
    *same* fixed random projection ``scwbd.dynamics.subcortical.Cerebellum`` uses
    (mossy fibre -> granule, tanh + mean subtraction), and the Purkinje readout
    is a fixed random contraction rather than a delta-rule-learned matrix: the
    delta rule in ``Cerebellum.learn`` is an offline ``@torch.no_grad`` update
    over an explicit history buffer, which a differentiable rollout cannot carry.
    The **eligibility trace** carries the ``error_delay`` that the delta rule
    depends on, so the timing structure survives.  Declared narrowing **`cerebellar-readout`**.

    ``mechanistic_status`` is ``effective``, matching ``Cerebellum.falsifier``.
    """

    info: ClassVar[BackendInfo] = BackendInfo(
        name="cerebellar_forward_model",
        family="surrogate",
        mechanistic_status="effective",
        state_names=("prediction", "error", "eligibility"),
        units=("dimensionless",) * 3,
        reference="Marr 1969; Albus 1971; Wolpert et al. 1998; scwbd.dynamics.subcortical.Cerebellum",
        falsifier=(
            "Adaptation timing or extinction asymmetries that a capacity-matched supervised "
            "regressor with the same inputs reproduces equally well — verbatim from "
            "Cerebellum.falsifier. Nothing here has been run against that control."
        ),
    )
    n_coupling_channels: ClassVar[int] = 2
    defaults: ClassVar[Mapping[str, float]] = {
        "tau_pred": 0.015,
        "tau_err": 0.050,
        "tau_elig": 0.100,
        "gain": 1.0,
        "G": 0.7,
        "sigma": 0.01,
        "ei_ratio": 1.0,
    }
    param_priors: ClassVar[Mapping[str, Prior]] = {
        "G": _P("G", 0.7, 0.5, "uniform", low=0.05, high=2.0),
        "tau_elig": _P("tau_elig", 0.10, 0.05, "uniform", low=0.02, high=0.30),
        "gain": _P("gain", 1.0, 0.4, "uniform", low=0.2, high=2.0),
        "sigma": _P("sigma", -4.6, 0.6, "lognormal", low=1e-4, high=0.05),
    }
    regional_params: ClassVar[tuple[str, ...]] = ("gain", "tau_elig")

    def __init__(self, *, d_prediction: int = 8, n_granule: int = 128, seed: int = 20260806) -> None:
        super().__init__()
        self.d_prediction = int(d_prediction)
        self.n_granule = int(n_granule)
        # the reference implementation supplies the granule expansion verbatim
        self._gr = Cerebellum(d_input=2, d_output=self.d_prediction, n_granule=self.n_granule, seed=seed, device="cpu")
        g = make_generator(seed + 1, "cpu")
        self.register_buffer(
            "W_pc", torch.randn(self.n_granule, self.d_prediction, generator=g) / math.sqrt(self.n_granule)
        )

    @property
    def state_dim(self) -> int:
        return 3 * self.d_prediction

    def _split(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        d = self.d_prediction
        return x[..., :d], x[..., d : 2 * d], x[..., 2 * d : 3 * d]

    def granule(self, mossy: Tensor) -> Tensor:
        """Mossy-fibre -> granule expansion, verbatim from ``Cerebellum.granule``."""
        W = self._gr.W_mf_gr.to(mossy.device, mossy.dtype)
        b = self._gr.b_gr.to(mossy.device, mossy.dtype)
        h = torch.tanh(mossy @ W + b)
        return torch.relu(h - h.mean(dim=-1, keepdim=True))

    def init_state(self, batch, n_regions, *, seed, device=None, dtype=DTYPE, theta=None):
        gen = make_generator(seed, device)
        return 0.02 * torch.randn(batch, n_regions, self.state_dim, generator=gen, device=gen.device, dtype=dtype)

    def drift(self, x, coupling_input, theta, u=None, t=0.0):
        pred, err, elig = self._split(x)
        mossy = coupling_input[..., 0:1] * theta.get("G")
        climbing = coupling_input[..., 1:2] if coupling_input.shape[-1] > 1 else torch.zeros_like(mossy)
        if u is not None:
            mossy = mossy + u[..., 0:1]
        gr = self.granule(torch.cat([mossy, climbing], dim=-1))
        target = theta.get("gain") * (gr @ self.W_pc.to(x.dtype))
        dpred = (target - pred) / theta.get("tau_pred")
        # climbing-fibre error: observed minus predicted, low-passed
        derr = (climbing.expand_as(err) - pred - err) / theta.get("tau_err")
        # eligibility carries the delay the delta rule in Cerebellum.learn needs
        delig = (err - elig) / theta.get("tau_elig")
        return torch.cat([dpred, derr, delig], dim=-1)

    def diffusion(self, x, theta):
        s = theta.get("sigma")
        pred, err, elig = self._split(x)
        return torch.cat([s.expand_as(pred), 0.5 * s.expand_as(err), torch.zeros_like(elig)], dim=-1)

    def observables(self, x):
        pred, err, elig = self._split(x)
        act = pred.mean(dim=-1)
        return {
            "activity": act,
            "rate_e": act.clamp_min(0),
            "rate_i": (-act).clamp_min(0),
            "eeg_source": act,
            "prediction_error": err.pow(2).mean(dim=-1),
        }

    def coupling_variable(self, x, theta=None):
        pred, err, elig = self._split(x)
        return torch.cat([pred.mean(dim=-1, keepdim=True), err.mean(dim=-1, keepdim=True)], dim=-1)


#: Backend names this module registers, by family kind.  ``scwbd.foundation.
#: families.DEFAULT_FAMILY_CORES`` refers to these names; if this module is not
#: imported, ``resolve_backend`` raises rather than silently falling back to the
#: generic core (that is the point of R12).
FAMILY_BACKENDS: dict[str, str] = {
    "thalamus": "thalamic_relay",
    "basal_ganglia": "basal_ganglia_gate",
    "hippocampus": "hippocampal_code",
    "cerebellum": "cerebellar_forward_model",
}
