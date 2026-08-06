"""Engineered subcortical, cerebellar and neuromodulatory backends (§5).

The thesis refuses one undifferentiated limbic module: these systems have
comparatively constrained circuit motifs, inputs, outputs and physiological
variables, so they get engineered backends rather than a generic block.  What
they do **not** get is a semantic shortcut.  From §5, verbatim in spirit:

    "Dopamine is not equated with reward, serotonin with punishment,
    acetylcholine with attention, or norepinephrine with arousal.
    Neuromodulators are receptor-, target-, timescale-, and state-dependent
    control fields that alter gain, plasticity, exploration, persistence, and
    uncertainty."

:class:`NeuromodulatoryField` enforces that: attaching a psychological label to
a transmitter raises :class:`~scwbd.dynamics.types.SemanticCollapseError`.  The
refusal is executable, not a comment.

Contents
--------
``ThalamicRelay``      tonic-relay / burst switching, TRN inhibition, routing gain
``BasalGangliaGate``   direct (D1) / indirect (D2) / hyperdirect (STN) pathways
``Cerebellum``         forward model + error-driven residual correction
``NeuromodulatoryField``  receptor-typed, target-specific gain & plasticity fields
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Iterable, Literal, Mapping, Sequence

import torch
from torch import Tensor, nn

from .types import DTYPE, ParamPack, SemanticCollapseError, default_device, make_generator

__all__ = [
    "ThalamicRelay",
    "BasalGangliaGate",
    "Cerebellum",
    "ReceptorSpec",
    "NeuromodulatoryField",
    "NeuromodulatorBank",
    "FORBIDDEN_SEMANTICS",
]


# ---------------------------------------------------------------------------
# Thalamus
# ---------------------------------------------------------------------------


class ThalamicRelay(nn.Module):
    """Relay/burst switching with reticular (TRN) inhibition and routing gain.

    Two-state relay cell per channel: membrane-like variable ``m`` and a slow
    T-type calcium de-inactivation variable ``h``.  Hyperpolarisation
    de-inactivates the T-current; a subsequent input then produces a **burst**
    rather than a linear relay of the input.  The mode is therefore a *state*,
    not a switch someone sets — which is the point of modelling it at all: the
    same cortical input has different consequences depending on thalamic state.

    Routing is a separate, explicit gain per (channel, target), so "thalamic
    routing" is a testable operator and not a metaphor.
    """

    state_dim: ClassVar[int] = 2  # (m, h)

    def __init__(
        self,
        n_channels: int,
        *,
        tau_m: float = 0.010,
        tau_h: float = 0.100,
        burst_threshold: float = -0.3,
        burst_gain: float = 3.0,
        relay_gain: float = 1.0,
        trn_gain: float = 1.0,
        device=None,
    ):
        super().__init__()
        self.n_channels = int(n_channels)
        self.tau_m, self.tau_h = float(tau_m), float(tau_h)
        self.burst_threshold = float(burst_threshold)
        self.burst_gain, self.relay_gain = float(burst_gain), float(relay_gain)
        self.trn_gain = float(trn_gain)
        self.device_ = default_device(device)

    def init_state(self, batch: int, *, device=None, dtype: torch.dtype = DTYPE) -> Tensor:
        dev = default_device(device or self.device_)
        x = torch.zeros(batch, self.n_channels, 2, device=dev, dtype=dtype)
        x[..., 1] = 0.2  # partially inactivated T-current at rest
        return x

    def drift(self, x: Tensor, drive: Tensor, trn: Tensor | None = None, u: Tensor | None = None) -> Tensor:
        """``x``: ``(B, C, 2)``, ``drive``: ``(B, C)`` cortical/sensory input."""
        m = x[..., 0:1]
        h = x[..., 1:2]
        inp = drive.unsqueeze(-1)
        if trn is not None:
            inp = inp - self.trn_gain * trn.unsqueeze(-1)
        if u is not None:
            inp = inp + (u.unsqueeze(-1) if u.ndim == 2 else u)
        dm = (-m + inp) / self.tau_m
        h_inf = torch.sigmoid(-(m - self.burst_threshold) * 10.0)  # hyperpolarisation de-inactivates
        dh = (h_inf - h) / self.tau_h
        return torch.cat([dm, dh], dim=-1)

    def output(self, x: Tensor) -> dict[str, Tensor]:
        m, h = x[..., 0], x[..., 1]
        tonic = torch.relu(m) * self.relay_gain
        burst = torch.relu(m) * h * self.burst_gain
        mode = h  # in [0,1]: 0 = tonic relay, 1 = burst-ready
        return {"relay": tonic + burst, "tonic": tonic, "burst": burst, "burst_readiness": mode}

    def route(self, relay: Tensor, routing_gain: Tensor) -> Tensor:
        """``relay``: ``(B, C)``; ``routing_gain``: ``(B, C, T)`` -> targets ``(B, T)``."""
        return torch.einsum("bc,bct->bt", relay, routing_gain)


# ---------------------------------------------------------------------------
# Basal ganglia
# ---------------------------------------------------------------------------


class BasalGangliaGate(nn.Module):
    """Direct / indirect / hyperdirect gating over action or WM channels.

    Rate model, per channel ``i``:

        striatum_D1 = f(w_d1 * (1 + kappa_D1 * DA) * cortex)
        striatum_D2 = f(w_d2 * (1 - kappa_D2 * DA) * cortex)
        GPe  = f(w_gpe  - striatum_D2 - w_stn_gpe * STN)
        STN  = f(w_hyper * cortex_global - w_gpe_stn * GPe)      # hyperdirect
        GPi  = f(w_gpi - striatum_D1 + w_stn_gpi * STN)
        gate = sigmoid(-slope * (GPi - GPi_baseline))            # disinhibition

    Dopamine here is a **gain on corticostriatal transmission with opposite sign
    at D1 and D2 receptors** — a receptor-typed control parameter.  It is not a
    reward signal, and nothing in this class consumes a reward.
    """

    def __init__(
        self,
        n_channels: int,
        *,
        w_d1: float = 1.0,
        w_d2: float = 1.0,
        kappa_d1: float = 0.6,
        kappa_d2: float = 0.6,
        w_gpe: float = 1.0,
        w_gpi: float = 1.0,
        w_stn_gpe: float = 0.8,
        w_gpe_stn: float = 1.0,
        w_stn_gpi: float = 1.0,
        w_hyper: float = 1.0,
        gate_slope: float = 4.0,
        gpi_baseline: float = 0.5,
        tau: float = 0.020,
        device=None,
    ):
        super().__init__()
        self.n_channels = int(n_channels)
        for k, v in dict(
            w_d1=w_d1, w_d2=w_d2, kappa_d1=kappa_d1, kappa_d2=kappa_d2, w_gpe=w_gpe, w_gpi=w_gpi,
            w_stn_gpe=w_stn_gpe, w_gpe_stn=w_gpe_stn, w_stn_gpi=w_stn_gpi, w_hyper=w_hyper,
            gate_slope=gate_slope, gpi_baseline=gpi_baseline, tau=tau,
        ).items():
            setattr(self, k, float(v))
        self.device_ = default_device(device)

    @staticmethod
    def _f(x: Tensor) -> Tensor:
        return torch.relu(torch.tanh(x))

    def init_state(self, batch: int, *, device=None, dtype: torch.dtype = DTYPE) -> Tensor:
        dev = default_device(device or self.device_)
        # (B, C, 3) = (GPe, STN, GPi)
        x = torch.zeros(batch, self.n_channels, 3, device=dev, dtype=dtype)
        x[..., 2] = self.gpi_baseline
        return x

    def forward(
        self,
        x: Tensor,
        cortex: Tensor,
        *,
        dopamine_gain: Tensor | float = 0.0,
        dt: float | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """One update.  ``cortex``: ``(B, C)`` per-channel cortical drive.

        Returns ``(new_state, outputs)``; with ``dt=None`` the internal nuclei
        are solved at their (fast) quasi-steady state, which is the usual
        approximation when the BG loop is faster than the cortical clock.
        """
        da = dopamine_gain if isinstance(dopamine_gain, Tensor) else torch.as_tensor(
            float(dopamine_gain), device=cortex.device, dtype=cortex.dtype
        )
        da = da.reshape(-1, 1) if da.ndim == 1 else da
        d1 = self._f(self.w_d1 * (1.0 + self.kappa_d1 * da) * cortex)
        d2 = self._f(self.w_d2 * (1.0 - self.kappa_d2 * da) * cortex)
        gpe_prev, stn_prev, gpi_prev = x[..., 0], x[..., 1], x[..., 2]
        cortex_global = cortex.mean(dim=-1, keepdim=True).expand_as(cortex)
        gpe_t = self._f(self.w_gpe - d2 - self.w_stn_gpe * stn_prev)
        stn_t = self._f(self.w_hyper * cortex_global - self.w_gpe_stn * gpe_prev)
        gpi_t = self._f(self.w_gpi - d1 + self.w_stn_gpi * stn_t)
        if dt is None:
            gpe, stn, gpi = gpe_t, stn_t, gpi_t
        else:
            a = min(dt / self.tau, 1.0)
            gpe = gpe_prev + a * (gpe_t - gpe_prev)
            stn = stn_prev + a * (stn_t - stn_prev)
            gpi = gpi_prev + a * (gpi_t - gpi_prev)
        gate = torch.sigmoid(-self.gate_slope * (gpi - self.gpi_baseline))
        new_state = torch.stack([gpe, stn, gpi], dim=-1)
        return new_state, {
            "gate": gate,
            "d1": d1,
            "d2": d2,
            "gpe": gpe,
            "stn": stn,
            "gpi": gpi,
            "hyperdirect_stop": stn,
        }


# ---------------------------------------------------------------------------
# Cerebellum
# ---------------------------------------------------------------------------


class Cerebellum(nn.Module):
    """Forward model + error-driven residual correction.

    A granule-layer random expansion followed by a linear Purkinje readout,
    trained by a delta rule on the *climbing-fibre* error (the observed outcome
    minus the predicted one), delayed by ``error_delay`` steps.  Two outputs:

    ``prediction``  the forward model's estimate of the plant's next state
    ``correction``  a residual control signal driving the plant towards target

    The functional analogy is strong in constrained domains and the
    implementation remains multilevel — so the label here is ``effective``, not
    ``mechanistic``, and the benchmark that would earn the stronger label is
    stated in :attr:`falsifier`.
    """

    falsifier: ClassVar[str] = (
        "Earns a mechanistic label only if the climbing-fibre-timed error rule predicts "
        "adaptation timing/extinction asymmetries that a capacity-matched supervised "
        "regressor with the same inputs does not."
    )

    def __init__(
        self,
        d_input: int,
        d_output: int,
        *,
        n_granule: int = 512,
        lr: float = 1e-2,
        error_delay: int = 2,
        leak: float = 1e-4,
        seed: int = 0,
        device=None,
    ):
        super().__init__()
        dev = default_device(device)
        g = make_generator(seed, dev)
        self.d_input, self.d_output = int(d_input), int(d_output)
        self.n_granule = int(n_granule)
        self.lr, self.leak = float(lr), float(leak)
        self.error_delay = int(error_delay)
        self.register_buffer(
            "W_mf_gr",
            torch.randn(d_input, n_granule, generator=g, device=dev, dtype=DTYPE) / math.sqrt(d_input),
        )
        self.register_buffer("b_gr", torch.zeros(n_granule, device=dev, dtype=DTYPE))
        self.register_buffer("W_pc", torch.zeros(n_granule, d_output, device=dev, dtype=DTYPE))
        self._history: list[tuple[Tensor, Tensor]] = []
        self.n_updates = 0

    def granule(self, x: Tensor) -> Tensor:
        """Sparse random expansion (mossy fibre -> granule cell)."""
        h = torch.tanh(x @ self.W_mf_gr + self.b_gr)
        return torch.relu(h - h.mean(dim=-1, keepdim=True))

    def predict(self, x: Tensor) -> Tensor:
        return self.granule(x) @ self.W_pc

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        gr = self.granule(x)
        pred = gr @ self.W_pc
        return {"prediction": pred, "granule": gr, "correction": pred}

    @torch.no_grad()
    def learn(self, x: Tensor, target: Tensor) -> Tensor:
        """Delta rule on the delayed climbing-fibre error.  Returns the error."""
        gr = self.granule(x)
        self._history.append((gr, target))
        if len(self._history) <= self.error_delay:
            return torch.zeros_like(target)
        gr_d, target_d = self._history.pop(0)
        err = target_d - gr_d @ self.W_pc
        self.W_pc += self.lr * (gr_d.T @ err) / max(gr_d.shape[0], 1)
        self.W_pc -= self.leak * self.W_pc
        self.n_updates += 1
        return err

    def reset(self) -> None:
        self._history.clear()


# ---------------------------------------------------------------------------
# Neuromodulation
# ---------------------------------------------------------------------------


#: Labels that collapse a neuromodulator into a psychological construct.  §5 is
#: explicit that this equation is not made; attempting it is a refusal.
FORBIDDEN_SEMANTICS: dict[str, tuple[str, ...]] = {
    "dopamine": ("reward", "value", "reward_prediction_error", "rpe", "pleasure", "motivation"),
    "serotonin": ("punishment", "mood", "happiness", "aversion"),
    "acetylcholine": ("attention", "attentional_gain", "focus"),
    "norepinephrine": ("arousal", "alertness", "vigilance"),
    "noradrenaline": ("arousal", "alertness", "vigilance"),
    "histamine": ("wakefulness",),
    "orexin": ("wakefulness", "arousal"),
}


@dataclass(frozen=True)
class ReceptorSpec:
    """One receptor's effect: which targets, on what, over what timescale.

    A neuromodulator's effect is not a scalar.  The same transmitter acting at
    two receptors on two targets can change gain in opposite directions with
    different time constants — which is precisely why "dopamine = reward" is
    unusable for intervention modelling.
    """

    receptor: str  # e.g. "D1", "D2", "alpha2A", "M1", "nAChR"
    targets: tuple[str, ...]  # region/population ids this receptor is expressed on
    gain_effect: float  # multiplicative gain change per unit occupancy (signed)
    plasticity_effect: float = 0.0  # learning-rate modulation per unit occupancy
    tau: float = 1.0  # s, effect time constant
    ec50: float = 0.5  # concentration at half occupancy
    hill: float = 1.0
    state_dependence: Literal["none", "activity", "phase"] = "none"
    density: Tensor | None = None  # (N,) receptor density prior from agent C

    def occupancy(self, concentration: Tensor) -> Tensor:
        c = concentration.clamp_min(0.0)
        return c.pow(self.hill) / (c.pow(self.hill) + self.ec50**self.hill)


class NeuromodulatoryField(nn.Module):
    """A receptor-, target-, timescale- and state-dependent control field.

    ``concentration`` evolves with its own (slow) kinetics from a source drive;
    each receptor converts concentration into occupancy, and occupancy into a
    **gain** and a **plasticity-rate** modulation on its declared targets, with
    its own time constant and optional state dependence.

    Constructing one with a psychological ``semantics`` label raises
    :class:`SemanticCollapseError`.
    """

    def __init__(
        self,
        name: str,
        n_regions: int,
        receptors: Sequence[ReceptorSpec],
        *,
        tau_release: float = 0.5,
        baseline: float = 0.1,
        semantics: str | None = None,
        device=None,
        batch: int = 1,
    ):
        super().__init__()
        self.name = str(name).lower()
        self._check_semantics(semantics)
        if not receptors:
            raise ValueError(
                f"neuromodulator {name!r} needs at least one ReceptorSpec: an unreceptored "
                "modulator is a scalar, which is exactly the collapse §5 forbids"
            )
        dev = default_device(device)
        self.n_regions = int(n_regions)
        self.receptors = list(receptors)
        self.tau_release = float(tau_release)
        self.baseline = float(baseline)
        self.register_buffer(
            "concentration", torch.full((batch, n_regions), float(baseline), device=dev, dtype=DTYPE)
        )
        self.register_buffer("gain", torch.ones(batch, n_regions, device=dev, dtype=DTYPE))
        self.register_buffer(
            "plasticity_rate", torch.ones(batch, n_regions, device=dev, dtype=DTYPE)
        )
        self.register_buffer(
            "target_masks",
            torch.stack([self._mask(r, n_regions, dev) for r in receptors]),
            persistent=False,
        )
        self._effect_state = torch.zeros(len(receptors), batch, n_regions, device=dev, dtype=DTYPE)

    def _check_semantics(self, semantics: str | None) -> None:
        if semantics is None:
            return
        bad = FORBIDDEN_SEMANTICS.get(self.name, ())
        if semantics.lower().replace(" ", "_") in bad:
            raise SemanticCollapseError(
                f"refusing to label {self.name} as {semantics!r}. Thesis §5: neuromodulators are "
                "receptor-, target-, timescale- and state-dependent control fields that alter gain, "
                "plasticity, exploration, persistence and uncertainty. Declare the receptor, the "
                "target population, the timescale and the state dependence instead."
            )

    @staticmethod
    def _mask(r: ReceptorSpec, n_regions: int, device) -> Tensor:
        if r.density is not None:
            return r.density.to(device=device, dtype=DTYPE).reshape(n_regions)
        m = torch.zeros(n_regions, device=device, dtype=DTYPE)
        for t in r.targets:
            if isinstance(t, int) or (isinstance(t, str) and t.isdigit()):
                m[int(t)] = 1.0
            elif t == "all":
                m.fill_(1.0)
        return m

    @torch.no_grad()
    def update(self, source_drive: Tensor, dt: float, *, activity: Tensor | None = None) -> dict[str, Tensor]:
        """Advance concentration and the per-receptor effects by ``dt``."""
        a = min(dt / self.tau_release, 1.0)
        self.concentration.add_(a * (self.baseline + source_drive - self.concentration))
        gain = torch.ones_like(self.gain)
        plast = torch.ones_like(self.plasticity_rate)
        for i, r in enumerate(self.receptors):
            occ = r.occupancy(self.concentration) * self.target_masks[i].reshape(1, -1)
            if r.state_dependence == "activity" and activity is not None:
                occ = occ * activity.abs().clamp(0, 1)
            ar = min(dt / r.tau, 1.0)
            self._effect_state[i].add_(ar * (occ - self._effect_state[i]))
            gain = gain * (1.0 + r.gain_effect * self._effect_state[i])
            plast = plast * (1.0 + r.plasticity_effect * self._effect_state[i])
        self.gain.copy_(gain.clamp_min(0.0))
        self.plasticity_rate.copy_(plast.clamp_min(0.0))
        return {"gain": self.gain, "plasticity_rate": self.plasticity_rate, "concentration": self.concentration}

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_receptors": len(self.receptors),
            "receptors": [
                {
                    "receptor": r.receptor,
                    "targets": list(r.targets),
                    "gain_effect": r.gain_effect,
                    "plasticity_effect": r.plasticity_effect,
                    "tau_s": r.tau,
                    "state_dependence": r.state_dependence,
                }
                for r in self.receptors
            ],
            "tau_release_s": self.tau_release,
            "note": (
                "receptor-, target-, timescale- and state-dependent control field; "
                "no psychological semantics attached (§5)"
            ),
        }


class NeuromodulatorBank(nn.Module):
    """Several modulatory fields whose gain effects **compose multiplicatively**.

    Composition matters: two modulators acting on the same target through
    different receptors do not add, and the interaction is where most
    intervention-relevant behaviour lives.
    """

    def __init__(self, fields: Sequence[NeuromodulatoryField]):
        super().__init__()
        self.fields = nn.ModuleList(fields)

    def update(self, drives: Mapping[str, Tensor], dt: float, *, activity: Tensor | None = None):
        outs = {}
        for f in self.fields:
            d = drives.get(f.name)
            if d is None:
                d = torch.zeros_like(f.concentration)
            outs[f.name] = f.update(d, dt, activity=activity)
        return outs

    def total_gain(self) -> Tensor:
        g = None
        for f in self.fields:
            g = f.gain.clone() if g is None else g * f.gain
        return g

    def total_plasticity_rate(self) -> Tensor:
        p = None
        for f in self.fields:
            p = f.plasticity_rate.clone() if p is None else p * f.plasticity_rate
        return p
