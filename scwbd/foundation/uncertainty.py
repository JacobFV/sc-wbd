"""``X_i^uncertainty`` as a live state component, and the typed observation boundary.

body.tex §2.1 lists ``X_i^uncertainty`` among the components of a region's state.
We declared it — ``state.default_layout`` and every family in
``foundation.families`` carry an ``uncertainty`` component — and then **read it
nowhere**.  ``grep -rn '"uncertainty"' scwbd/foundation`` returns declarations
only.  Meanwhile ``EEGHead.log_noise`` and ``BOLDHead.log_noise`` are
``nn.Parameter`` vectors broadcast with ``expand_as``: the predictive variance of
both instrument heads is **constant in state, time, horizon, window, participant
and condition**, while the five held-out-calibrated baselines get a variance of
shape ``(horizon, C)``.

**What this does and does not repair.**  Turing's decomposition of the +0.4469
excess over the Gaussian-entropy floor, conditional mean held fixed:
``scale 0.4467`` (100% of the gap to the flat ceiling), ``channel 0.1113``,
``horizon 0.0096``, ``state 0.1896`` per-window scalar / ``0.2587`` per-window
per-channel.  Run 1's FAIL is attributable to **scale** — one scalar asserting
variance 1.31 against a held-out residual variance of 3.97, uniformly
overconfident by 3.0x — which is a training-schedule defect, not an
architectural one.  **This module does not repair run 1.**  It is the only one
of the three structural terms that needs an architectural change at all, it is
~20x the horizon term, and after the schedule fix it is the only place left to
win an NLL claim; the bar is the matched-calibration ceiling L4 = 2.0205 and only
sub-2.0205 counts as new content.  It must earn that on its own.

This module supplies the **state side** of the fix and nothing else.  It does not
touch ``heads.py``:

``UncertaintyPropagator``
    Integrates ``du/dt = innovation(x, c) - decay * u`` over the region's own
    ``uncertainty`` channels, in physical time.  ``innovation`` reads the
    region's state and its arriving coupling, so the *growth rate* is
    state-dependent; integrating it over a rollout is what makes the variance
    grow with the horizon **without the head being told what the horizon is**.
    The time index after assimilation *is* the horizon step; a variance that
    grows because the state says so is falsifiable, whereas one that grows
    because it was handed ``h`` is not.

``ObservationInterface`` / ``FamilyObservationInterface``
    The typed boundary an observation head reads: ``source_features`` (what the
    instrument sees) and ``predictive_logvar`` (how uncertain the model is
    there).  ``predictive_logvar`` reads **only** the ``uncertainty``
    component — so "the predictive variance is sourced from ``X_i^uncertainty``"
    is literally true and not a figure of speech — through a **sign-constrained**
    map, so more accumulated uncertainty always means more predicted variance.
    An unconstrained map would let the channel fit anything and mean nothing.

Both arms of body.tex §11.4 get an interface.  Giving the treatment arm a state
dependent variance and leaving the control arm on a broadcast parameter would
make A1 measure the variance path rather than the structured state, which is not
the ablation A1 exists to run.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .config import ModelConfig
from .families import FamilyStateLayout
from .state import StateLayout

__all__ = [
    "UncertaintyPropagator",
    "ObservationInterface",
    "FlatObservationInterface",
    "FamilyObservationInterface",
    "UNCERTAINTY_COMPONENT",
    "LOGVAR_CLAMP",
]

#: The state component predictive variance is sourced from (body.tex §2.1).
UNCERTAINTY_COMPONENT = "uncertainty"
#: Clamp on the emitted log-variance.  Wide enough not to bind in practice,
#: narrow enough that a diverging uncertainty channel cannot silently buy an
#: arbitrarily good NLL by predicting infinite variance on hard samples.
LOGVAR_CLAMP = (-8.0, 6.0)


class UncertaintyPropagator(nn.Module):
    """``du/dt = softplus(innovation(x, c)) - softplus(log_decay) * u``.

    A leaky accumulator of predictive uncertainty, integrated in **seconds** at
    ``dt``.  Three properties are deliberate:

    * **The innovation is non-negative.**  Uncertainty is generated, never
      destroyed except by the decay term.  A signed innovation could cancel
      accumulated uncertainty to fit a single easy sample.
    * **The innovation reads the state.**  Its rate therefore differs across
      parcels, across conditions and across time — which is the thing a
      broadcast ``nn.Parameter`` cannot do, and the thing the firing test checks.
    * **It has a fixed point** at ``u* = innovation / decay``, so the variance
      saturates rather than diverging over a long rollout.

    At initialisation the innovation has a small positive bias, so ``u`` grows
    with the horizon before any training.  That growth is a *floor*, not the
    claim: what training shapes, and what the test measures, is the growth's
    **dependence on state**.
    """

    def __init__(
        self,
        state_dim: int,
        n_uncertainty: int,
        *,
        in_extra: int = 0,
        hidden: int = 64,
        dt: float = 0.008,
        init_innovation: float = 0.5413,  # softplus(0.5413) ~= 1.0 /s
        init_log_decay: float = -2.0,  # softplus(-2.0) ~= 0.127 /s -> ~8 s time constant
        init_state_gain: float = 0.05,
    ) -> None:
        super().__init__()
        self.n_uncertainty = int(n_uncertainty)
        self.dt = float(dt)
        self.net = nn.Sequential(
            nn.Linear(state_dim + in_extra, hidden), nn.GELU(), nn.Linear(hidden, n_uncertainty)
        )
        # NOT zero-initialised, which is the usual "start as a no-op" default and
        # is wrong here. With a zero output layer the innovation is EXACTLY
        # constant at step 0, so the one property this module exists to provide
        # -- a variance that depends on state -- starts dead and only appears if
        # training happens to find it. Measured on an untrained model: with a
        # zero init the across-parcel spread of the emitted log-variance is
        # 0.0056 (and that only via the assimilated initial condition and the
        # decay term), against 0.25 across time. The state path was a shape, not
        # a mechanism. A small non-zero gain makes it live from step 0 and makes
        # the counterfactual test in tests/foundation/test_uncertainty_state.py
        # measure something.
        nn.init.normal_(self.net[-1].weight, std=float(init_state_gain))
        nn.init.constant_(self.net[-1].bias, float(init_innovation))
        self.log_decay = nn.Parameter(torch.full((n_uncertainty,), float(init_log_decay)))

    def innovation(self, x: Tensor, extra: Tensor | None = None) -> Tensor:
        """Non-negative uncertainty generation rate, ``(..., n_uncertainty)`` per second."""
        z = x if extra is None else torch.cat([x, extra], dim=-1)
        return torch.nn.functional.softplus(self.net(z))

    def forward(self, x: Tensor, u: Tensor, extra: Tensor | None = None) -> Tensor:
        """``du`` for one step of ``dt``.  ``u`` is the current uncertainty channel."""
        a = self.innovation(x, extra)
        lam = torch.nn.functional.softplus(self.log_decay).to(u.dtype)
        return (a - lam * u) * self.dt


class ObservationInterface(nn.Module):
    """What an observation head is allowed to read from the state.

    A head gets two things and no state slices: the features the instrument
    couples to, and the model's own predictive log-variance there.  Everything
    instrument-specific — the lead field, the region-to-channel mapping, the
    per-channel noise floor — stays in ``heads.py``, which is not this file's to
    edit.
    """

    feature_dim: int
    logvar_dim: int

    def source_features(self, x: Tensor) -> Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def predictive_logvar(self, x: Tensor) -> Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {
            "kind": type(self).__name__,
            "feature_dim": self.feature_dim,
            "logvar_dim": self.logvar_dim,
            "logvar_source": UNCERTAINTY_COMPONENT,
            "logvar_clamp": list(LOGVAR_CLAMP),
            "state_dependent": True,
        }


class _LogVarHead(nn.Module):
    """``uncertainty -> log-variance``, monotone by construction.

    The weights are passed through ``softplus`` so they are strictly positive:
    accumulating uncertainty can only ever *raise* the predicted variance.  With
    an unconstrained map the channel could learn to mean anything, including its
    own negation, and "the variance is sourced from `X^uncertainty`" would stop
    being a statement about the model.
    """

    def __init__(self, n_uncertainty: int, out_dim: int = 1) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_uncertainty, out_dim))
        self.b = nn.Parameter(torch.zeros(out_dim))

    def forward(self, u: Tensor) -> Tensor:
        w = torch.nn.functional.softplus(self.w).to(u.dtype)
        return (u @ w + self.b.to(u.dtype)).clamp(*LOGVAR_CLAMP)


class FlatObservationInterface(ObservationInterface):
    """The §11.4 **control** arm's boundary: one state space for every parcel.

    Present so that the ablation compares structured state against a pooled
    vector, and not a state-dependent variance against a broadcast constant.
    """

    def __init__(self, layout: StateLayout, cfg: ModelConfig, *, logvar_dim: int = 1) -> None:
        super().__init__()
        self.layout = layout
        self._exported = list(layout.exported_names())
        self.feature_dim = sum(layout.spec(n).dim for n in self._exported)
        self.logvar_dim = int(logvar_dim)
        if UNCERTAINTY_COMPONENT not in layout:
            raise KeyError(
                f"layout has no {UNCERTAINTY_COMPONENT!r} component; body.tex §2.1 names it and the "
                "predictive variance is sourced from it"
            )
        self.head = _LogVarHead(layout.spec(UNCERTAINTY_COMPONENT).dim, self.logvar_dim)

    def source_features(self, x: Tensor) -> Tensor:
        return torch.cat([self.layout.get(x, n) for n in self._exported], dim=-1)

    def predictive_logvar(self, x: Tensor) -> Tensor:
        return self.head(self.layout.get(x, UNCERTAINTY_COMPONENT))


class FamilyObservationInterface(ObservationInterface):
    """The treatment arm's boundary: each family exposes its own declared out-ports.

    Fixes two things at once.  The **mean** path: with families on, a head handed
    ``SCWBD.layout`` sees only the shared interface prefix (``rate_e``,
    ``rate_i``) and loses everything a family actually exports — a cortical
    family's spectral quadrature, a hippocampal family's ``(v, rho)``, a
    basal-ganglia family's ``gate``.  The **variance** path: each family maps its
    own ``uncertainty`` channels to a log-variance, because how uncertainty
    accumulates in a thalamic relay is not how it accumulates in a cortical
    column.
    """

    def __init__(self, flayout: FamilyStateLayout, cfg: ModelConfig, *, logvar_dim: int = 1) -> None:
        super().__init__()
        self.flayout = flayout
        self.feature_dim = int(cfg.message_dim)
        self.logvar_dim = int(logvar_dim)
        self.feat = nn.ModuleDict()
        self.logvar = nn.ModuleDict()
        self._ports: dict[str, tuple[str, ...]] = {}
        for f in flayout:
            names = tuple(p.name for p in f.out_ports())
            width = sum(f.port_dim(n) for n in names)
            self._ports[f.name] = names
            self.feat[f.name] = nn.Linear(width, self.feature_dim)
            if UNCERTAINTY_COMPONENT not in f.layout:
                raise KeyError(
                    f"family {f.name!r} declares no {UNCERTAINTY_COMPONENT!r} component; every "
                    "family must carry X_i^uncertainty (body.tex §2.1)"
                )
            self.logvar[f.name] = _LogVarHead(f.layout.spec(UNCERTAINTY_COMPONENT).dim, self.logvar_dim)

    def source_features(self, x: Tensor) -> Tensor:
        chunks = []
        for f in self.flayout:
            src = torch.cat([self.flayout.port(x, f.name, p) for p in self._ports[f.name]], dim=-1)
            chunks.append(self.feat[f.name](src))
        return self.flayout.assemble(chunks)

    def predictive_logvar(self, x: Tensor) -> Tensor:
        chunks = []
        for f in self.flayout:
            u = self.flayout.get(x, f.name, UNCERTAINTY_COMPONENT)
            chunks.append(self.logvar[f.name](u))
        return self.flayout.assemble(chunks)

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d["per_family_logvar"] = sorted(self.logvar.keys())
        d["out_ports"] = {k: list(v) for k, v in self._ports.items()}
        return d
