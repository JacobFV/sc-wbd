"""The learned residual ``R_theta`` and the R05 mechanism-dominance guard.

Refusal R05 (thesis_contract Table 1): *a learned residual allowed to dominate a
mechanistic term silently* makes the named mechanism unfalsifiable.  The remedy
is prescribed: enforce a preregistered energy/gain ratio

    rho = ||R_theta|| / ||F_mech||  <=  rho_max

on the **declared validity set**, report violations, or reclassify the entire
module as a surrogate.

This module implements that as a *runtime check that actually fires*:

* :class:`MechanismDominanceGuard` accumulates the ratio over the declared
  validity set and acts according to ``on_violation``:
  ``"raise"`` (fail closed), ``"clamp"`` (project the residual back onto the
  admissible ball and record it), or ``"reclassify"`` (relabel the module's
  ``mechanistic_status`` to ``"surrogate"`` and keep going honestly).
* The default is ``"reclassify"``, because silently clamping is itself a way of
  hiding that the mechanism did not carry the dynamics; the guard's report is
  the scientific output.
* The validity set is explicit.  A ratio measured off the declared validity set
  is recorded under ``ood`` and never used to certify the mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

import torch
from torch import Tensor, nn

from .types import DTYPE, GuardViolation, MechanismRefusal, ParamPack

__all__ = [
    "ResidualOperator",
    "MechanismDominanceGuard",
    "DominanceReport",
    "ValiditySet",
]


# ---------------------------------------------------------------------------
# Validity set
# ---------------------------------------------------------------------------


@dataclass
class ValiditySet:
    """The declared domain on which the mechanism claim is made.

    Boxes on named parameters and on state norms.  ``contains(x, theta)``
    returns a per-sample boolean ``(B,)``; samples outside are excluded from the
    certification statistic and counted separately.  An empty box means "all
    samples are in the validity set", which is a legitimate declaration but is
    recorded as such in the report.
    """

    param_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    state_norm_max: float | None = None
    name: str = "declared_validity_set"

    def contains(self, x: Tensor, theta: ParamPack | None) -> Tensor:
        B = x.shape[0]
        ok = torch.ones(B, dtype=torch.bool, device=x.device)
        if theta is not None:
            for k, (lo, hi) in self.param_bounds.items():
                if k not in theta:
                    continue
                v = theta.get(k).reshape(theta.batch, -1)
                inside = ((v >= lo) & (v <= hi)).all(dim=-1)
                ok = ok & (inside if inside.shape[0] == B else inside.expand(B))
        if self.state_norm_max is not None:
            ok = ok & (x.reshape(B, -1).norm(dim=-1) <= self.state_norm_max)
        return ok

    def is_empty(self) -> bool:
        return not self.param_bounds and self.state_norm_max is None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class DominanceReport:
    """Machine-readable R05 evidence.  Emitted whether or not the guard fired."""

    rho_max: float
    n_observations: int = 0
    n_in_validity: int = 0
    n_violations: int = 0
    ratio_sum: float = 0.0
    ratio_max: float = 0.0
    ratio_quantiles: dict[str, float] = field(default_factory=dict)
    ood_ratio_max: float = 0.0
    status: str = "mechanistic"
    violations: list[GuardViolation] = field(default_factory=list)
    validity_set: str = ""

    @property
    def ratio_mean(self) -> float:
        return self.ratio_sum / max(self.n_in_validity, 1)

    @property
    def violated(self) -> bool:
        return self.n_violations > 0

    @property
    def violation_rate(self) -> float:
        return self.n_violations / max(self.n_in_validity, 1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": "R05",
            "rho_max": self.rho_max,
            "n_observations": self.n_observations,
            "n_in_validity": self.n_in_validity,
            "n_violations": self.n_violations,
            "violation_rate": self.violation_rate,
            "ratio_mean": self.ratio_mean,
            "ratio_max": self.ratio_max,
            "ratio_quantiles": dict(self.ratio_quantiles),
            "ood_ratio_max": self.ood_ratio_max,
            "status": self.status,
            "validity_set": self.validity_set,
            "violations": [v.__dict__ for v in self.violations[:16]],
        }


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


class MechanismDominanceGuard(nn.Module):
    """Runtime enforcement of ``||R_theta|| / ||F_mech|| <= rho_max`` (R05).

    Norms are per-sample RMS over regions and state dimensions by default; a
    weighted norm ``W`` (e.g. per-state-variable scaling from the uncertainty
    ledger) may be supplied so that the ratio is dimensionally sensible for
    heterogeneous state.
    """

    def __init__(
        self,
        rho_max: float = 0.25,
        *,
        validity: ValiditySet | None = None,
        on_violation: Literal["raise", "clamp", "reclassify", "report"] = "reclassify",
        weight: Tensor | None = None,
        eps: float = 1e-12,
        owner: str = "hybrid_field",
        violation_rate_tolerance: float = 0.0,
    ):
        super().__init__()
        if rho_max <= 0:
            raise ValueError("rho_max must be positive and preregistered")
        self.rho_max = float(rho_max)
        self.validity = validity or ValiditySet()
        self.on_violation = on_violation
        self.eps = eps
        self.owner = owner
        self.violation_rate_tolerance = float(violation_rate_tolerance)
        if weight is not None:
            self.register_buffer("weight", weight, persistent=False)
        else:
            self.weight = None
        self.report = DominanceReport(rho_max=self.rho_max, validity_set=self.validity.name)
        self._ratios: list[Tensor] = []
        self._last_ratio: Tensor | None = None

    # -- norms -------------------------------------------------------------
    def _norm(self, v: Tensor) -> Tensor:
        if self.weight is not None:
            v = v * self.weight
        B = v.shape[0]
        return v.reshape(B, -1).pow(2).mean(dim=-1).sqrt()

    def ratio(self, f_mech: Tensor, r: Tensor) -> Tensor:
        return self._norm(r) / (self._norm(f_mech) + self.eps)

    # -- observation -------------------------------------------------------
    @torch.no_grad()
    def observe(self, f_mech: Tensor, r: Tensor, theta: ParamPack | None = None) -> Tensor:
        """Record one batch of ratios.  Raises immediately in ``raise`` mode."""
        rho = self.ratio(f_mech.detach(), r.detach())
        self._last_ratio = rho
        inside = self.validity.contains(f_mech, theta)
        rep = self.report
        rep.n_observations += int(rho.numel())
        in_rho = rho[inside]
        out_rho = rho[~inside]
        if out_rho.numel():
            rep.ood_ratio_max = max(rep.ood_ratio_max, float(out_rho.max()))
        if in_rho.numel():
            rep.n_in_validity += int(in_rho.numel())
            rep.ratio_sum += float(in_rho.sum())
            rep.ratio_max = max(rep.ratio_max, float(in_rho.max()))
            n_bad = int((in_rho > self.rho_max).sum())
            rep.n_violations += n_bad
            self._ratios.append(in_rho.flatten().cpu())
            if n_bad:
                v = GuardViolation(
                    code="R05",
                    detail="learned residual dominates the mechanistic term",
                    value=float(in_rho.max()),
                    tolerance=self.rho_max,
                    offending_object=self.owner,
                    remedy=(
                        "reduce residual capacity/regularisation, extend the mechanistic term, "
                        "or reclassify the module as a surrogate"
                    ),
                )
                if len(rep.violations) < 64:
                    rep.violations.append(v)
                if self.on_violation == "raise" and rep.violation_rate > self.violation_rate_tolerance:
                    rep.status = "refused"
                    raise MechanismRefusal(v)
                if self.on_violation == "reclassify":
                    rep.status = "surrogate"
        return rho

    def apply(self, f_mech: Tensor, r: Tensor) -> Tensor:
        """Optionally project the residual back into the admissible ball.

        Only active in ``clamp`` mode.  Clamping *changes the dynamics*, so it is
        always accompanied by the violation record above — a clamp that leaves no
        trace is exactly the silent domination R05 forbids.
        """
        if self.on_violation != "clamp":
            return r
        rho = self.ratio(f_mech, r)
        scale = torch.clamp(self.rho_max / (rho + self.eps), max=1.0)
        return r * scale.reshape(-1, *([1] * (r.ndim - 1)))

    # -- certification -----------------------------------------------------
    def finalize(self) -> DominanceReport:
        """Compute quantiles and the final mechanistic status."""
        rep = self.report
        if self._ratios:
            allr = torch.cat(self._ratios)
            qs = torch.tensor([0.5, 0.9, 0.95, 0.99])
            vals = torch.quantile(allr.double(), qs.double())
            rep.ratio_quantiles = {f"q{int(q*100)}": float(v) for q, v in zip(qs.tolist(), vals.tolist())}
        if rep.violated and rep.violation_rate > self.violation_rate_tolerance:
            rep.status = "refused" if self.on_violation == "raise" else "surrogate"
        elif not rep.violated:
            rep.status = "mechanistic"
        return rep

    def reset(self) -> None:
        self.report = DominanceReport(rho_max=self.rho_max, validity_set=self.validity.name)
        self._ratios = []
        self._last_ratio = None

    def mechanistic_status(self) -> str:
        return self.finalize().status

    def assert_mechanistic(self) -> None:
        """Fail closed at the end of a run if the mechanism did not dominate."""
        rep = self.finalize()
        if rep.status != "mechanistic":
            raise MechanismRefusal(
                GuardViolation(
                    code="R05",
                    detail=(
                        f"module reclassified as {rep.status}: residual/mechanism ratio exceeded "
                        f"rho_max on {rep.violation_rate:.1%} of the declared validity set"
                    ),
                    value=rep.ratio_max,
                    tolerance=self.rho_max,
                    offending_object=self.owner,
                    remedy="report the reclassification; do not label this operator mechanistic",
                )
            )


# ---------------------------------------------------------------------------
# The residual operator
# ---------------------------------------------------------------------------


class ResidualOperator(nn.Module):
    """``R_theta(X, C)`` — a *local*, magnitude-regularised learned correction.

    Regularised for magnitude, locality, and out-of-distribution behaviour
    (thesis §4.1):

    * **magnitude** — the output passes through ``scale * tanh(.)`` so the
      residual is bounded a priori, and ``scale`` is the knob the R05 guard's
      ``rho_max`` constrains;
    * **locality** — the residual sees only the region's own state, its coupling
      input and the declared context; it has no access to the global state, so
      it cannot silently reimplement long-range coupling;
    * **OOD uncertainty** — :meth:`uncertainty` returns a per-region log-variance
      head used to widen prediction intervals where the residual is
      extrapolating.
    """

    def __init__(
        self,
        state_dim: int,
        n_coupling_channels: int = 1,
        *,
        context_dim: int = 0,
        cond_names: Sequence[str] = (),
        width: int = 32,
        scale: float = 0.1,
        predict_uncertainty: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.state_dim = int(state_dim)
        self.cond_names = tuple(cond_names)
        d_in = state_dim + n_coupling_channels + context_dim + len(self.cond_names)
        self.net = nn.Sequential(
            nn.Linear(d_in, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh()
        )
        self.head = nn.Linear(width, state_dim)
        self.logvar = nn.Linear(width, state_dim) if predict_uncertainty else None
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.weight.mul_(0.1)
        self.register_buffer("scale", torch.tensor(float(scale)), persistent=False)
        self._last_h: Tensor | None = None

    def features(self, x: Tensor, coupling: Tensor, theta: ParamPack | None, context: Tensor | None) -> Tensor:
        parts = [x, coupling]
        if context is not None:
            parts.append(context.expand(x.shape[0], x.shape[1], -1))
        if self.cond_names and theta is not None:
            parts += [theta.get(n).expand(x.shape[0], x.shape[1], 1) for n in self.cond_names]
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        x: Tensor,
        coupling: Tensor,
        theta: ParamPack | None = None,
        context: Tensor | None = None,
    ) -> Tensor:
        h = self.net(self.features(x, coupling, theta, context))
        self._last_h = h
        return self.scale * torch.tanh(self.head(h))

    def uncertainty(self, x: Tensor, coupling: Tensor, theta=None, context=None) -> Tensor:
        if self.logvar is None:
            raise RuntimeError("this ResidualOperator was built without an uncertainty head")
        h = self.net(self.features(x, coupling, theta, context))
        return self.logvar(h)

    def locality_penalty(self) -> Tensor:
        """L2 on the first-layer weights reading the coupling channels.

        Discourages the residual from re-deriving long-range effects that the
        connectome operator is supposed to own.
        """
        w = self.net[0].weight
        return w[:, self.state_dim :].pow(2).sum()

    def magnitude_penalty(self, x: Tensor, coupling: Tensor, theta=None, context=None) -> Tensor:
        return self.forward(x, coupling, theta, context).pow(2).mean()
