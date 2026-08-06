""":math:`\\mathcal A_{\\rm safe}` as a refusal mechanism, not a permission.

**SIMULATION ONLY.**  Nothing in this module authorises stimulating anybody.
:math:`\\mathcal A_{\\rm safe}` here is a *feasible set that blocks a simulated
optimizer*.  Passing every check in this module means only that the optimizer
was not blocked on those axes; it does not mean an intervention is safe,
approved, or applicable to a person.

Design, following thesis Sec. 7.4:

* Limits live in a **declarative, citable file** (``limits/a_safe.toml``) and
  are **never learned**.  :class:`SafetyLimits` refuses mutation, refuses to
  hand parameters to an optimizer, and refuses to load a limit that has no
  citation.
* The feasible set sits **outside the learned objective**.  The optimizer
  never sees :math:`\\mathcal A_{\\rm safe}` as a penalty it can trade away; it
  sees a hard filter that raises :class:`CompilerRefusal` with ``code="R11"``.
* The controller can answer :class:`Defer` -- a safer measurement or a
  reversible probe -- and, per ``thesis_contract.tex`` Sec. 0.5 step 6, *must*
  do so when model disagreement or transform uncertainty dominates the
  estimated benefit difference.
* :class:`NoRecommendation` is a first-class outcome.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

import torch
from torch import Tensor

from .base import SIMULATION_ONLY_NOTICE, InterventionRefusal, Ledger

__all__ = [
    "CompilerRefusal",
    "SafetyLimits",
    "LimitSpec",
    "Violation",
    "SafetyVerdict",
    "ProposedIntervention",
    "FeasibleSet",
    "Defer",
    "NoRecommendation",
    "SimulatedRanking",
    "RiskSensitiveController",
    "DEFAULT_LIMITS_PATH",
]

DEFAULT_LIMITS_PATH = Path(__file__).with_name("limits") / "a_safe.toml"


class CompilerRefusal(InterventionRefusal):
    """Refusal raised when a program leaves :math:`\\mathcal A_{\\rm safe}`.

    Named to match ``thesis_contract.tex`` Table ``tab:compiler-refusals`` and
    ``scwbd.compiler``: ``R11`` is *intervention optimization outside an
    independently validated feasible set*.
    """

    def __init__(
        self,
        code: str = "R11",
        message: str = "intervention optimization outside A_safe",
        *,
        remedy: str = (
            "restrict the search to A_safe, and have the envelope validated "
            "independently of the optimizer"
        ),
        offending_object: Any = None,
        violations: Sequence["Violation"] = (),
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(code, message, remedy=remedy, offending_object=offending_object)
        self.violations = tuple(violations)
        #: Structured detail a consumer can branch on.  The governance gate
        #: puts the specific ``AUTH_*`` failures here so that "expired" and
        #: "wrong modality" are never the same message.
        self.evidence: dict[str, Any] = dict(evidence or {})


# ---------------------------------------------------------------------------
# declarative limits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LimitSpec:
    """One declared bound on one named quantity. Immutable and citable."""

    modality: str
    quantity: str
    minimum: float | None
    maximum: float | None
    units: str
    basis: str
    citation: str

    def check(self, value: float) -> "Violation | None":
        if self.minimum is not None and value < self.minimum:
            return Violation(self, float(value), "below_minimum")
        if self.maximum is not None and value > self.maximum:
            return Violation(self, float(value), "above_maximum")
        return None

    @property
    def key(self) -> str:
        return f"{self.modality}.{self.quantity}"


@dataclass(frozen=True)
class Violation:
    limit: LimitSpec
    value: float
    kind: Literal[
        "below_minimum",
        "above_maximum",
        "missing",
        "unknown_axis",
        "undeclared_by_proposal",
    ]

    def __str__(self) -> str:
        if self.kind == "undeclared_by_proposal":
            return (
                f"{self.limit.key} is a declared limit that this proposal "
                f"supplies no value for, so it was never checked "
                f"({self.limit.citation})"
            )
        bound = self.limit.minimum if self.kind == "below_minimum" else self.limit.maximum
        return (
            f"{self.limit.key} = {self.value:g} {self.limit.units} violates "
            f"{self.kind} {bound} ({self.limit.citation})"
        )


class SafetyLimits:
    """Loaded, immutable, never-learned limits.

    Refuses in three ways:

    * ``__setattr__`` after construction -> ``CompilerRefusal`` (limits are not
      state an optimizer may write).
    * a limit entry without a ``citation`` -> refused at load time.
    * :meth:`parameters` -> refuses; there is nothing here for an optimizer or
      an autograd graph to touch.
    """

    _frozen = False

    def __init__(self, limits: Mapping[str, LimitSpec], meta: Mapping[str, Any]):
        object.__setattr__(self, "_limits", dict(limits))
        object.__setattr__(self, "_meta", dict(meta))
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name: str, value: Any) -> None:  # pragma: no cover - guard
        raise CompilerRefusal(
            "R11",
            f"safety limits are declarative and never learned; refused write to {name!r}",
            remedy="edit limits/a_safe.toml with a citation, and re-review",
            offending_object=name,
        )

    def parameters(self) -> Iterable[Tensor]:  # pragma: no cover - guard
        raise CompilerRefusal(
            "R11",
            "SafetyLimits exposes no learnable parameters; A_safe sits outside "
            "the learned objective (thesis Sec. 7.4)",
            remedy="do not place A_safe inside the optimization graph",
            offending_object="SafetyLimits.parameters",
        )

    # -- loading ------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path | None = None) -> "SafetyLimits":
        p = Path(path) if path is not None else DEFAULT_LIMITS_PATH
        with open(p, "rb") as fh:
            raw = tomllib.load(fh)

        meta = {
            k: v for k, v in raw.items() if not isinstance(v, dict)
        }
        meta["source_path"] = str(p)
        limits: dict[str, LimitSpec] = {}
        for modality, block in raw.items():
            if not isinstance(block, dict):
                continue
            if modality == "decision":
                # The one namespace that is explicitly rules-for-the-controller
                # rather than bounds-on-an-exposure.  Read into meta below.
                continue
            for quantity, entry in block.items():
                if not isinstance(entry, dict):
                    continue
                if "min" not in entry and "max" not in entry:
                    # Previously this was `continue`, and it is how
                    # `protocol.reversibility` sat in this file for the whole
                    # project without ever becoming a LimitSpec: declared,
                    # cited, never loaded, never checkable, never able to fail.
                    # A bound nothing can check is decoration
                    # (reports/decorative_guards.md).  Refuse it at load so the
                    # failure is at startup rather than invisible forever.
                    raise CompilerRefusal(
                        "R11",
                        f"limit {modality}.{quantity} declares neither `min` nor "
                        "`max`, so nothing can ever check it; a bound that "
                        "cannot fire is not a bound",
                        remedy=(
                            "give it a numeric `min`/`max`, or move it under "
                            "[decision] where controller rules live"
                        ),
                        offending_object=f"{modality}.{quantity}",
                    )
                if not entry.get("citation"):
                    raise CompilerRefusal(
                        "R11",
                        f"limit {modality}.{quantity} has no citation; an "
                        "uncited limit is not an independently validated limit",
                        remedy="add `citation` and `basis` to the limits file",
                        offending_object=f"{modality}.{quantity}",
                    )
                spec = LimitSpec(
                    modality=modality,
                    quantity=quantity,
                    minimum=entry.get("min"),
                    maximum=entry.get("max"),
                    units=entry.get("units", "dimensionless"),
                    basis=entry.get("basis", ""),
                    citation=entry["citation"],
                )
                limits[spec.key] = spec
        # non-numeric declarative rules kept in meta for the controller
        meta["decision"] = raw.get("decision", {})
        return cls(limits, meta)

    # -- access -------------------------------------------------------------

    @property
    def meta(self) -> Mapping[str, Any]:
        return dict(self._meta)  # type: ignore[attr-defined]

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._limits))  # type: ignore[attr-defined]

    def get(self, key: str) -> LimitSpec:
        try:
            return self._limits[key]  # type: ignore[attr-defined]
        except KeyError as exc:
            raise CompilerRefusal(
                "R11",
                f"no declared limit for axis {key!r}; an undeclared exposure "
                "axis cannot be certified as inside A_safe",
                remedy="declare the axis in limits/a_safe.toml with a citation",
                offending_object=key,
            ) from exc

    def all_specs(self) -> tuple[LimitSpec, ...]:
        """Every loaded bound.

        Exists so a test can sweep the file rather than a hardcoded list: a
        bound added tomorrow is covered tomorrow, and one that cannot be made
        to fire fails the suite instead of passing it quietly.
        """
        return tuple(v for _, v in sorted(self._limits.items()))  # type: ignore[attr-defined]

    def for_modality(self, modality: str) -> tuple[LimitSpec, ...]:
        return tuple(
            v for k, v in sorted(self._limits.items())  # type: ignore[attr-defined]
            if v.modality == modality
        )

    def citations(self) -> tuple[str, ...]:
        return tuple(sorted({v.citation for v in self._limits.values()}))  # type: ignore[attr-defined]

    def defer_ratio(self) -> float:
        return float(
            self._meta.get("decision", {})  # type: ignore[attr-defined]
            .get("defer", {})
            .get("disagreement_dominance_ratio", 1.0)
        )

    def min_candidate_models(self) -> int:
        return int(
            self._meta.get("decision", {})  # type: ignore[attr-defined]
            .get("epistemic", {})
            .get("min_candidate_models", 2)
        )


# ---------------------------------------------------------------------------
# proposals and the feasible set
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProposedIntervention:
    """A candidate :math:`u` from a simulated optimizer.

    ``exposure`` maps declared axis names (``"tms.peak_efield_v_per_m"``, ...)
    to simulated values.  ``pose_certified`` records whether the pose passed
    :mod:`scwbd.intervene.tms.pose` (a scalp label alone is not a pose).

    """

    label: str
    modality: str
    exposure: Mapping[str, float]
    pose_certified: bool = False
    reversible: bool = False
    provenance: Mapping[str, Any] = field(default_factory=dict)
    notice: str = SIMULATION_ONLY_NOTICE


@dataclass(frozen=True)
class SafetyVerdict:
    feasible: bool
    violations: tuple[Violation, ...]
    checked_axes: tuple[str, ...]
    unchecked_declared_axes: tuple[str, ...]
    notice: str = SIMULATION_ONLY_NOTICE

    def raise_if_infeasible(self, offending: Any = None) -> None:
        if not self.feasible:
            raise CompilerRefusal(
                "R11",
                "proposal leaves A_safe: " + "; ".join(str(v) for v in self.violations),
                offending_object=offending,
                violations=self.violations,
            )


class FeasibleSet:
    """:math:`\\mathcal A_{\\rm safe}`. A filter that refuses, never a scorer.

    Deliberately has **no** ``score``/``penalty`` method: turning the feasible
    set into a soft penalty is exactly the failure mode R11 names.
    """

    def __init__(
        self,
        limits: SafetyLimits | None = None,
        *,
        require_pose_certification: bool = True,
        require_complete_coverage: bool = False,
    ) -> None:
        self.limits = limits or SafetyLimits.load()
        self.require_pose_certification = require_pose_certification
        #: When set, an axis declared for this modality that the proposal does
        #: not supply is a violation rather than a note.  Off by default
        #: because most axes have no producer for most proposals; on, it is how
        #: a caller says "check the whole envelope, not the part I supplied".
        self.require_complete_coverage = require_complete_coverage

    def contains(self, proposal: ProposedIntervention) -> SafetyVerdict:
        violations: list[Violation] = []
        declared = {s.key for s in self.limits.for_modality(proposal.modality)}
        declared |= {s.key for s in self.limits.for_modality("protocol")}
        checked: list[str] = []

        for axis, value in proposal.exposure.items():
            spec = self.limits.get(axis)  # refuses on undeclared axis
            checked.append(axis)
            v = spec.check(float(value))
            if v is not None:
                violations.append(v)

        unchecked = tuple(sorted(declared - set(checked)))

        # A declared axis the proposal simply omits is reported, not violated
        # -- for a simulated study that is right, because most axes have no
        # producer for most proposals.  For a *live* plan it is exactly wrong:
        # a plan could pass by supplying the three axes it is comfortable with
        # and omitting the thermal ones.  `tfus.cem43_minutes` and
        # `tfus.temperature_rise_c` have no producer anywhere in `scwbd`, so
        # under the old rule a live tFUS plan was silently unchecked on
        # thermal dose. Requiring complete coverage turns that silence into a
        # refusal that names the missing axes.
        if self.require_complete_coverage and unchecked:
            for axis in unchecked:
                spec = self.limits.get(axis)
                violations.append(Violation(spec, float("nan"), "undeclared_by_proposal"))

        if self.require_pose_certification and not proposal.pose_certified:
            violations.append(
                Violation(
                    LimitSpec(
                        modality=proposal.modality,
                        quantity="pose_certified",
                        minimum=1.0,
                        maximum=None,
                        units="dimensionless",
                        basis=(
                            "a target is not a scalp label; the full device pose "
                            "and its transform chain must be certified"
                        ),
                        citation="body.tex Sec. 2.8 / 7.2; thesis_contract Sec. 0.5 step 2",
                    ),
                    0.0,
                    "below_minimum",
                )
            )

        return SafetyVerdict(
            feasible=not violations,
            violations=tuple(violations),
            checked_axes=tuple(checked),
            unchecked_declared_axes=unchecked,
        )

    def guard(self, proposal: ProposedIntervention) -> ProposedIntervention:
        """Pass a proposal through, or raise ``CompilerRefusal(code='R11')``."""
        self.contains(proposal).raise_if_infeasible(offending=proposal.label)
        return proposal

    def guard_optimizer(
        self,
        propose: Callable[[], ProposedIntervention],
        *,
        max_attempts: int = 1,
    ) -> ProposedIntervention:
        """Wrap a simulated optimizer so a proposal outside A_safe is blocked.

        There is no repair loop by design: silently projecting an infeasible
        proposal back into the set would hide that the objective wanted to
        leave it.  ``max_attempts > 1`` only re-queries the optimizer and the
        final failure still refuses.
        """
        last: SafetyVerdict | None = None
        for _ in range(max(1, max_attempts)):
            proposal = propose()
            last = self.contains(proposal)
            if last.feasible:
                return proposal
        assert last is not None
        last.raise_if_infeasible(offending="optimizer proposal")
        raise AssertionError("unreachable")


# ---------------------------------------------------------------------------
# controller outcomes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Defer:
    """The controller declines to act and names a cheaper, safer next step."""

    reason: str
    suggested_action: Literal[
        "additional_calibration_measurement", "reversible_probe", "no_action"
    ]
    detail: Mapping[str, float] = field(default_factory=dict)
    notice: str = SIMULATION_ONLY_NOTICE

    def __str__(self) -> str:
        return f"Defer({self.suggested_action}): {self.reason}"


@dataclass(frozen=True)
class NoRecommendation:
    """No candidate is distinguishable; the honest output is nothing."""

    reason: str
    detail: Mapping[str, float] = field(default_factory=dict)
    notice: str = SIMULATION_ONLY_NOTICE


@dataclass(frozen=True)
class SimulatedRanking:
    """A ranking over *simulated* candidates. Never a protocol for a person."""

    ordered_labels: tuple[str, ...]
    objective_values: Tensor
    benefit_gap: float
    epistemic_uncertainty: float
    limits_citations: tuple[str, ...]
    ledger: Ledger = field(default_factory=Ledger)
    notice: str = SIMULATION_ONLY_NOTICE


Decision = SimulatedRanking | Defer | NoRecommendation


# ---------------------------------------------------------------------------
# risk-sensitive controller (thesis Sec. 7.4)
# ---------------------------------------------------------------------------


class RiskSensitiveController:
    """Selects across the posterior, inside :math:`\\mathcal A_{\\rm safe}` only.

    Implements

    .. math::

        u^* \\in \\arg\\max_{u\\in\\mathcal A_{\\rm safe}}
          \\mathbb E\\left[B(X_T) - \\lambda C(u) - \\beta L_{\\rm harm}\\right]
          - \\gamma\\,\\mathcal U_{\\rm epi}(u)

    with three properties the thesis requires:

    1. :math:`\\mathcal A_{\\rm safe}` is a **filter applied before** the
       objective is evaluated, not a term inside it.
    2. :math:`L_{\\rm harm}` is averaged **once** under the posterior and
       :math:`\\mathcal U_{\\rm epi}` penalises only *reducible* model
       disagreement -- aleatoric outcome risk is not double counted.
    3. If disagreement or transform uncertainty dominates the benefit
       difference, the output is :class:`Defer` or :class:`NoRecommendation`
       (``thesis_contract.tex`` Sec. 0.5 step 6).
    """

    def __init__(
        self,
        feasible_set: FeasibleSet | None = None,
        *,
        lam: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
    ) -> None:
        self.feasible_set = feasible_set or FeasibleSet()
        self.lam = lam
        self.beta = beta
        self.gamma = gamma

    def decide(
        self,
        candidates: Sequence[ProposedIntervention],
        *,
        benefit: Tensor,  # [n_models, n_candidates] posterior samples of B(X_T)
        burden: Tensor,  # [n_candidates]
        harm: Tensor,  # [n_models, n_candidates]
        model_log_weights: Tensor,  # [n_models]
        transform_uncertainty: Tensor | None = None,  # [n_candidates]
        reversible_probe_available: bool = True,
    ) -> Decision:
        if len(candidates) == 0:
            return NoRecommendation(reason="no candidates supplied")

        n_models = int(benefit.shape[0])
        if n_models < self.feasible_set.limits.min_candidate_models():
            return Defer(
                reason=(
                    f"only {n_models} response model(s) considered; the "
                    "mechanism is unresolved and comparison under a single "
                    "model is not admissible"
                ),
                suggested_action="additional_calibration_measurement",
                detail={"n_models": float(n_models)},
            )

        # 1. A_safe filter -- BEFORE the objective. Infeasible proposals do not
        #    get a score they could trade against.
        feasible: list[int] = []
        for i, c in enumerate(candidates):
            if self.feasible_set.contains(c).feasible:
                feasible.append(i)
        if not feasible:
            return NoRecommendation(
                reason="every candidate lies outside A_safe",
                detail={"n_candidates": float(len(candidates))},
            )

        idx = torch.tensor(feasible, dtype=torch.long)
        b = benefit.to(torch.float64)[:, idx]
        h = harm.to(torch.float64)[:, idx]
        c_burden = burden.to(torch.float64)[idx]
        w = torch.softmax(model_log_weights.to(torch.float64), dim=0).reshape(-1, 1)

        # 2. posterior expectation, harm averaged exactly once
        exp_b = (w * b).sum(0)
        exp_h = (w * h).sum(0)

        # reducible model disagreement only (across-model spread of the benefit)
        mean_b = exp_b.unsqueeze(0)
        u_epi = ((w * (b - mean_b) ** 2).sum(0)).sqrt()
        if transform_uncertainty is not None:
            u_epi = u_epi + transform_uncertainty.to(torch.float64)[idx]

        objective = exp_b - self.lam * c_burden - self.beta * exp_h - self.gamma * u_epi

        order = torch.argsort(objective, descending=True)
        labels = tuple(candidates[feasible[int(j)]].label for j in order)
        sorted_obj = objective[order]

        # 3. Sec. 0.5 step 6: disagreement or transform uncertainty dominating
        #    the benefit difference -> defer.
        if sorted_obj.numel() == 1:
            gap = float("inf")
        else:
            gap = float(sorted_obj[0] - sorted_obj[1])
        top_u = float(u_epi[order[0]])
        ratio = self.feasible_set.limits.defer_ratio()

        if sorted_obj.numel() > 1 and top_u >= ratio * abs(gap):
            reason = (
                f"epistemic uncertainty {top_u:.4g} dominates the benefit "
                f"difference {gap:.4g} between the top two admissible "
                f"candidates (threshold ratio {ratio:g})"
            )
            if reversible_probe_available:
                return Defer(
                    reason=reason,
                    suggested_action="reversible_probe",
                    detail={"benefit_gap": gap, "epistemic": top_u},
                )
            return Defer(
                reason=reason,
                suggested_action="additional_calibration_measurement",
                detail={"benefit_gap": gap, "epistemic": top_u},
            )

        if not math.isfinite(gap) or gap <= 0.0:
            return NoRecommendation(
                reason="admissible candidates are not distinguishable",
                detail={"benefit_gap": gap},
            )

        return SimulatedRanking(
            ordered_labels=labels,
            objective_values=sorted_obj,
            benefit_gap=gap,
            epistemic_uncertainty=top_u,
            limits_citations=self.feasible_set.limits.citations(),
            ledger=Ledger(
                variance={"model": float(u_epi.mean() ** 2)},
                bias_status="prior_specified_sensitivity",
                validity_domain={"scope": "simulation_only"},
            ),
        )
