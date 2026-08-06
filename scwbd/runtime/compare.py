"""Offline ranking of a small **preregistered** set of candidate coil poses.

``thesis_contract.tex`` Sec. 0.5 asks a bounded question: *"which of a small
preregistered set of left-DLPFC coil poses is predicted to produce the desired
short-horizon change in a measured network response while remaining inside
independent exposure and protocol limits?"*  This module answers exactly that
question and no larger one.

Three properties are load-bearing:

**Preregistration is enforced, not encouraged.**
    :class:`PreregisteredPoseSet` hashes its candidates at construction.  The
    ranking carries that hash.  Adding a candidate after seeing the results
    produces a different set with a different hash, so a post-hoc winner cannot
    be passed off as a preregistered one.

**The ambiguity structure survives the ranking.**
    Two candidates whose predicted distributions overlap under the posterior
    are *not* tie-broken.  They are grouped and reported as
    :class:`UnresolvedCausalAmbiguity`, together with the measurement that
    would discriminate them.  A ranking that always produces a strict order is
    a ranking that has hidden its own non-identifiability.

**The four validation levels never fuse.**
    Field accuracy, target engagement, network effect and utility are reported
    per candidate as four separate objects.  The ordering uses the
    risk-sensitive *decision* value of thesis Sec. 7.4, which is declared in
    :attr:`CandidateRanking.ordering_basis` and is explicitly not a summary
    score over the four levels.  There is no attribute anywhere in this module
    that combines them.

Claim limits
------------
``evaluate_pose`` and this module rank *hypotheses offline*.  They never emit a
protocol for a person, and a ``Recommend`` here means one simulated candidate
separated from another by more than the simulated uncertainty.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch

from ._compat import Defer, SafetyLimits
from .head import HeadModel
from .targeting import TargetingService
from .types import (
    Decision,
    EngagementDistribution,
    FieldAccuracy,
    NetworkResponse,
    PoseEvaluation,
    PoseRequest,
    Recommend,
    Refuse,
    UtilityStatus,
)

__all__ = [
    "PreregisteredCandidate",
    "PreregisteredPoseSet",
    "RankedCandidate",
    "UnresolvedCausalAmbiguity",
    "CandidateRanking",
    "compare_poses",
]

_DT = torch.float64


# ---------------------------------------------------------------------------
# preregistration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreregisteredCandidate:
    """One candidate, with the hypothesis it was registered to test."""

    label: str
    request: PoseRequest
    hypothesis: str
    registered_at: str = ""

    def __post_init__(self) -> None:
        if not self.hypothesis.strip():
            raise ValueError(
                f"candidate {self.label!r} registers no hypothesis; a pose "
                "entered into a comparison without a stated hypothesis cannot "
                "be a preregistered candidate"
            )

    def identity(self) -> dict[str, Any]:
        pose = self.request.pose
        return {
            "label": self.label,
            "hypothesis": self.hypothesis,
            "registered_at": self.registered_at,
            "frame": None if pose is None else pose.parent,
            "matrix": None if pose is None else [
                [round(float(v), 12) for v in row] for row in pose.matrix.tolist()
            ],
            "description": self.request.description,
        }


@dataclass(frozen=True)
class PreregisteredPoseSet:
    """A frozen, hashed set of candidate poses and the question they answer.

    ``question`` is the bounded question of Sec. 0.5 written out.  It is
    required and non-empty because a comparison whose question was never
    written down cannot have been preregistered.
    """

    study_id: str
    question: str
    candidates: tuple[PreregisteredCandidate, ...]
    comparator: str = "no_stimulation"
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("a preregistered set must state its bounded question")
        if len(self.candidates) < 2:
            raise ValueError(
                "a comparison needs at least two candidates; a set of one is "
                "not a comparison and cannot produce a benefit difference"
            )
        labels = [c.label for c in self.candidates]
        if len(set(labels)) != len(labels):
            raise ValueError(f"duplicate candidate labels in {labels}")

    def preregistration_hash(self) -> str:
        payload = {
            "study_id": self.study_id,
            "question": self.question,
            "comparator": self.comparator,
            "candidates": [c.identity() for c in self.candidates],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(c.label for c in self.candidates)


# ---------------------------------------------------------------------------
# ranking outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RankedCandidate:
    """One candidate's place in the ranking, with its four quantities intact.

    There is no ``score`` attribute and no method that combines the four
    reported quantities.  ``decision_value`` is the risk-sensitive *decision*
    objective of thesis Sec. 7.4 -- benefit minus burden, harm and epistemic
    penalty -- which is what the ordering uses and is a fifth, differently
    named thing.
    """

    label: str
    hypothesis: str
    #: Group index. Candidates sharing a group are observationally
    #: indistinguishable and are *not* ordered within it.
    group: int
    decision_value: float
    epistemic_uncertainty: float

    # the four separate reported quantities
    field_accuracy: FieldAccuracy
    target_engagement: EngagementDistribution
    network_effect: NetworkResponse
    utility: UtilityStatus

    evaluation: PoseEvaluation

    def four_quantities(self) -> dict[str, Any]:
        return {
            "field_accuracy": self.field_accuracy,
            "target_engagement": self.target_engagement,
            "network_effect": self.network_effect,
            "utility": self.utility,
        }

    def __float__(self) -> float:  # pragma: no cover - guard
        raise TypeError(
            "a ranked candidate has no scalar value; field accuracy, target "
            "engagement, network effect and utility are four separate reported "
            "quantities and this type refuses to summarise them"
        )


@dataclass(frozen=True)
class UnresolvedCausalAmbiguity:
    """Candidates the posterior cannot tell apart, reported instead of a tie-break.

    ``separation`` is the difference in decision value between the two extreme
    members; ``combined_epistemic`` is the uncertainty that swallows it.  The
    ``discriminating_measurement`` is the concrete next step -- which is the
    only useful thing to say about a non-identifiability.
    """

    labels: tuple[str, ...]
    reason: str
    separation: float
    combined_epistemic: float
    discriminating_measurement: str
    dominant_term: str  # "transform_uncertainty" | "model_disagreement"

    def __bool__(self) -> bool:  # pragma: no cover - guard against truthiness
        return False


@dataclass(frozen=True)
class CandidateRanking:
    """The output of :func:`compare_poses`.

    ``ordered`` is ordered *between* groups and unordered *within* them.
    ``ambiguities`` names every group with more than one member.  ``decision``
    is the study-level answer: a ``Recommend`` only when the top group has one
    member and beats the next group by more than the combined uncertainty.
    """

    study_id: str
    question: str
    preregistration_hash: str
    ordering_basis: str
    ordered: tuple[RankedCandidate, ...]
    ambiguities: tuple[UnresolvedCausalAmbiguity, ...]
    refused: tuple[tuple[str, Refuse], ...]
    decision: Decision
    limits_citations: tuple[str, ...]
    notice: str

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(c.label for c in self.ordered)

    @property
    def has_strict_winner(self) -> bool:
        return isinstance(self.decision, Recommend)

    def candidate(self, label: str) -> RankedCandidate:
        for c in self.ordered:
            if c.label == label:
                return c
        raise KeyError(f"{label!r} is not in this ranking")

    def four_quantities(self, label: str) -> dict[str, Any]:
        return self.candidate(label).four_quantities()

    def summary(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "question": self.question,
            "preregistration_hash": self.preregistration_hash,
            "ordering_basis": self.ordering_basis,
            "groups": [
                {"group": c.group, "label": c.label, "decision_value": c.decision_value}
                for c in self.ordered
            ],
            "ambiguities": [
                {
                    "labels": list(a.labels),
                    "separation": a.separation,
                    "combined_epistemic": a.combined_epistemic,
                    "dominant_term": a.dominant_term,
                    "discriminating_measurement": a.discriminating_measurement,
                }
                for a in self.ambiguities
            ],
            "refused": [{"label": k, "code": v.code, "reason": v.reason} for k, v in self.refused],
            "decision": type(self.decision).__name__,
            "decision_detail": str(self.decision),
            "notice": self.notice,
        }


# ---------------------------------------------------------------------------
# the comparison
# ---------------------------------------------------------------------------


def compare_poses(
    service: TargetingService,
    head_model: HeadModel,
    pose_set: PreregisteredPoseSet,
    *,
    burden: Mapping[str, float] | None = None,
    harm: Mapping[str, float] | None = None,
) -> CandidateRanking:
    """Rank a preregistered candidate set, preserving the ambiguity structure.

    Steps, in the order of ``thesis_contract.tex`` Sec. 0.5:

    5. **Counterfactually compare.**  Each candidate is passed through the
       field solve and the retained response operators; field accuracy, target
       engagement, network effect and utility stay separate.
    6. **Decide or defer.**  Candidates outside :math:`\\mathcal A_{\\rm safe}`
       are removed from the ordering entirely -- they never receive a decision
       value they could trade against -- and reported under ``refused``.
       Candidates the posterior cannot separate are grouped and reported as
       :class:`UnresolvedCausalAmbiguity`.  A ``Recommend`` is emitted only
       when a single candidate separates from the rest by more than the
       combined epistemic uncertainty.

    Claim limits: offline hypothesis ranking on a simulation.  Never a protocol
    for a person.
    """
    burden = dict(burden or {})
    harm = dict(harm or {})
    limits: SafetyLimits = service.feasible_set.limits
    ratio = limits.defer_ratio()

    evaluations: dict[str, PoseEvaluation] = {}
    refused: list[tuple[str, Refuse]] = []
    admissible: list[PreregisteredCandidate] = []
    for cand in pose_set.candidates:
        ev = service.evaluate_pose(head_model, cand.request, label=cand.label)
        evaluations[cand.label] = ev
        if isinstance(ev.decision, Refuse):
            refused.append((cand.label, ev.decision))
            continue
        admissible.append(cand)

    if not admissible:
        return _empty_ranking(
            pose_set,
            refused,
            limits,
            Refuse(
                code="R11",
                reason="every preregistered candidate lies outside A_safe",
                remedy="revise the preregistered set; do not project into the set",
                offending=pose_set.study_id,
                violations=tuple(v.reason for _, v in refused),
            ),
        )

    values: list[float] = []
    epistemics: list[float] = []
    for cand in admissible:
        ev = evaluations[cand.label]
        benefit, epi = _benefit_and_epistemic(ev)
        # thesis Sec. 7.4: A_safe already filtered; the objective now trades
        # benefit against burden, harm and *reducible* epistemic uncertainty.
        values.append(
            benefit
            - float(burden.get(cand.label, 0.0))
            - float(harm.get(cand.label, 0.0))
            - epi
        )
        epistemics.append(epi)

    groups = _group_indistinguishable(values, epistemics)
    order = sorted(
        range(len(admissible)),
        key=lambda i: (-_group_value(groups, values, groups[i]), groups[i], admissible[i].label),
    )

    ranked: list[RankedCandidate] = []
    remap: dict[int, int] = {}
    for i in order:
        g = groups[i]
        if g not in remap:
            remap[g] = len(remap)
        ev = evaluations[admissible[i].label]
        ranked.append(
            RankedCandidate(
                label=admissible[i].label,
                hypothesis=admissible[i].hypothesis,
                group=remap[g],
                decision_value=values[i],
                epistemic_uncertainty=epistemics[i],
                field_accuracy=ev.field_accuracy,
                target_engagement=ev.target_engagement,
                network_effect=ev.network_response,
                utility=ev.utility,
                evaluation=ev,
            )
        )

    ambiguities = _ambiguities(admissible, values, epistemics, groups, remap, evaluations)
    decision = _study_decision(
        pose_set=pose_set,
        ranked=ranked,
        ambiguities=ambiguities,
        ratio=ratio,
        limits=limits,
        evaluations=evaluations,
    )
    return CandidateRanking(
        study_id=pose_set.study_id,
        question=pose_set.question,
        preregistration_hash=pose_set.preregistration_hash(),
        ordering_basis=(
            "risk-sensitive decision value (body.tex Sec. 7.4): expected "
            "benefit on the declared readout minus burden, harm and reducible "
            "epistemic uncertainty. This is a decision quantity and is NOT a "
            "summary of field accuracy, target engagement, network effect and "
            "utility, which remain four separately reported quantities."
        ),
        ordered=tuple(ranked),
        ambiguities=ambiguities,
        refused=tuple(refused),
        decision=decision,
        limits_citations=limits.citations(),
        notice=service.provenance.notice,
    )


# -- internals --------------------------------------------------------------


def _benefit_and_epistemic(ev: PoseEvaluation) -> tuple[float, float]:
    """Read the benefit and epistemic terms back off an evaluation's decision.

    Both terms were already computed inside ``evaluate_pose``; re-deriving them
    here would risk the ranking and the per-pose decision disagreeing, which is
    exactly the kind of quiet inconsistency this repository exists to prevent.
    """
    d = ev.decision
    if isinstance(d, Recommend):
        return d.benefit_margin, d.epistemic_uncertainty
    if isinstance(d, Defer):
        detail = dict(d.detail)
        return float(detail.get("benefit_margin", 0.0)), float(detail.get("epistemic", 0.0))
    raise TypeError(  # pragma: no cover - refusals are filtered before this
        f"cannot rank a candidate whose decision is {type(d).__name__}"
    )


def _group_indistinguishable(
    values: Sequence[float], epistemics: Sequence[float]
) -> list[int]:
    """Union-find over pairs whose separation is inside their joint uncertainty."""
    n = len(values)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i in range(n):
        for j in range(i + 1, n):
            if abs(values[i] - values[j]) < _combined(epistemics[i], epistemics[j]):
                union(i, j)
    return [find(i) for i in range(n)]


def _combined(a: float, b: float) -> float:
    return float((a * a + b * b) ** 0.5)


def _group_value(groups: Sequence[int], values: Sequence[float], g: int) -> float:
    members = [values[i] for i in range(len(values)) if groups[i] == g]
    return max(members) if members else float("-inf")


def _ambiguities(
    candidates: Sequence[PreregisteredCandidate],
    values: Sequence[float],
    epistemics: Sequence[float],
    groups: Sequence[int],
    remap: Mapping[int, int],
    evaluations: Mapping[str, PoseEvaluation],
) -> tuple[UnresolvedCausalAmbiguity, ...]:
    out: list[UnresolvedCausalAmbiguity] = []
    for g in sorted(set(groups), key=lambda g: remap.get(g, 10**6)):
        members = [i for i in range(len(candidates)) if groups[i] == g]
        if len(members) < 2:
            continue
        labels = tuple(candidates[i].label for i in members)
        sep = max(values[i] for i in members) - min(values[i] for i in members)
        comb = _combined(
            max(epistemics[i] for i in members), min(epistemics[i] for i in members)
        )
        transform, model = 0.0, 0.0
        for i in members:
            d = evaluations[candidates[i].label].decision
            detail = dict(getattr(d, "detail", {}) or {})
            transform = max(transform, float(detail.get("transform_sd", 0.0)))
            model = max(model, float(detail.get("model_disagreement", 0.0)))
        dominant = "transform_uncertainty" if transform >= model else "model_disagreement"
        out.append(
            UnresolvedCausalAmbiguity(
                labels=labels,
                reason=(
                    f"candidates {list(labels)} are separated by {sep:.4g} in "
                    f"decision value, inside their combined epistemic "
                    f"uncertainty {comb:.4g}; under this posterior they are "
                    "observationally indistinguishable and the ranking refuses "
                    "to break the tie"
                ),
                separation=float(sep),
                combined_epistemic=float(comb),
                dominant_term=dominant,
                discriminating_measurement=(
                    "a calibration measurement of the registration chain "
                    "(fiducial re-digitisation, tracker-to-coil re-calibration) "
                    "-- the transform term is the larger one"
                    if dominant == "transform_uncertainty"
                    else "a reversible probe whose predicted responses differ "
                    "between the retained response operators -- model-class "
                    "disagreement is the larger term"
                ),
            )
        )
    return tuple(out)


def _study_decision(
    *,
    pose_set: PreregisteredPoseSet,
    ranked: Sequence[RankedCandidate],
    ambiguities: Sequence[UnresolvedCausalAmbiguity],
    ratio: float,
    limits: SafetyLimits,
    evaluations: Mapping[str, PoseEvaluation],
) -> Decision:
    top = [c for c in ranked if c.group == 0]
    if len(top) > 1:
        amb = next((a for a in ambiguities if set(a.labels) >= {c.label for c in top}), None)
        return Defer(
            reason=(
                f"the leading group {[c.label for c in top]} is an unresolved "
                "causal ambiguity: the candidates are not distinguishable under "
                "this posterior"
                + (f"; {amb.reason}" if amb is None else "")
            ),
            suggested_action=(
                "additional_calibration_measurement"
                if amb is not None and amb.dominant_term == "transform_uncertainty"
                else "reversible_probe"
            ),
            detail={
                "n_tied": float(len(top)),
                "combined_epistemic": float(
                    amb.combined_epistemic if amb is not None else 0.0
                ),
            },
        )

    winner = top[0]
    runner_up = next((c for c in ranked if c.group == 1), None)
    if runner_up is None:
        gap = float("inf")
        combined = winner.epistemic_uncertainty
    else:
        gap = winner.decision_value - runner_up.decision_value
        combined = _combined(
            winner.epistemic_uncertainty, runner_up.epistemic_uncertainty
        )

    if runner_up is not None and combined >= ratio * abs(gap):
        detail = dict(getattr(evaluations[winner.label].decision, "detail", {}) or {})
        transform_dominates = float(detail.get("transform_sd", 0.0)) >= float(
            detail.get("model_disagreement", 0.0)
        )
        return Defer(
            reason=(
                f"epistemic uncertainty {combined:.4g} dominates the benefit "
                f"difference {gap:.4g} between {winner.label!r} and "
                f"{runner_up.label!r} (threshold ratio {ratio:g})"
            ),
            suggested_action=(
                "additional_calibration_measurement"
                if transform_dominates
                else "reversible_probe"
            ),
            detail={"benefit_gap": float(gap), "epistemic": float(combined)},
        )

    if isinstance(evaluations[winner.label].decision, Defer):
        return evaluations[winner.label].decision

    return Recommend(
        label=winner.label,
        rationale=(
            f"{winner.label!r} leads the preregistered set {pose_set.study_id!r} "
            f"by {gap:.4g} in decision value against a combined epistemic "
            f"uncertainty of {combined:.4g}"
        ),
        benefit_margin=float(gap),
        epistemic_uncertainty=float(combined),
        comparator=pose_set.comparator,
        basis=limits.citations(),
    )


def _empty_ranking(
    pose_set: PreregisteredPoseSet,
    refused: Sequence[tuple[str, Refuse]],
    limits: SafetyLimits,
    decision: Decision,
) -> CandidateRanking:
    from ._compat import SIMULATION_ONLY_NOTICE

    return CandidateRanking(
        study_id=pose_set.study_id,
        question=pose_set.question,
        preregistration_hash=pose_set.preregistration_hash(),
        ordering_basis="no admissible candidate; nothing was ordered",
        ordered=(),
        ambiguities=(),
        refused=tuple(refused),
        decision=decision,
        limits_citations=limits.citations(),
        notice=SIMULATION_ONLY_NOTICE,
    )

