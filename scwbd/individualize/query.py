"""Honest degradation: a query that needs a parameter the data cannot identify defers.

The failure this exists to prevent is specific and it is not hypothetical: an
MRI-only patient is individualized (their head geometry genuinely is), a
coupling-dependent number is then requested, the runtime happily evaluates the
model -- because the model *has* coupling values, they are just the population's
-- and a clinician reads a patient-specific answer that contains no information
about that patient.  Nothing in that chain errors.  The number is finite,
plausible and wrong in the only way that matters.

So the refusal is placed at the query, not at the fit.  A :class:`Query`
declares which parameter groups its answer depends on.
:func:`answer` evaluates it only if every dependency was individualized; if not
it returns :class:`~scwbd.intervene.safety.Defer` -- the project's existing
"decline and name a cheaper next step" type -- naming the offending groups, the
measured ``lambda_min``, and, when counterfactuals were computed, the modality
that would actually fix it.

:attr:`Query.scope` covers the other direction: a query declared
``"population_level"`` is allowed through carrying that label, because refusing
to answer a population question for a patient with no data would be
obstruction rather than honesty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping

import numpy as np

from scwbd.intervene.safety import Defer
from scwbd.observe.base import Unresolved

from .fit import POPULATION_PRIOR, IndividualizationResult
from .groups import group_by_name

__all__ = [
    "Defer",
    "Query",
    "QueryAnswer",
    "Unresolved",
    "answer",
    "coupling_gain_query",
    "explain",
]


@dataclass(frozen=True)
class Query:
    """A question asked of an individualized model, with its dependencies declared.

    ``depends_on`` names parameter groups.  Declaring it is not optional: a
    query with no declared dependencies cannot be checked, so
    :func:`answer` refuses it rather than assuming it depends on nothing.
    """

    name: str
    depends_on: tuple[str, ...]
    scope: Literal["patient_specific", "population_level"] = "patient_specific"
    description: str = ""
    #: What a clinician would do with the answer.  Printed in the refusal.
    clinical_use: str = ""

    def __post_init__(self) -> None:
        if self.scope == "patient_specific" and not self.depends_on:
            raise ValueError(
                f"query {self.name!r} is patient-specific but declares no "
                "parameter-group dependencies; an undeclared dependency cannot "
                "be checked, and an unchecked dependency is how a population "
                "value gets returned as a patient's."
            )
        for g in self.depends_on:
            group_by_name(g)  # raises on an unknown group


@dataclass(frozen=True)
class QueryAnswer:
    """An answered query, carrying what it was answered from."""

    query: str
    value: Any
    scope: str
    depends_on: tuple[str, ...]
    group_status: Mapping[str, str]
    patient_id: str
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "value": self.value.tolist() if isinstance(self.value, np.ndarray) else self.value,
            "scope": self.scope,
            "depends_on": list(self.depends_on),
            "group_status": dict(self.group_status),
            "patient_id": self.patient_id,
            "notes": list(self.notes),
        }


def _unmet(result: IndividualizationResult, query: Query) -> list[str]:
    bad = []
    for g in query.depends_on:
        out = result.outcomes.get(g)
        if out is None or out.status == POPULATION_PRIOR:
            bad.append(g)
    return bad


def answer(
    result: IndividualizationResult,
    query: Query,
    evaluate: Callable[[np.ndarray], Any],
) -> QueryAnswer | Defer:
    """Evaluate ``query`` against ``result``, or :class:`Defer` with the reason.

    ``evaluate`` receives ``theta_trait`` (the unconstrained coordinate) and
    returns the number.  It is called **only** when every declared dependency was
    individualized -- the guard is upstream of the arithmetic, so a
    non-identifiable dependency cannot produce a number that then has to be
    suppressed.
    """
    status = {
        g: (result.outcomes[g].status if g in result.outcomes else "absent")
        for g in query.depends_on
    }
    if query.scope == "population_level":
        return QueryAnswer(
            query=query.name,
            value=evaluate(result.theta_trait),
            scope=query.scope,
            depends_on=query.depends_on,
            group_status=status,
            patient_id=result.patient_id,
            notes=(
                "POPULATION-LEVEL answer: this is not a statement about this "
                "patient beyond the groups listed as individualized.",
            ),
        )

    bad = _unmet(result, query)
    if not bad:
        return QueryAnswer(
            query=query.name,
            value=evaluate(result.theta_trait),
            scope=query.scope,
            depends_on=query.depends_on,
            group_status=status,
            patient_id=result.patient_id,
            notes=tuple(
                f"{g}: {result.outcomes[g].status} "
                f"(lambda_min = {result.outcomes[g].lambda_min!r})"
                for g in query.depends_on
            ),
        )

    detail: dict[str, float] = {}
    remedy_lines: list[str] = []
    for g in bad:
        gi = result.profile.groups.get(g)
        if gi is not None and gi.lambda_min is not None:
            detail[f"lambda_min[{g}]"] = float(gi.lambda_min)
        for r in result.profile.remedies(g):
            remedy_lines.append(
                f"adding {r['add_modality']} would make {g} identifiable "
                f"(measured lambda_min = {r['lambda_min']:.4g})"
            )
    reason = (
        f"query {query.name!r} depends on parameter group(s) {bad}, which were "
        f"NOT individualized for patient {result.patient_id!r} from "
        f"{list(result.availability.present)}. "
        + " ".join(
            result.outcomes[g].reason for g in bad if g in result.outcomes
        )
        + " Returning the model's value here would return the POPULATION value "
        "as though it were this patient's."
    )
    if remedy_lines:
        reason += " Measured remedy: " + "; ".join(remedy_lines) + "."
    suggested = (
        "additional_calibration_measurement" if remedy_lines else "no_action"
    )
    return Defer(reason=reason, suggested_action=suggested, detail=detail)


def explain(outcome: "Defer | QueryAnswer") -> str:
    if isinstance(outcome, Defer):
        return f"DEFER ({outcome.suggested_action}): {outcome.reason}"
    return f"ANSWER {outcome.query} = {outcome.value!r} [{outcome.scope}]"


# --------------------------------------------------------------------------
# a concrete, clinically shaped query
# --------------------------------------------------------------------------
def coupling_gain_query(name: str = "predicted_downstream_response") -> Query:
    """The query the brief names: a coupling-dependent prediction."""
    return Query(
        name=name,
        depends_on=("coupling", "conduction_delay"),
        scope="patient_specific",
        description=(
            "predicted amplitude and latency of the response in region 3 to a "
            "perturbation of region 1, for THIS patient"
        ),
        clinical_use=(
            "choosing a stimulation target and an inter-pulse interval; both "
            "are functions of the coupling gains and the conduction delay"
        ),
    )
