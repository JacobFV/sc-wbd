"""The live-application gate: what leaves this repository toward a person.

Why this module exists, and what it is *not*
--------------------------------------------
Everything inside SC-WBD is computational: simulated fields, simulated tissue,
and previously-recorded open data.  That work is not gated here and this
module has no opinion about it.  What is gated is the **export edge** -- a plan
intended to drive real hardware, or to be applied to a real person.

This module exists because that distinction was previously carried by prose
and not by code.  ``scwbd.intervene`` asserted in several places that
prospective human TMS/tFUS was "out of scope" because "there is no ethics
approval, no consent, no participant, and no device".  Every one of those was
a claim about the world rather than a property of the software, none of them
was checked anywhere, and ``scwbd.schema.authorization`` had already replaced
the underlying refusal with a real gate that admits a complete, in-date,
in-scope declaration.  A string asserting a restriction the code does not
enforce is the same defect as a string asserting a permission the code does
not enforce; the second is merely more dangerous.

The rule this module enforces
-----------------------------
A live application is refused **unless a record exists that the preliminary
review actually occurred and reached an approving outcome.**

:data:`PRELIMINARY_REVIEW_SCHEDULED` is the date that review is scheduled for.
It binds as a **lower bound on the review record**, never as an unlock:

* a plan with no review record refuses, whatever today's date is -- including
  after the scheduled date.  A date passing is not evidence of an outcome;
* a record claiming the review happened *before* the scheduled date refuses as
  impossible;
* a record claiming the review happened *after* now refuses as not-yet-occurred;
* a record whose outcome is anything other than approving refuses, and says
  which outcome it read.

There is deliberately no code path that consults the calendar and opens.  The
test ``test_deployment.py::TestTheDateIsNotAnUnlock`` sets the clock years past
the scheduled date with no review record and asserts the refusal stands; that
test is the reason the date in this file cannot go quietly stale.

An :class:`~scwbd.schema.authorization.AuthorizationRecord` is **necessary and
not sufficient** for a live plan.  The two records answer different questions:
the authorization record says a protocol was approved; the review record says
the specific preliminary review gating live use took place and passed.  A
record authorising computational work has never been sufficient to unlock a
live-patient plan, and this module is where that stops depending on anyone
remembering it.

WHAT THIS MODULE DOES NOT ESTABLISH -- READ THIS
------------------------------------------------
Exactly as in :mod:`scwbd.schema.authorization`: validation here checks that a
**declaration** is well formed, complete, internally consistent, and in scope.
It does not contact a review board, an IRB, an institution or a regulator, and
it cannot confirm that the review occurred, that the outcome recorded is the
outcome reached, or that the person who signed the record was entitled to sign
it.  **No software can do that.**  A record that passes every check here
establishes one thing: *a claim that the review happened was recorded, and that
claim is internally coherent with what is being requested.*  Whether it is true
is a fact about the world held by the people who signed it.

Nothing here authorises stimulating anybody.  This package builds no
stimulation controller, no device command path and no dosing computation for a
person; a review record is an input to a *refusal*, not a permission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal, Mapping, Sequence

from ..schema.authorization import (
    AuthorizationRecord,
    AuthorizationVerdict,
    epoch_seconds,
    is_placeholder,
    validate_authorization,
)
from .base import InterventionRefusal

__all__ = [
    "LIVE_APPLICATION_DECLARATION_NOTICE",
    "PRELIMINARY_REVIEW_SCHEDULED",
    "PRELIMINARY_REVIEW_SCHEDULED_S",
    "ApplicationMode",
    "LiveApplicationVerdict",
    "PreliminaryReviewRecord",
    "ReviewFailure",
    "ReviewFailureCode",
    "ReviewOutcome",
    "authorize_live_application",
    "review_has_occurred",
]

# ---------------------------------------------------------------------------
# the one date
# ---------------------------------------------------------------------------

#: The preliminary review that gates live application.  **One constant, one
#: file.**  Correcting it is a one-line change and every refusal message,
#: every failure code and every test derives from it rather than restating it.
#:
#: This is the date the review is *scheduled* for.  It is used only as the
#: earliest date on which a review record could honestly exist.  Nothing in
#: this module opens because this date has passed; see
#: :func:`authorize_live_application`.
PRELIMINARY_REVIEW_SCHEDULED: date = date(2026, 8, 25)

#: The same instant on the ``wall`` clock, for comparison against request and
#: record times.  Derived, never restated.
PRELIMINARY_REVIEW_SCHEDULED_S: float = epoch_seconds(
    PRELIMINARY_REVIEW_SCHEDULED.isoformat()
)

#: Attached to every verdict, admitted or refused.
LIVE_APPLICATION_DECLARATION_NOTICE = (
    "DECLARATION, NOT VERIFICATION. A preliminary review record states a claim "
    "that a review occurred and reached an outcome, made by the people "
    "responsible for it. Validation checks only that the claim is complete, "
    "internally consistent, dated no earlier than the scheduled review, not "
    "dated in the future, and in scope for the request. It does not confirm "
    "with any review board, IRB, institution or regulator that the review took "
    "place or that the recorded outcome is the outcome reached, and no software "
    "can. A validated record means a claim was recorded and checked, never that "
    "permission was granted."
)

#: What a plan is *for*.  ``"computational"`` is the default and is not gated
#: by this module: simulation, modelling, intervention physics, dose-response
#: on simulated tissue, planning against open data, benchmarks and training all
#: sit here.  ``"live"`` means the plan is intended to drive real hardware or
#: to be applied to a person, and is refused until the review record exists.
ApplicationMode = Literal["computational", "live"]

ReviewOutcome = Literal[
    "approved",
    "approved_with_conditions",
    "not_approved",
    "deferred",
    "undeclared",
]

#: Outcomes that can admit.  ``approved_with_conditions`` admits only when
#: every declared condition is recorded as satisfied.
_APPROVING: frozenset[str] = frozenset({"approved", "approved_with_conditions"})

ReviewFailureCode = Literal[
    "REVIEW_ABSENT",
    "REVIEW_NOT_YET_OCCURRED",
    "REVIEW_BEFORE_SCHEDULED",
    "REVIEW_TIME_UNDECLARED",
    "REVIEW_OUTCOME_NOT_APPROVING",
    "REVIEW_CONDITIONS_UNMET",
    "REVIEW_FIELD_MISSING",
    "REVIEW_SCOPE_MISMATCH",
    "REVIEW_AUTHORIZATION_MISSING",
]


@dataclass(frozen=True)
class ReviewFailure:
    """One distinguishable way a live-application request fails."""

    code: ReviewFailureCode
    field: str
    detail: str
    remedy: str = ""

    def __str__(self) -> str:
        return f"{self.code} [{self.field}]: {self.detail}"

    def as_provenance(self) -> dict[str, str]:
        return {
            "code": self.code,
            "field": self.field,
            "detail": self.detail,
            "remedy": self.remedy,
        }


@dataclass(frozen=True)
class PreliminaryReviewRecord:
    """A recorded claim that the preliminary review **happened**.

    Deliberately not the same type as
    :class:`~scwbd.schema.authorization.AuthorizationRecord`.  That record says
    a protocol was approved; this one says the specific review gating live use
    took place and reached an outcome.  Keeping them separate is what makes
    "approved for computational studies" structurally incapable of unlocking a
    live-patient plan: satisfying one says nothing about the other.
    """

    #: Who conducted the review, named.
    review_body: str
    #: The review's own identifier, as issued.
    identifier: str
    #: The date the review is claimed to have occurred, ISO ``YYYY-MM-DD``.
    occurred_on: str | None
    outcome: ReviewOutcome = "undeclared"
    #: Intervention classes the review's outcome covers.  A review of a TMS
    #: protocol does not clear a tFUS one.
    covered_intervention_classes: tuple[str, ...] = ()
    #: Conditions attached to an ``approved_with_conditions`` outcome.
    conditions: tuple[str, ...] = ()
    #: Conditions the requester records as satisfied.  Must cover every entry
    #: in :attr:`conditions` for that outcome to admit.
    conditions_satisfied: tuple[str, ...] = ()
    #: Who recorded this declaration.  Not the same as who reviewed.
    declared_by: str = ""
    notes: Mapping[str, Any] = field(default_factory=dict)

    @property
    def occurred_on_s(self) -> float | None:
        if not self.occurred_on or is_placeholder(self.occurred_on):
            return None
        try:
            return epoch_seconds(self.occurred_on)
        except (ValueError, TypeError):
            return None

    def as_provenance(self) -> dict[str, Any]:
        return {
            "review_body": self.review_body,
            "identifier": self.identifier,
            "occurred_on": self.occurred_on,
            "outcome": self.outcome,
            "covered_intervention_classes": list(self.covered_intervention_classes),
            "conditions": list(self.conditions),
            "conditions_satisfied": list(self.conditions_satisfied),
            "declared_by": self.declared_by,
            "scheduled_review_date": PRELIMINARY_REVIEW_SCHEDULED.isoformat(),
        }


@dataclass(frozen=True)
class LiveApplicationVerdict:
    """The result of asking whether a live plan may proceed.  Never raises."""

    admitted: bool
    mode: ApplicationMode
    intervention_class: str
    failures: tuple[ReviewFailure, ...] = ()
    review: PreliminaryReviewRecord | None = None
    authorization: AuthorizationVerdict | None = None
    at_time_s: float | None = None
    notice: str = LIVE_APPLICATION_DECLARATION_NOTICE

    @property
    def failure_codes(self) -> tuple[str, ...]:
        return tuple(f.code for f in self.failures)

    def reason(self) -> str:
        if self.admitted:
            return (
                f"a preliminary review record covering {self.intervention_class} "
                f"was declared, dated {self.review.occurred_on if self.review else '?'}, "
                f"outcome {self.review.outcome if self.review else '?'}"
            )
        return "; ".join(str(f) for f in self.failures) or "refused"

    def as_provenance(self) -> dict[str, Any]:
        return {
            "admitted": self.admitted,
            "application_mode": self.mode,
            "intervention_class": self.intervention_class,
            "failures": [f.as_provenance() for f in self.failures],
            "failure_codes": list(self.failure_codes),
            "review": self.review.as_provenance() if self.review else None,
            "authorization": (
                self.authorization.as_provenance() if self.authorization else None
            ),
            "scheduled_review_date": PRELIMINARY_REVIEW_SCHEDULED.isoformat(),
            "notice": self.notice,
        }

    def raise_if_refused(self, offending: Any = None) -> None:
        """Raise ``InterventionRefusal(code="R11")`` naming every specific cause."""
        if self.admitted:
            return
        raise InterventionRefusal(
            "R11",
            (
                f"a live application of a {self.intervention_class} plan is "
                f"refused: {self.reason()}"
            ),
            remedy="; ".join(f.remedy for f in self.failures if f.remedy)
            or (
                "supply a PreliminaryReviewRecord showing the review occurred "
                "with an approving outcome"
            ),
            offending_object=offending,
        )


def review_has_occurred(
    review: PreliminaryReviewRecord | None, *, at_time_s: float | None
) -> bool:
    """Whether the record credibly claims a *completed* review, as of ``at_time_s``.

    Separated out so the property can be read on its own, and so nothing in
    this module can accidentally substitute ``at_time_s >=
    PRELIMINARY_REVIEW_SCHEDULED_S`` for it.  The scheduled date never appears
    on the admitting side of any comparison.
    """
    if review is None or at_time_s is None:
        return False
    t = review.occurred_on_s
    if t is None:
        return False
    return PRELIMINARY_REVIEW_SCHEDULED_S <= t <= at_time_s


def _require(value: str | None, field_name: str, detail: str) -> ReviewFailure | None:
    if value is None or not str(value).strip() or is_placeholder(str(value)):
        return ReviewFailure(
            "REVIEW_FIELD_MISSING", field_name, detail, remedy=f"record {field_name}"
        )
    return None


def authorize_live_application(
    *,
    mode: ApplicationMode,
    intervention_class: str,
    at_time_s: float | None,
    review: PreliminaryReviewRecord | None = None,
    authorization: AuthorizationRecord | None = None,
    a_safe_id: str | None = None,
    required_a_safe_axes: Sequence[str] = (),
) -> LiveApplicationVerdict:
    """Decide whether a plan may be applied live.  Returns a verdict; never raises.

    ``mode="computational"`` admits unconditionally and is the ordinary path
    for everything in this repository.  This function deliberately does no work
    at all in that case: simulation is not gated, and a gate that fires on
    simulation would be a gate everyone learns to route around.

    ``mode="live"`` requires, conjunctively:

    1. a validated :class:`~scwbd.schema.authorization.AuthorizationRecord`
       covering ``intervention_class`` at ``at_time_s`` -- necessary, and on
       its own **not sufficient**;
    2. a :class:`PreliminaryReviewRecord` that names its body and identifier,
       claims a date no earlier than :data:`PRELIMINARY_REVIEW_SCHEDULED` and
       no later than ``at_time_s``, records an approving outcome, covers
       ``intervention_class``, and has every attached condition recorded as
       satisfied.

    Every distinguishable failure gets its own code, so a refusal says which
    thing is wrong rather than restating the policy.
    """
    if mode == "computational":
        return LiveApplicationVerdict(
            admitted=True,
            mode="computational",
            intervention_class=intervention_class,
            at_time_s=at_time_s,
        )

    failures: list[ReviewFailure] = []

    # -- 1. authorization is necessary, never sufficient ---------------------
    auth_verdict = validate_authorization(
        authorization,
        intervention_class=intervention_class,
        at_time_s=at_time_s,
        a_safe_id=a_safe_id,
        required_a_safe_axes=tuple(required_a_safe_axes),
        what=f"live application of a {intervention_class} plan",
    )
    if not auth_verdict.admitted:
        failures.append(
            ReviewFailure(
                "REVIEW_AUTHORIZATION_MISSING",
                "authorization",
                (
                    "a live application additionally requires a validated "
                    f"AuthorizationRecord, and none admits this request: "
                    f"{auth_verdict.reason()}"
                ),
                remedy=(
                    "supply a complete, in-date AuthorizationRecord covering "
                    "this intervention class"
                ),
            )
        )

    # -- 2. the review must have happened ------------------------------------
    if review is None:
        # The whole point of the module, stated once, in the one place a caller
        # will actually read it.
        failures.append(
            ReviewFailure(
                "REVIEW_ABSENT",
                "review",
                (
                    "no preliminary review record was supplied. Live application "
                    "is gated on a review scheduled for "
                    f"{PRELIMINARY_REVIEW_SCHEDULED.isoformat()}; a scheduled "
                    "review is not a completed one, and this refusal does not "
                    "lift when that date passes"
                ),
                remedy=(
                    "supply a PreliminaryReviewRecord recording that the review "
                    "occurred and what outcome it reached"
                ),
            )
        )
        return LiveApplicationVerdict(
            admitted=False,
            mode="live",
            intervention_class=intervention_class,
            failures=tuple(failures),
            review=None,
            authorization=auth_verdict,
            at_time_s=at_time_s,
        )

    for f in (
        _require(review.review_body, "review_body", "the reviewing body is not named"),
        _require(
            review.identifier,
            "identifier",
            "the review has no recorded identifier",
        ),
    ):
        if f is not None:
            failures.append(f)

    if at_time_s is None:
        failures.append(
            ReviewFailure(
                "REVIEW_TIME_UNDECLARED",
                "at_time_s",
                (
                    "the request carries no time, so a review claimed to have "
                    "occurred cannot be checked against when it is being relied on"
                ),
                remedy="date the request on the wall clock",
            )
        )

    occurred = review.occurred_on_s
    if occurred is None:
        failures.append(
            ReviewFailure(
                "REVIEW_NOT_YET_OCCURRED",
                "occurred_on",
                (
                    "the review record declares no date on which the review "
                    "occurred; an undated review is not a completed review"
                ),
                remedy="record the date the review actually took place",
            )
        )
    else:
        if occurred < PRELIMINARY_REVIEW_SCHEDULED_S:
            failures.append(
                ReviewFailure(
                    "REVIEW_BEFORE_SCHEDULED",
                    "occurred_on",
                    (
                        f"the record claims the review occurred on "
                        f"{review.occurred_on}, before the review scheduled for "
                        f"{PRELIMINARY_REVIEW_SCHEDULED.isoformat()}; it cannot "
                        "be a record of that review"
                    ),
                    remedy=(
                        "record the preliminary review that gates live "
                        "application, not an earlier one"
                    ),
                )
            )
        if at_time_s is not None and occurred > at_time_s:
            failures.append(
                ReviewFailure(
                    "REVIEW_NOT_YET_OCCURRED",
                    "occurred_on",
                    (
                        f"the record claims the review occurred on "
                        f"{review.occurred_on}, which is in the future relative "
                        "to this request; a review that has not happened cannot "
                        "have an outcome"
                    ),
                    remedy="request this after the review has actually occurred",
                )
            )

    if review.outcome not in _APPROVING:
        failures.append(
            ReviewFailure(
                "REVIEW_OUTCOME_NOT_APPROVING",
                "outcome",
                (
                    f"the review outcome is recorded as {review.outcome!r}; live "
                    "application requires an approving outcome"
                ),
                remedy="live application is not available on this outcome",
            )
        )
    elif review.outcome == "approved_with_conditions":
        unmet = tuple(
            c for c in review.conditions if c not in set(review.conditions_satisfied)
        )
        if not review.conditions:
            failures.append(
                ReviewFailure(
                    "REVIEW_CONDITIONS_UNMET",
                    "conditions",
                    (
                        "the outcome is 'approved_with_conditions' but no "
                        "conditions are recorded, so there is nothing to satisfy "
                        "and nothing to check"
                    ),
                    remedy="record the conditions the review attached",
                )
            )
        elif unmet:
            failures.append(
                ReviewFailure(
                    "REVIEW_CONDITIONS_UNMET",
                    "conditions",
                    f"conditions not recorded as satisfied: {list(unmet)}",
                    remedy="satisfy and record each condition the review attached",
                )
            )

    if intervention_class not in review.covered_intervention_classes:
        failures.append(
            ReviewFailure(
                "REVIEW_SCOPE_MISMATCH",
                "covered_intervention_classes",
                (
                    f"the review covers "
                    f"{list(review.covered_intervention_classes)} and does not "
                    f"cover {intervention_class!r}"
                ),
                remedy=f"obtain a review covering {intervention_class!r}",
            )
        )

    return LiveApplicationVerdict(
        admitted=not failures,
        mode="live",
        intervention_class=intervention_class,
        failures=tuple(failures),
        review=review,
        authorization=auth_verdict,
        at_time_s=at_time_s,
    )
