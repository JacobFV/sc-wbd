"""The live-application gate: what leaves this repo toward a person.

Two properties, and the second is the one that matters:

1. the computational path -- everything this repository actually does -- is
   **not** gated, and admits without a review record;
2. **the live path refuses, and keeps refusing.**  It refuses today for the
   review-pending reason; it refuses in 2027 with no review record, so the
   scheduled date is not an unlock; it refuses a review dated before the
   scheduled one as impossible; and an ``AuthorizationRecord`` covering
   computational work does not lift it.

Every refusal below is executed, not described.  A gate nobody has seen refuse
is indistinguishable from one that cannot -- see ``reports/decorative_guards.md``.
"""

from __future__ import annotations

import pytest

from scwbd.intervene.deployment import (
    PRELIMINARY_REVIEW_SCHEDULED,
    PRELIMINARY_REVIEW_SCHEDULED_S,
    PreliminaryReviewRecord,
    authorize_live_application,
    review_has_occurred,
)
from scwbd.intervene.safety import (
    AuthorizationGate,
    AuthorizedRequest,
    FeasibleSet,
    ProposedIntervention,
)
from scwbd.intervene.base import InterventionRefusal
from scwbd.schema.authorization import epoch_seconds

#: Inside the fixture authorization's validity window, and before the review.
TODAY_S = epoch_seconds("2026-08-06")
#: Well after the scheduled review. Used to prove the date is not an unlock.
LONG_AFTER_S = epoch_seconds("2027-12-31")
#: The day after the scheduled review.
DAY_AFTER_REVIEW_S = epoch_seconds("2026-08-26")

_EXPOSURE = {
    "tms.peak_efield_v_per_m": 95.0,
    "tms.pulses_per_session": 600.0,
    "tms.coil_scalp_distance_mm": 4.0,
    "tms.frequency_hz": 10.0,
    "tms.intertrain_interval_s": 26.0,
    "protocol.session_duration_s": 1200.0,
}


def _approving_review(**over) -> PreliminaryReviewRecord:
    """A *fictional* record that the review happened and passed."""
    fields = dict(
        review_body="Example University preliminary review panel",
        identifier="PRELIM-2026-0001",
        occurred_on="2026-08-25",
        outcome="approved",
        covered_intervention_classes=("tms",),
        declared_by="test fixture",
    )
    fields.update(over)
    return PreliminaryReviewRecord(**fields)


def _proposal(application="computational", **over) -> ProposedIntervention:
    fields = dict(
        label="pose_a",
        modality="tms",
        exposure=dict(_EXPOSURE),
        pose_certified=True,
        reversible=True,
        application=application,
    )
    fields.update(over)
    return ProposedIntervention(**fields)


# ---------------------------------------------------------------------------
# 1. the computational path is not gated
# ---------------------------------------------------------------------------


class TestComputationalWorkIsNotGated:
    def test_a_computational_plan_admits_with_no_review_record(self):
        v = authorize_live_application(
            mode="computational",
            intervention_class="tms",
            at_time_s=TODAY_S,
            review=None,
            authorization=None,
        )
        assert v.admitted
        assert v.failures == ()

    def test_it_admits_even_with_no_authorization_record_at_all(self):
        """Simulation is not gated on governance by this module.

        The `AuthorizationRecord` gate still applies wherever it applied
        before; this module simply does not add a second one on the
        computational path.
        """
        v = authorize_live_application(
            mode="computational",
            intervention_class="tfus",
            at_time_s=None,
            review=None,
            authorization=None,
        )
        assert v.admitted

    def test_the_verdict_still_records_that_it_was_not_live(self):
        """Absence of a gate is recorded, not left to be inferred."""
        v = authorize_live_application(
            mode="computational", intervention_class="tms", at_time_s=TODAY_S
        )
        assert v.as_provenance()["application_mode"] == "computational"

    def test_an_admitted_computational_verdict_does_not_raise(self):
        authorize_live_application(
            mode="computational", intervention_class="tms", at_time_s=TODAY_S
        ).raise_if_refused()


# ---------------------------------------------------------------------------
# 2. the live path refuses -- each cause fired separately
# ---------------------------------------------------------------------------


class TestTheLivePathRefuses:
    def test_a_live_plan_today_refuses_with_the_review_pending_reason(
        self, authorization
    ):
        v = authorize_live_application(
            mode="live",
            intervention_class="tms",
            at_time_s=TODAY_S,
            review=None,
            authorization=authorization,
        )
        assert not v.admitted
        assert "REVIEW_ABSENT" in v.failure_codes
        # the refusal names the pending review, not "no ethics approval"
        assert PRELIMINARY_REVIEW_SCHEDULED.isoformat() in v.reason()
        assert "no ethics approval" not in v.reason()

    def test_it_raises_r11_naming_the_cause(self, authorization):
        v = authorize_live_application(
            mode="live",
            intervention_class="tms",
            at_time_s=TODAY_S,
            authorization=authorization,
        )
        with pytest.raises(InterventionRefusal) as e:
            v.raise_if_refused(offending="pose_a")
        assert e.value.code == "R11"
        assert "preliminary review" in str(e.value).lower()

    def test_a_review_dated_before_the_scheduled_one_is_impossible(
        self, authorization
    ):
        v = authorize_live_application(
            mode="live",
            intervention_class="tms",
            at_time_s=DAY_AFTER_REVIEW_S,
            review=_approving_review(occurred_on="2026-08-24"),
            authorization=authorization,
        )
        assert not v.admitted
        assert "REVIEW_BEFORE_SCHEDULED" in v.failure_codes

    def test_a_review_dated_in_the_future_has_not_happened(self, authorization):
        v = authorize_live_application(
            mode="live",
            intervention_class="tms",
            at_time_s=TODAY_S,
            review=_approving_review(occurred_on="2026-08-25"),
            authorization=authorization,
        )
        assert not v.admitted
        assert "REVIEW_NOT_YET_OCCURRED" in v.failure_codes

    def test_an_undated_review_is_not_a_completed_review(self, authorization):
        v = authorize_live_application(
            mode="live",
            intervention_class="tms",
            at_time_s=LONG_AFTER_S,
            review=_approving_review(occurred_on=None),
            authorization=authorization,
        )
        assert not v.admitted
        assert "REVIEW_NOT_YET_OCCURRED" in v.failure_codes

    @pytest.mark.parametrize("outcome", ["not_approved", "deferred", "undeclared"])
    def test_a_non_approving_outcome_refuses_and_says_which(
        self, authorization, outcome
    ):
        v = authorize_live_application(
            mode="live",
            intervention_class="tms",
            at_time_s=LONG_AFTER_S,
            review=_approving_review(outcome=outcome),
            authorization=authorization,
        )
        assert not v.admitted
        assert "REVIEW_OUTCOME_NOT_APPROVING" in v.failure_codes
        assert outcome in v.reason()

    def test_unmet_conditions_refuse(self, authorization):
        v = authorize_live_application(
            mode="live",
            intervention_class="tms",
            at_time_s=LONG_AFTER_S,
            review=_approving_review(
                outcome="approved_with_conditions",
                conditions=("dosimetry re-check", "on-site physicist present"),
                conditions_satisfied=("dosimetry re-check",),
            ),
            authorization=authorization,
        )
        assert not v.admitted
        assert "REVIEW_CONDITIONS_UNMET" in v.failure_codes
        assert "on-site physicist present" in v.reason()

    def test_conditional_approval_with_no_conditions_recorded_refuses(
        self, authorization
    ):
        """Nothing to satisfy is not the same as everything satisfied."""
        v = authorize_live_application(
            mode="live",
            intervention_class="tms",
            at_time_s=LONG_AFTER_S,
            review=_approving_review(outcome="approved_with_conditions"),
            authorization=authorization,
        )
        assert not v.admitted
        assert "REVIEW_CONDITIONS_UNMET" in v.failure_codes

    def test_a_tms_review_does_not_clear_a_tfus_plan(self, make_authorization):
        v = authorize_live_application(
            mode="live",
            intervention_class="tfus",
            at_time_s=LONG_AFTER_S,
            review=_approving_review(),  # covers tms only
            authorization=make_authorization(
                authorized_intervention_classes=("tfus",)
            ),
        )
        assert not v.admitted
        assert "REVIEW_SCOPE_MISMATCH" in v.failure_codes

    @pytest.mark.parametrize("blank", ["", "   ", "TBD", "n/a"])
    def test_an_unnamed_review_body_refuses(self, authorization, blank):
        v = authorize_live_application(
            mode="live",
            intervention_class="tms",
            at_time_s=LONG_AFTER_S,
            review=_approving_review(review_body=blank),
            authorization=authorization,
        )
        assert not v.admitted
        assert "REVIEW_FIELD_MISSING" in v.failure_codes

    @pytest.mark.parametrize("blank", ["", "TODO", "placeholder"])
    def test_an_unidentified_review_refuses(self, authorization, blank):
        v = authorize_live_application(
            mode="live",
            intervention_class="tms",
            at_time_s=LONG_AFTER_S,
            review=_approving_review(identifier=blank),
            authorization=authorization,
        )
        assert not v.admitted
        assert "REVIEW_FIELD_MISSING" in v.failure_codes

    def test_an_undated_request_cannot_be_checked(self, authorization):
        v = authorize_live_application(
            mode="live",
            intervention_class="tms",
            at_time_s=None,
            review=_approving_review(),
            authorization=authorization,
        )
        assert not v.admitted
        assert "REVIEW_TIME_UNDECLARED" in v.failure_codes


# ---------------------------------------------------------------------------
# 3. the date is a lower bound, never an unlock  <-- the load-bearing test
# ---------------------------------------------------------------------------


class TestTheDateIsNotAnUnlock:
    def test_a_live_plan_long_after_the_review_date_still_refuses(
        self, authorization
    ):
        """The whole point.

        If this gate were a calendar comparison, this request -- dated more
        than a year past the scheduled review, with a fully valid
        authorization -- would be admitted. It is refused, because no record
        says the review happened. A date passing is not evidence of an outcome.
        """
        v = authorize_live_application(
            mode="live",
            intervention_class="tms",
            at_time_s=LONG_AFTER_S,
            review=None,
            authorization=authorization,
        )
        assert not v.admitted
        assert "REVIEW_ABSENT" in v.failure_codes

    def test_the_review_refusal_does_not_lift_with_the_passage_of_time(
        self, authorization
    ):
        """``REVIEW_ABSENT`` is present on both sides of the scheduled date.

        The verdicts are *not* identical, and the difference is instructive:
        at 2027-12-31 the fixture authorization has also expired (its window
        ends 2027-01-04), so the later request refuses for two reasons rather
        than one. Governance decaying over time while the review requirement
        holds constant is the correct shape -- an approval is a thing that
        lapses, a completed review is a thing that either happened or did not.
        """
        before = authorize_live_application(
            mode="live",
            intervention_class="tms",
            at_time_s=TODAY_S,
            authorization=authorization,
        )
        after = authorize_live_application(
            mode="live",
            intervention_class="tms",
            at_time_s=LONG_AFTER_S,
            authorization=authorization,
        )
        assert "REVIEW_ABSENT" in before.failure_codes
        assert "REVIEW_ABSENT" in after.failure_codes
        assert not before.admitted and not after.admitted
        # the later one additionally fails governance, and says so
        assert "REVIEW_AUTHORIZATION_MISSING" in after.failure_codes
        assert "AUTH_EXPIRED" in after.authorization.failure_codes

    def test_review_has_occurred_is_false_on_the_calendar_alone(self):
        assert not review_has_occurred(None, at_time_s=LONG_AFTER_S)

    def test_review_has_occurred_needs_a_record_dated_in_the_window(self):
        rec = _approving_review()
        assert not review_has_occurred(rec, at_time_s=TODAY_S)  # not yet
        assert review_has_occurred(rec, at_time_s=DAY_AFTER_REVIEW_S)

    def test_the_scheduled_date_appears_exactly_once_as_a_literal(self):
        """A date restated in several files goes stale in all but one of them."""
        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parents[2] / "scwbd"
        literal = re.compile(r"2026[-/,\s]*0?8[-/,\s]*25")
        hits = [
            p
            for p in root.rglob("*.py")
            if literal.search(p.read_text(encoding="utf-8"))
        ]
        assert [p.name for p in hits] == ["deployment.py"], (
            f"the review date is written out in {[str(p) for p in hits]}; it "
            "must exist once, as PRELIMINARY_REVIEW_SCHEDULED"
        )

    def test_the_seconds_constant_is_derived_not_restated(self):
        from scwbd.schema.authorization import epoch_seconds as es

        assert PRELIMINARY_REVIEW_SCHEDULED_S == es(
            PRELIMINARY_REVIEW_SCHEDULED.isoformat()
        )


# ---------------------------------------------------------------------------
# 4. an authorization for computational work does not unlock live use
# ---------------------------------------------------------------------------


class TestAuthorizationIsNecessaryNotSufficient:
    def test_a_valid_authorization_alone_does_not_admit_a_live_plan(
        self, authorization
    ):
        """The defect this whole module exists to prevent.

        The authorization record here is complete, in date, TMS-scoped and
        admits at the governance gate. It still does not unlock live use.
        """
        from scwbd.schema.authorization import validate_authorization

        assert validate_authorization(
            authorization, intervention_class="tms", at_time_s=TODAY_S
        ).admitted

        v = authorize_live_application(
            mode="live",
            intervention_class="tms",
            at_time_s=TODAY_S,
            authorization=authorization,
        )
        assert not v.admitted

    def test_a_review_without_an_authorization_also_refuses(self):
        v = authorize_live_application(
            mode="live",
            intervention_class="tms",
            at_time_s=DAY_AFTER_REVIEW_S,
            review=_approving_review(),
            authorization=None,
        )
        assert not v.admitted
        assert "REVIEW_AUTHORIZATION_MISSING" in v.failure_codes

    def test_both_together_admit(self, authorization):
        """The gate is real in both directions: it can also let something through."""
        v = authorize_live_application(
            mode="live",
            intervention_class="tms",
            at_time_s=DAY_AFTER_REVIEW_S,
            review=_approving_review(),
            authorization=authorization,
        )
        assert v.admitted, v.reason()
        assert v.as_provenance()["review"]["occurred_on"] == "2026-08-25"


# ---------------------------------------------------------------------------
# 5. wired into the gate the rest of the stack actually calls
# ---------------------------------------------------------------------------


class TestTheGateEnforcesIt:
    @pytest.fixture
    def gate(self):
        return AuthorizationGate(FeasibleSet(), a_safe_id="EX-TMS-DLPFC-01-asafe")

    def test_a_computational_proposal_is_admitted(self, gate, authorization):
        admitted = gate.admit(
            _proposal(),
            AuthorizedRequest(
                record=authorization, intervention_class="tms", at_time_s=TODAY_S
            ),
        )
        assert admitted.application is not None
        assert admitted.application.mode == "computational"

    def test_the_same_proposal_marked_live_refuses(self, gate, authorization):
        """Identical exposure, identical authorization; only the intent differs."""
        with pytest.raises(InterventionRefusal) as e:
            gate.admit(
                _proposal(application="live"),
                AuthorizedRequest(
                    record=authorization,
                    intervention_class="tms",
                    at_time_s=TODAY_S,
                ),
            )
        assert e.value.code == "R11"
        assert "review" in str(e.value).lower()

    def test_a_live_proposal_with_a_completed_review_is_admitted(
        self, gate, authorization
    ):
        admitted = gate.admit(
            _proposal(application="live"),
            AuthorizedRequest(
                record=authorization,
                intervention_class="tms",
                at_time_s=DAY_AFTER_REVIEW_S,
                review=_approving_review(),
            ),
        )
        assert admitted.application is not None
        assert admitted.application.admitted

    def test_a_live_proposal_outside_a_safe_still_refuses_on_a_safe(
        self, gate, authorization
    ):
        """The review record does not widen the feasible set, and cannot."""
        with pytest.raises(InterventionRefusal) as e:
            gate.admit(
                _proposal(
                    application="live",
                    exposure=dict(_EXPOSURE, **{"tms.peak_efield_v_per_m": 5000.0}),
                ),
                AuthorizedRequest(
                    record=authorization,
                    intervention_class="tms",
                    at_time_s=DAY_AFTER_REVIEW_S,
                    review=_approving_review(),
                ),
            )
        assert "peak_efield_v_per_m" in str(e.value)

    def test_the_admitted_proposal_carries_the_review_in_provenance(
        self, gate, authorization
    ):
        admitted = gate.admit(
            _proposal(application="live"),
            AuthorizedRequest(
                record=authorization,
                intervention_class="tms",
                at_time_s=DAY_AFTER_REVIEW_S,
                review=_approving_review(),
            ),
        )
        prov = admitted.provenance()["application"]
        assert prov["review"]["identifier"] == "PRELIM-2026-0001"
        assert prov["scheduled_review_date"] == "2026-08-25"

    def test_the_notice_says_it_is_a_declaration_not_a_verification(self):
        v = authorize_live_application(
            mode="live", intervention_class="tms", at_time_s=TODAY_S
        )
        assert "DECLARATION, NOT VERIFICATION" in v.notice
        assert "no software can" in v.notice
