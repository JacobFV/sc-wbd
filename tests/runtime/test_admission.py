"""Every admission condition, fired.

``reports/decorative_guards.md`` catalogues ~26 guards in this codebase that
looked green and could not fire.  The discipline this file follows, for each
condition:

1. make it **fail**, and assert the refusal names the condition by code;
2. make it **pass** on an input that differs *only* in that condition, so the
   check is shown to discriminate rather than to be permanently red or
   permanently green.

(2) is the part that is usually missing.  A gate that can never open is as
useless as one that never closes, and a test suite that only ever exercises the
refusal cannot tell the two apart.
"""

from __future__ import annotations

import pytest

from scwbd.intervene.deployment import (
    PRELIMINARY_REVIEW_SCHEDULED,
    PreliminaryReviewRecord,
)
from scwbd.runtime.admission import (
    CONSUMER_STANDING_INVARIANTS,
    EXPORT_PURPOSES,
    LIVE_PURPOSES,
    MODE_OF_PURPOSE,
    CheckpointClaims,
    CheckpointRefused,
    ConsumerInvariants,
    ConsumerInvariantViolation,
    admit,
)
from scwbd.schema.authorization import epoch_seconds

#: The review date, derived from the one constant and never restated.
REVIEW_DAY = PRELIMINARY_REVIEW_SCHEDULED.isoformat()
#: A time after the review could have happened.
AFTER_S = epoch_seconds(REVIEW_DAY) + 86400.0
#: Years later. Used to show the date is not an unlock at *this* call site too.
LONG_AFTER_S = epoch_seconds("2027-12-31")


def clean_claims(**overrides) -> CheckpointClaims:
    """A checkpoint that passes every condition.  Override one to break one."""
    base = dict(
        manifest_id="run2-treatment-arm",
        claim_class="mechanistic",
        posterior_class="generalized",
        is_control_arm=False,
        control_arm_of="treatment",
        anatomy_is_biological=True,
        anatomy_provenance="enigma_hcp_dki",
        gates={"G1": "PASS", "G2": "PASS", "G3": "PASS"},
        weights_trained=True,
        port_contract_digest="deadbeef",
    )
    base.update(overrides)
    return CheckpointClaims(**base)


def approving_review(**overrides) -> PreliminaryReviewRecord:
    """A *fictional* record that the review happened and passed."""
    base = dict(
        review_body="Example University preliminary review panel",
        identifier="PRELIM-2026-0001",
        occurred_on=REVIEW_DAY,
        outcome="approved",
        covered_intervention_classes=("tms",),
        declared_by="test fixture",
    )
    base.update(overrides)
    return PreliminaryReviewRecord(**base)


#: Sentinel so that ``as_of=None`` can be passed *deliberately* and reach the
#: gate as None. Defaulting it to "now" here would have hidden the undated-
#: request refusal behind the helper.
_UNSET = object()


def refusal(claims, *, purpose="live_hardware", review=None, authorization=None,
            as_of=_UNSET, **kw):
    with pytest.raises(CheckpointRefused) as exc:
        admit(
            claims,
            purpose=purpose,
            review=review,
            authorization=authorization,
            as_of=AFTER_S if as_of is _UNSET else as_of,
            **kw,
        )
    return exc.value


# ---------------------------------------------------------------------------
# the baseline: the gate opens
# ---------------------------------------------------------------------------

class TestTheGateCanOpen:
    """Without this, every refusal below is unfalsifiable."""

    def test_a_clean_checkpoint_with_an_approving_record_is_admitted(self, authorization):
        v = admit(
            clean_claims(),
            purpose="live_hardware",
            review=approving_review(),
            authorization=authorization,
            as_of=AFTER_S,
        )
        assert v.admitted
        assert v.failed == ()
        assert {c.code for c in v.conditions} == {"A0", "A1", "A2", "A3", "A4", "A5", "A6"}
        assert v.live_application.admitted
        assert v.live_application.mode == "live"

    def test_simulation_asks_nothing_of_the_checkpoint(self):
        """A simulation reported as a simulation needs no artifact to be valid."""
        v = admit(CheckpointClaims.absent(), purpose="simulation", as_of=AFTER_S)
        assert v.admitted
        # ...but every non-required condition is still *recorded* as having
        # failed, so the verdict shows what was not asked rather than hiding it.
        not_required = {c.code for c in v.conditions if not c.required}
        assert {"A1", "A2", "A3", "A4", "A5", "A6"} <= not_required
        assert not all(c.passed for c in v.conditions)

    def test_the_admission_record_is_content_hashed(self):
        a = admit(clean_claims(), purpose="research_offline", as_of=AFTER_S)
        b = admit(clean_claims(), purpose="research_offline", as_of=AFTER_S)
        c = admit(clean_claims(manifest_id="other"), purpose="research_offline",
                  as_of=AFTER_S)
        assert a.content_hash() == b.content_hash()
        assert a.content_hash() != c.content_hash()


# ---------------------------------------------------------------------------
# A0 -- the standing invariants
# ---------------------------------------------------------------------------

class TestA0StandingInvariants:
    @pytest.mark.parametrize("name", sorted(CONSUMER_STANDING_INVARIANTS))
    def test_widening_any_invariant_is_refused(self, name):
        with pytest.raises(ConsumerInvariantViolation) as exc:
            ConsumerInvariants(**{name: True})
        assert name in str(exc.value)

    def test_the_default_is_all_false(self):
        assert ConsumerInvariants().as_dict() == dict(CONSUMER_STANDING_INVARIANTS)

    def test_an_approving_review_does_not_relax_them(self, authorization):
        """The review boundary and the promotion boundary are unrelated."""
        v = admit(
            clean_claims(),
            purpose="patient_directed",
            review=approving_review(),
            authorization=authorization,
            as_of=AFTER_S,
        )
        assert v.admitted
        assert v.invariants.as_dict() == {
            "sim2real_ready": False,
            "promotion_eligible": False,
            "robot_command_authority": False,
        }
        with pytest.raises(ConsumerInvariantViolation):
            admit(
                clean_claims(),
                purpose="patient_directed",
                review=approving_review(),
                authorization=authorization,
                invariants=ConsumerInvariants(promotion_eligible=True),
                as_of=AFTER_S,
            )


# ---------------------------------------------------------------------------
# A1..A5 -- what the checkpoint is
# ---------------------------------------------------------------------------

class TestA1ManifestPresent:
    def test_a_checkpoint_with_no_manifest_is_refused(self):
        e = refusal(CheckpointClaims.absent())
        assert "A1" in e.codes
        assert "claim_manifest.json" in str(e)

    def test_absence_is_not_permission_even_for_research(self):
        e = refusal(CheckpointClaims.absent(), purpose="research_offline")
        assert e.codes == ("A1",)

    def test_a_stated_manifest_passes_A1(self):
        v = admit(clean_claims(), purpose="research_offline", as_of=AFTER_S)
        assert v.admitted


class TestA2ControlArm:
    def test_the_control_arm_is_refused(self):
        e = refusal(clean_claims(is_control_arm=True, control_arm_of="body.tex 11.4"))
        assert "A2" in e.codes
        assert "control arm" in str(e)

    def test_an_artifact_that_does_not_say_is_treated_as_the_control_arm(self):
        """Run 1 was the control arm while claiming otherwise. Silence != treatment."""
        claims = CheckpointClaims.from_manifest({"id": "unsaid"})
        assert claims.is_control_arm is True
        assert "A2" in refusal(claims).codes

    def test_a_declared_treatment_arm_passes_A2(self):
        claims = CheckpointClaims.from_manifest(
            {"id": "run2", "is_control_arm": False, "arm": "treatment"}
        )
        assert claims.is_control_arm is False
        assert "A2" not in refusal(claims).codes


class TestA3AnatomyIsBiological:
    def test_synthetic_anatomy_is_refused(self):
        e = refusal(
            clean_claims(
                anatomy_is_biological=False, anatomy_provenance="synthetic_fallback"
            )
        )
        assert "A3" in e.codes
        assert "synthetic_fallback" in str(e)
        assert "not anatomy" in str(e)

    def test_an_unstated_anatomy_record_is_refused_not_assumed(self):
        claims = CheckpointClaims.from_manifest({"id": "x", "is_control_arm": False})
        assert claims.anatomy_is_biological is False
        assert "A3" in refusal(claims).codes

    def test_biological_anatomy_passes_A3(self):
        assert "A3" not in refusal(clean_claims()).codes


class TestA4Gates:
    def test_could_not_run_gates_are_refused(self):
        e = refusal(clean_claims(gates={"G1": "COULD_NOT_RUN", "G2": "PASS"}))
        assert "A4" in e.codes
        assert "COULD_NOT_RUN" in str(e) and "G1" in str(e)

    def test_a_failing_gate_is_refused(self):
        e = refusal(clean_claims(gates={"G1": "PASS", "G2": "FAIL"}))
        assert "A4" in e.codes
        assert "FAIL" in str(e) and "G2" in str(e)

    def test_no_gate_results_at_all_is_refused(self):
        e = refusal(clean_claims(gates={}))
        assert "A4" in e.codes
        assert "not an artifact whose gates passed" in str(e)

    def test_all_passing_gates_pass_A4(self):
        assert "A4" not in refusal(clean_claims()).codes


class TestA5TrainedWeights:
    def test_an_analytic_backend_is_refused_for_live_use(self):
        e = refusal(clean_claims(weights_trained=False))
        assert "A5" in e.codes
        assert "surrogate propagators" in str(e)

    def test_trained_weights_pass_A5(self):
        assert "A5" not in refusal(clean_claims()).codes


# ---------------------------------------------------------------------------
# A6 -- live-use authorization, by record and never by date
# ---------------------------------------------------------------------------

class TestA6DelegatesToTheOneLiveApplicationGate:
    """A6 has no logic of its own: it asks Faraday's predicate and records it.

    ``scwbd.intervene.deployment`` owns the rule and
    ``tests/intervene/test_deployment.py`` proves it.  What these tests prove is
    the thing that file cannot: that the *runtime* call site asks the same
    question, with the mode derived from the purpose rather than the caller.
    """

    @pytest.mark.parametrize("purpose", sorted(LIVE_PURPOSES))
    def test_no_review_record_refuses_every_live_purpose(self, purpose, authorization):
        e = refusal(clean_claims(), purpose=purpose, review=None,
                    authorization=authorization)
        assert e.codes == ("A6",)
        assert "REVIEW_ABSENT" in str(e) or "no preliminary review record" in str(e)

    @pytest.mark.parametrize(
        "purpose", [p for p in EXPORT_PURPOSES if p not in LIVE_PURPOSES]
    )
    def test_non_live_purposes_are_not_gated_on_a_review(self, purpose):
        v = admit(clean_claims(), purpose=purpose, as_of=AFTER_S)
        assert v.admitted
        assert v.live_application.mode == "computational"

    def test_the_mode_is_derived_from_the_purpose_not_the_caller(self):
        """A caller cannot get a live purpose checked as computational."""
        assert MODE_OF_PURPOSE == {
            "simulation": "computational",
            "research_offline": "computational",
            "live_hardware": "live",
            "patient_directed": "live",
        }
        for purpose in EXPORT_PURPOSES:
            v = admit(
                clean_claims(), purpose=purpose, as_of=AFTER_S,
                raise_on_refusal=False,
            )
            assert v.live_application.mode == MODE_OF_PURPOSE[purpose]

    def test_the_date_is_not_an_unlock_at_this_call_site_either(self, authorization):
        """Years past the scheduled review, with no record: still refused."""
        e = refusal(clean_claims(), review=None, authorization=authorization,
                    as_of=LONG_AFTER_S)
        assert e.codes == ("A6",)

    def test_an_authorization_alone_is_necessary_and_not_sufficient(self, authorization):
        with_auth = refusal(clean_claims(), review=None, authorization=authorization)
        assert with_auth.codes == ("A6",)
        # ...and adding the review is what admits it.
        assert admit(
            clean_claims(), purpose="live_hardware", review=approving_review(),
            authorization=authorization, as_of=AFTER_S,
        ).admitted

    def test_a_review_without_an_authorization_also_refuses(self):
        e = refusal(clean_claims(), review=approving_review(), authorization=None)
        assert e.codes == ("A6",)

    @pytest.mark.parametrize(
        "outcome", ["pending", "deferred", "disapproved", "undeclared"]
    )
    def test_a_non_approving_outcome_is_refused(self, outcome, authorization):
        e = refusal(
            clean_claims(),
            review=approving_review(outcome=outcome),
            authorization=authorization,
        )
        assert e.codes == ("A6",)

    def test_a_tms_review_does_not_clear_a_tfus_export(self, authorization):
        tms_review = approving_review(covered_intervention_classes=("tms",))
        # the runtime's default intervention_class is tms, so this admits...
        assert admit(
            clean_claims(), purpose="live_hardware", review=tms_review,
            authorization=authorization, as_of=AFTER_S,
        ).admitted
        # ...and the same record asked about tfus does not.
        e = refusal(
            clean_claims(), review=tms_review, authorization=authorization,
            intervention_class="tfus",
        )
        assert e.codes == ("A6",)

    def test_an_undated_request_cannot_be_checked_and_is_refused(self, authorization):
        """``as_of=None`` is not silently replaced with the wall clock."""
        e = refusal(clean_claims(), review=approving_review(),
                    authorization=authorization, as_of=None)
        assert e.codes == ("A6",)

    def test_as_of_accepts_a_date_an_iso_string_or_epoch_seconds(self, authorization):
        from datetime import date as _date

        results = [
            admit(clean_claims(), purpose="live_hardware", review=approving_review(),
                  authorization=authorization, as_of=v).admitted
            for v in (AFTER_S, REVIEW_DAY, _date.fromisoformat(REVIEW_DAY))
        ]
        assert results == [True, True, True]

    def test_the_verdict_carries_the_shared_provenance_object(self, authorization):
        v = admit(clean_claims(), purpose="live_hardware", review=approving_review(),
                  authorization=authorization, as_of=AFTER_S)
        prov = v.canonical()["live_application"]
        assert prov["application_mode"] == "live"
        assert prov["review"]["identifier"] == "PRELIM-2026-0001"
        # the one date, reported from the one place that owns it
        assert prov["scheduled_review_date"] == REVIEW_DAY


class TestRefusalsCompose:
    def test_the_run_one_artifact_fails_five_conditions_at_once(self):
        """What SC-WBD-001-beta actually is, stated as an admission verdict."""
        run1 = CheckpointClaims.from_manifest(
            {
                "id": "scwbd-001-beta",
                "claim_class": "surrogate",
                "arm": "equal_capacity_generic_operator_control",
                "anatomy": {
                    "is_biological": False,
                    "provenance": "synthetic_fallback",
                },
                "gates": {f"G{i}": "COULD_NOT_RUN" for i in range(1, 6)},
                "weights_trained": False,
            }
        )
        e = refusal(run1)
        assert set(e.codes) == {"A2", "A3", "A4", "A5", "A6"}
        # A1 passes: it *does* have a manifest. The refusal is about content.
        assert "A1" not in e.codes
        text = str(e)
        for code in ("A2", "A3", "A4", "A5", "A6"):
            assert f"[{code}]" in text

    def test_an_unknown_purpose_is_refused_rather_than_defaulted(self):
        with pytest.raises(ValueError) as exc:
            admit(clean_claims(), purpose="whatever")
        assert "no rules for" in str(exc.value)

    def test_the_non_raising_form_reports_without_admitting(self):
        v = admit(
            CheckpointClaims.absent(),
            purpose="live_hardware",
            as_of=AFTER_S,
            raise_on_refusal=False,
        )
        assert v.admitted is False
        assert {c.code for c in v.failed} >= {"A1", "A2", "A3", "A4", "A5", "A6"}
