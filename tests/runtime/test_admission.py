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

import json
from datetime import date, timedelta

import pytest

from scwbd.runtime.admission import (
    CONSUMER_STANDING_INVARIANTS,
    EARLIEST_CREDIBLE_REVIEW,
    EXPORT_PURPOSES,
    LIVE_PURPOSES,
    AuthorizationInvalid,
    CheckpointClaims,
    CheckpointRefused,
    ConsumerInvariants,
    ConsumerInvariantViolation,
    LiveUseAuthorization,
    admit,
)

AFTER = EARLIEST_CREDIBLE_REVIEW + timedelta(days=1)


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


def approving_record(**overrides) -> LiveUseAuthorization:
    base = dict(
        review_body="UT Arlington IRB",
        reference="2026-0817",
        outcome="approved",
        reviewed_on=AFTER,
        scope=("live_hardware", "patient_directed"),
        recorded_by="pi",
    )
    base.update(overrides)
    return LiveUseAuthorization(**base)


def refusal(claims, *, purpose="live_hardware", auth=None, as_of=None):
    with pytest.raises(CheckpointRefused) as exc:
        admit(
            claims,
            purpose=purpose,
            live_use_authorization=auth,
            as_of=as_of or AFTER + timedelta(days=30),
        )
    return exc.value


# ---------------------------------------------------------------------------
# the baseline: the gate opens
# ---------------------------------------------------------------------------

class TestTheGateCanOpen:
    """Without this, every refusal below is unfalsifiable."""

    def test_a_clean_checkpoint_with_an_approving_record_is_admitted(self):
        v = admit(
            clean_claims(),
            purpose="live_hardware",
            live_use_authorization=approving_record(),
            as_of=AFTER + timedelta(days=30),
        )
        assert v.admitted
        assert v.failed == ()
        assert {c.code for c in v.conditions} == {"A0", "A1", "A2", "A3", "A4", "A5", "A6"}

    def test_simulation_asks_nothing_of_the_checkpoint(self):
        """A simulation reported as a simulation needs no artifact to be valid."""
        v = admit(CheckpointClaims.absent(), purpose="simulation", as_of=AFTER)
        assert v.admitted
        # ...but every non-required condition is still *recorded* as having
        # failed, so the verdict shows what was not asked rather than hiding it.
        not_required = {c.code for c in v.conditions if not c.required}
        assert {"A1", "A2", "A3", "A4", "A5", "A6"} <= not_required
        assert not all(c.passed for c in v.conditions)

    def test_the_admission_record_is_content_hashed(self):
        a = admit(clean_claims(), purpose="research_offline", as_of=AFTER)
        b = admit(clean_claims(), purpose="research_offline", as_of=AFTER)
        c = admit(clean_claims(manifest_id="other"), purpose="research_offline", as_of=AFTER)
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

    def test_an_approving_review_does_not_relax_them(self):
        """The IRB boundary and the promotion boundary are unrelated."""
        v = admit(
            clean_claims(),
            purpose="patient_directed",
            live_use_authorization=approving_record(),
            as_of=AFTER + timedelta(days=30),
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
                live_use_authorization=approving_record(),
                invariants=ConsumerInvariants(promotion_eligible=True),
                as_of=AFTER + timedelta(days=30),
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
        v = admit(clean_claims(), purpose="research_offline", as_of=AFTER)
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

class TestA6LiveUseAuthorization:
    @pytest.mark.parametrize("purpose", sorted(LIVE_PURPOSES))
    def test_no_record_refuses_every_live_purpose(self, purpose):
        e = refusal(clean_claims(), purpose=purpose, auth=None)
        assert e.codes == ("A6",)
        assert "no LiveUseAuthorization" in str(e)

    @pytest.mark.parametrize(
        "purpose", [p for p in EXPORT_PURPOSES if p not in LIVE_PURPOSES]
    )
    def test_non_live_purposes_do_not_require_a_record(self, purpose):
        v = admit(clean_claims(), purpose=purpose, as_of=AFTER)
        assert v.admitted

    def test_the_gate_does_not_open_when_the_date_passes(self):
        """The whole point. Ten years of waiting admits nothing."""
        for as_of in (
            EARLIEST_CREDIBLE_REVIEW,
            EARLIEST_CREDIBLE_REVIEW + timedelta(days=1),
            EARLIEST_CREDIBLE_REVIEW + timedelta(days=3650),
        ):
            e = refusal(clean_claims(), auth=None, as_of=as_of)
            assert e.codes == ("A6",)
        assert "not the passing of" in str(e)

    @pytest.mark.parametrize(
        "outcome", ["pending", "scheduled", "deferred", "disapproved", "not_yet_reviewed"]
    )
    def test_a_non_approving_outcome_is_refused(self, outcome):
        rec = LiveUseAuthorization(
            review_body="UT Arlington IRB",
            reference="2026-0817",
            outcome=outcome,
            reviewed_on=AFTER,
            scope=("live_hardware",),
            recorded_by="pi",
        )
        e = refusal(clean_claims(), auth=rec)
        assert "A6" in e.codes
        assert outcome in str(e)

    def test_a_record_dated_in_the_future_is_refused(self):
        rec = approving_record(reviewed_on=AFTER + timedelta(days=365))
        e = refusal(clean_claims(), auth=rec, as_of=AFTER + timedelta(days=30))
        assert "A6" in e.codes
        assert "has not happened yet" in str(e)

    def test_an_expired_record_is_refused(self):
        rec = approving_record(expires_on=AFTER + timedelta(days=10))
        e = refusal(clean_claims(), auth=rec, as_of=AFTER + timedelta(days=90))
        assert "A6" in e.codes
        assert "expired" in str(e)

    def test_approval_of_one_purpose_is_not_approval_of_another(self):
        rec = approving_record(scope=("live_hardware",))
        e = refusal(clean_claims(), purpose="patient_directed", auth=rec)
        assert "A6" in e.codes
        assert "not in the record's scope" in str(e)
        # ...and the purpose it *does* cover is admitted.
        assert admit(
            clean_claims(),
            purpose="live_hardware",
            live_use_authorization=rec,
            as_of=AFTER + timedelta(days=30),
        ).admitted

    def test_an_approval_predating_the_review_cannot_be_constructed(self):
        with pytest.raises(AuthorizationInvalid) as exc:
            approving_record(reviewed_on=EARLIEST_CREDIBLE_REVIEW - timedelta(days=1))
        assert "could not have concluded before" in str(exc.value)
        assert "not a date on which anything becomes permitted" in str(exc.value)

    def test_a_non_approving_outcome_may_predate_the_review_floor(self):
        """The floor guards approvals, not the existence of earlier records."""
        rec = LiveUseAuthorization(
            review_body="UT Arlington IRB",
            reference="2026-0817",
            outcome="not_yet_reviewed",
            reviewed_on=date(2026, 8, 1),
            scope=("live_hardware",),
            recorded_by="pi",
        )
        assert rec.outcome == "not_yet_reviewed"

    @pytest.mark.parametrize(
        "kwargs, fragment",
        [
            ({"review_body": ""}, "reviewing body"),
            ({"reference": ""}, "reviewing body"),
            ({"recorded_by": ""}, "who recorded it"),
            ({"scope": ()}, "purposes it covers"),
            ({"scope": ("teleoperation",)}, "unknown purposes"),
            (
                {"outcome": "approved_with_conditions"},
                "no conditions listed",
            ),
        ],
    )
    def test_an_underspecified_record_cannot_be_constructed(self, kwargs, fragment):
        with pytest.raises(AuthorizationInvalid) as exc:
            approving_record(**kwargs)
        assert fragment in str(exc.value)

    def test_approved_with_conditions_admits_when_conditions_are_listed(self):
        rec = approving_record(
            outcome="approved_with_conditions",
            conditions=("no operator-free sessions", "log every pose"),
        )
        v = admit(
            clean_claims(),
            purpose="live_hardware",
            live_use_authorization=rec,
            as_of=AFTER + timedelta(days=30),
        )
        assert v.admitted
        assert v.authorization.conditions

    def test_the_record_is_content_hashed(self):
        assert approving_record().record_hash() == approving_record().record_hash()
        assert (
            approving_record().record_hash()
            != approving_record(reference="other").record_hash()
        )

    def test_round_trips_through_json(self):
        rec = approving_record(expires_on=AFTER + timedelta(days=365))
        again = LiveUseAuthorization.from_dict(json.loads(json.dumps(rec.canonical())))
        assert again.record_hash() == rec.record_hash()


# ---------------------------------------------------------------------------
# refusals compose, and name every failing condition
# ---------------------------------------------------------------------------

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
            as_of=AFTER,
            raise_on_refusal=False,
        )
        assert v.admitted is False
        assert {c.code for c in v.failed} >= {"A1", "A2", "A3", "A4", "A5", "A6"}
