"""Two refusals and four labels, each fired, each paired with its opposite.

``ARCHITECTURE.md`` Sec. 7a: **ship the artifact and label it; never refuse to
produce it.** A precondition blocks only if it changes what a number *means*.

An earlier version of this module refused on the control arm, on synthetic
anatomy and on ``COULD_NOT_RUN`` gates. That was a product-safety posture
imported into a research repository, and it would have withheld the only
checkpoint we have from the only people who can do anything with it. Those
three are now **labels**: loud, carried in the provenance, and non-blocking.

The discipline, per condition and per label:

1. make it fire, and assert the message names it by code;
2. make it **not** fire on an input differing *only* in that respect.

(2) is the part usually missing. A label that is always set is as useless as a
gate that never opens; a suite that only exercises one side cannot tell them
apart.
"""

from __future__ import annotations

import pytest

from scwbd.runtime.admission import (
    CONSUMER_STANDING_INVARIANTS,
    EXPORT_PURPOSES,
    CheckpointClaims,
    CheckpointRefused,
    ConsumerInvariants,
    ConsumerInvariantViolation,
    admit,
)


def clean_claims(**overrides) -> CheckpointClaims:
    """An artifact with every label clean. Override one to flag one."""
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
        manifest_readable=True,
    )
    base.update(overrides)
    return CheckpointClaims(**base)


def verdict(claims, purpose="live_hardware", **kw):
    return admit(claims, purpose=purpose, raise_on_refusal=False, **kw)


# ---------------------------------------------------------------------------
# the posture itself
# ---------------------------------------------------------------------------

class TestTheArtifactShipsAndIsLabelled:
    """The rule the module exists to implement, stated as tests."""

    @pytest.mark.parametrize("purpose", EXPORT_PURPOSES)
    def test_the_worst_artifact_we_hold_still_loads_for_every_purpose(self, purpose):
        """Control arm, synthetic anatomy, COULD_NOT_RUN gates, no weights."""
        run1 = CheckpointClaims.from_manifest(
            {
                "id": "scwbd-001-beta",
                "arm": "equal_capacity_generic_operator_control",
                "anatomy": {"is_biological": False, "provenance": "synthetic_fallback"},
                "gates": {f"G{i}": "COULD_NOT_RUN" for i in range(1, 6)},
                "weights_trained": False,
            }
        )
        v = admit(run1, purpose=purpose)
        assert v.admitted, "the artifact must ship; refusals belong on claims"
        # ...and every one of its problems is on the record.
        assert set(v.label_codes) == {"L1", "L2", "L3", "L4"}
        assert not v.is_clean

    def test_every_flagged_label_says_what_it_changes_about_the_numbers(self):
        v = verdict(
            clean_claims(
                is_control_arm=True, anatomy_is_biological=False,
                gates={"G1": "COULD_NOT_RUN"}, weights_trained=False,
            )
        )
        for label in v.flagged:
            assert label.consequence, f"{label.code} flags without a consequence"
        text = "\n".join(v.warnings())
        assert "measure the control" in text
        assert "not anatomy" in text
        assert "not a pending PASS" in text
        assert "fitted to data" in text

    def test_a_clean_artifact_produces_no_banner(self):
        v = verdict(clean_claims())
        assert v.is_clean
        assert v.flagged == ()
        assert v.banner() == ""
        assert v.warnings() == ()

    def test_the_banner_is_loud_when_there_is_something_to_say(self):
        v = verdict(clean_claims(is_control_arm=True))
        assert "change what its numbers mean" in v.banner()
        assert "[L1]" in v.banner()


# ---------------------------------------------------------------------------
# A0, A1 -- the only two refusals
# ---------------------------------------------------------------------------

class TestA0StandingInvariants:
    @pytest.mark.parametrize("name", sorted(CONSUMER_STANDING_INVARIANTS))
    def test_widening_any_invariant_is_refused(self, name):
        with pytest.raises(ConsumerInvariantViolation) as exc:
            ConsumerInvariants(**{name: True})
        assert name in str(exc.value)

    def test_the_default_is_all_false(self):
        assert ConsumerInvariants().as_dict() == dict(CONSUMER_STANDING_INVARIANTS)

    def test_they_survive_an_admitted_patient_directed_load(self):
        v = admit(clean_claims(), purpose="patient_directed")
        assert v.admitted
        assert v.invariants.as_dict() == {
            "sim2real_ready": False,
            "promotion_eligible": False,
            "robot_command_authority": False,
        }


#: A checkpoint that exists on disk but carries no readable sidecar.
UNLABELLABLE = CheckpointClaims(checkpoint_present=True, manifest_readable=False)


class TestA1AnExistingCheckpointMustBeLabellable:
    def test_a_present_checkpoint_with_no_readable_manifest_refuses(self):
        with pytest.raises(CheckpointRefused) as exc:
            admit(UNLABELLABLE, purpose="research_offline")
        assert exc.value.codes == ("A1",)
        assert "every label below would be a guess" in str(exc.value)

    @pytest.mark.parametrize("purpose", EXPORT_PURPOSES)
    def test_it_refuses_for_every_purpose_including_simulation(self, purpose):
        """Unlike the labels, this one is correctness and binds everywhere."""
        with pytest.raises(CheckpointRefused) as exc:
            admit(UNLABELLABLE, purpose=purpose)
        assert exc.value.codes == ("A1",)

    def test_no_checkpoint_at_all_is_admitted_not_refused(self):
        """The analytic backend has no claims to mislabel; L4 says what it is.

        Refusing here would break the load path's deliberate design of not
        failing when there is no trained artifact -- which would push consumers
        toward building their own uncontrolled fallback, the exact outcome that
        design exists to prevent.
        """
        v = admit(CheckpointClaims.absent(), purpose="patient_directed")
        assert v.admitted
        assert v.claims.checkpoint_present is False
        assert "L4" in v.label_codes

    def test_a_manifest_that_says_terrible_things_still_passes_A1(self):
        """A1 is about readability, never content. This is the whole point."""
        awful = CheckpointClaims.from_manifest(
            {
                "id": "the-worst",
                "arm": "control",
                "anatomy": {"is_biological": False},
                "gates": {"G1": "FAIL"},
            }
        )
        assert awful.manifest_readable is True
        v = admit(awful, purpose="live_hardware")
        assert v.admitted
        assert set(v.label_codes) == {"L1", "L2", "L3", "L4"}


# ---------------------------------------------------------------------------
# L1..L4 -- the labels
# ---------------------------------------------------------------------------

def label(v, code):
    return next(x for x in v.labels if x.code == code)


class TestL1AblationArm:
    def test_the_control_arm_is_flagged_and_still_loads(self):
        v = verdict(clean_claims(is_control_arm=True, control_arm_of="body.tex 11.4"))
        assert v.admitted
        assert "L1" in v.label_codes
        assert "control arm" in label(v, "L1").detail

    def test_an_artifact_that_does_not_say_is_labelled_the_control_arm(self):
        """Run 1 was the control arm while claiming otherwise. Silence is not clean."""
        claims = CheckpointClaims.from_manifest({"id": "unsaid"})
        assert claims.is_control_arm is True
        assert "L1" in verdict(claims).label_codes

    def test_a_declared_treatment_arm_is_clean_on_L1(self):
        claims = CheckpointClaims.from_manifest(
            {"id": "run2", "is_control_arm": False, "arm": "treatment"}
        )
        assert label(verdict(claims), "L1").clean
        assert label(verdict(claims), "L1").consequence == ""


class TestL2AnatomyProvenance:
    def test_synthetic_anatomy_is_flagged_and_still_loads(self):
        v = verdict(
            clean_claims(anatomy_is_biological=False,
                         anatomy_provenance="synthetic_fallback")
        )
        assert v.admitted
        assert "L2" in v.label_codes
        assert "synthetic_fallback" in label(v, "L2").detail
        assert "not anatomy" in label(v, "L2").consequence

    def test_an_unstated_anatomy_record_is_flagged_not_assumed(self):
        claims = CheckpointClaims.from_manifest({"id": "x", "is_control_arm": False})
        assert claims.anatomy_is_biological is False
        assert "L2" in verdict(claims).label_codes

    def test_biological_anatomy_is_clean_on_L2(self):
        assert label(verdict(clean_claims()), "L2").clean


class TestL3ClaimGates:
    def test_could_not_run_is_flagged_and_named(self):
        v = verdict(clean_claims(gates={"G1": "COULD_NOT_RUN", "G2": "PASS"}))
        assert v.admitted
        assert "COULD_NOT_RUN" in label(v, "L3").detail
        assert "G1" in label(v, "L3").detail

    def test_a_failing_gate_is_flagged_and_named(self):
        v = verdict(clean_claims(gates={"G1": "PASS", "G2": "FAIL"}))
        assert "FAIL" in label(v, "L3").detail and "G2" in label(v, "L3").detail

    def test_no_gate_results_at_all_is_flagged(self):
        """Resolves the old A4 gap: nothing needs writing into a sidecar."""
        v = verdict(clean_claims(gates={}))
        assert v.admitted
        assert "no gate statuses recorded" in label(v, "L3").detail

    def test_all_passing_gates_are_clean_on_L3(self):
        assert label(verdict(clean_claims()), "L3").clean


class TestL4Weights:
    def test_an_analytic_backend_is_flagged(self):
        v = verdict(clean_claims(weights_trained=False))
        assert v.admitted
        assert "L4" in v.label_codes
        assert "fitted to data" in label(v, "L4").consequence

    def test_trained_weights_are_clean_on_L4(self):
        assert label(verdict(clean_claims()), "L4").clean


# ---------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------

class TestTheVerdictRecord:
    def test_it_is_content_hashed(self):
        a = admit(clean_claims(), purpose="research_offline")
        b = admit(clean_claims(), purpose="research_offline")
        c = admit(clean_claims(manifest_id="other"), purpose="research_offline")
        assert a.content_hash() == b.content_hash()
        assert a.content_hash() != c.content_hash()

    def test_the_hash_changes_when_a_label_changes(self):
        """A label that did not affect the record would be decoration."""
        a = admit(clean_claims(), purpose="research_offline")
        b = admit(clean_claims(is_control_arm=True), purpose="research_offline")
        assert a.content_hash() != b.content_hash()

    def test_canonical_carries_conditions_and_labels_separately(self):
        c = admit(clean_claims(is_control_arm=True), purpose="simulation").canonical()
        assert [x["code"] for x in c["conditions"]] == ["A0", "A1"]
        assert [x["code"] for x in c["labels"]] == ["L1", "L2", "L3", "L4"]
        assert c["flagged"] == ["L1"]

    def test_an_unknown_purpose_is_refused_rather_than_defaulted(self):
        with pytest.raises(ValueError) as exc:
            admit(clean_claims(), purpose="whatever")
        assert "unknown export purpose" in str(exc.value)

    def test_the_non_raising_form_reports_without_admitting(self):
        v = admit(UNLABELLABLE, purpose="live_hardware", raise_on_refusal=False)
        assert v.admitted is False
        assert {c.code for c in v.failed} == {"A1"}


class TestNoLiveApplicationGateRemains:
    """The delegation target was removed from the repository; nothing replaced it."""

    def test_the_module_exposes_no_live_application_surface(self):
        import scwbd.runtime.admission as adm

        for gone in (
            "LIVE_PURPOSES", "MODE_OF_PURPOSE", "LiveUseAuthorization",
            "authorize_live_application", "EARLIEST_CREDIBLE_REVIEW",
        ):
            assert not hasattr(adm, gone), f"{gone} still present"

    def test_admit_takes_no_review_or_authorization_argument(self):
        import inspect

        params = set(inspect.signature(admit).parameters)
        assert params == {
            "claims", "purpose", "invariants", "designation", "raise_on_refusal"
        }
