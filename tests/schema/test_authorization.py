"""The R11 governance gate: what admits, what refuses, and with which reason.

The refusal R11 used to reject every prospective human TMS/tFUS protocol
unconditionally.  That was a constant, and a constant cannot discriminate: it
said the same thing to a project holding a current IRB approval and to one
holding nothing, so no observation could ever change it.  These tests pin the
replacement: a gate that **admits a validated declaration and refuses each
defect for its own reason**.

Nothing here authorises anything, and validation establishes nothing about the
world.  ``scwbd.schema.authorization`` records and checks a *claim* of
authorization; whether an IRB approval exists is a fact held by the people who
signed the protocol, and no software can check it.  The fixture record names a
fictional committee at a fictional university on purpose.
"""

from __future__ import annotations

import pytest

from scwbd.compiler import compile
from scwbd.schema.authorization import (
    ASafeAttribution,
    AuthorizationRecord,
    ConsentScope,
    EnrollmentScope,
    RegulatoryStatus,
    ResponsibleInvestigator,
    ValidityWindow,
    epoch_seconds,
    is_placeholder,
    no_authorization_verdict,
    validate_authorization,
)
from scwbd.schema.refusals import CompilerRefusal

from .refusals.fixtures import build_valid

#: The fixture record in ``tests/conftest.py`` is valid 2026-01-04..2027-01-04.
WITHIN_WINDOW_S = epoch_seconds("2026-08-05")
AFTER_WINDOW_S = epoch_seconds("2027-06-01")
BEFORE_WINDOW_S = epoch_seconds("2025-11-01")


def _validate(record, cls="tms", at=WITHIN_WINDOW_S, **kw):
    return validate_authorization(
        record, intervention_class=cls, at_time_s=at, **kw
    )


# ---------------------------------------------------------------------------
# the positive case: a valid record admits
# ---------------------------------------------------------------------------


class TestAValidRecordAdmits:
    def test_a_complete_in_date_in_scope_record_admits(self, authorization):
        verdict = _validate(authorization)
        assert verdict.admitted, verdict.reason()
        assert verdict.failures == ()

    def test_every_named_check_actually_ran(self, authorization):
        """An admitting verdict must be able to list what it checked."""
        verdict = _validate(authorization)
        assert set(verdict.checks_passed) == {
            "approval_identity",
            "responsible_investigator",
            "validity_window",
            "intervention_class_authorized",
            "consent_scope",
            "device_regulatory_status",
            "a_safe_traceable",
            "enrollment_declared",
        }

    def test_the_record_is_frozen_and_content_addressed(self, authorization):
        assert authorization.content_hash() == authorization.content_hash()
        with pytest.raises(Exception):
            authorization.approval_identifier = "IRB-2026-9999"  # type: ignore[misc]
        other = authorization.model_copy(update={"protocol_version": "3.3"})
        assert other.content_hash() != authorization.content_hash()

    def test_the_verdict_carries_the_declaration_notice(self, authorization):
        verdict = _validate(authorization)
        assert "DECLARATION, NOT VERIFICATION" in verdict.notice
        assert "no software can" in verdict.notice
        assert "DECLARATION, NOT VERIFICATION" in verdict.as_provenance()["notice"]

    def test_admission_sets_a_protocol_claim_scope(self, authorization):
        verdict = _validate(authorization)
        assert verdict.claim_scope == "protocol:EX-TMS-DLPFC-01@3.2"
        assert _validate(authorization, at=AFTER_WINDOW_S).claim_scope == "simulation_only"


# ---------------------------------------------------------------------------
# each invalidity refuses with its own code and its own reason
# ---------------------------------------------------------------------------


class TestEachInvalidityRefusesForItsOwnReason:
    def test_no_record_at_all_refuses_as_absent_not_as_forbidden(self):
        verdict = no_authorization_verdict("tms", at_time_s=WITHIN_WINDOW_S)
        assert not verdict.admitted
        assert verdict.failure_codes == ("AUTH_ABSENT",)
        # The remedy names a supplyable artifact rather than asserting a
        # belief about the world -- that is the whole correction.
        assert "AuthorizationRecord" in verdict.failures[0].remedy
        assert _validate(None).failure_codes == ("AUTH_ABSENT",)

    def test_an_expired_approval_refuses_as_expired(self, authorization):
        verdict = _validate(authorization, at=AFTER_WINDOW_S)
        assert not verdict.admitted
        assert "AUTH_EXPIRED" in verdict.failure_codes
        assert "expired" in verdict.reason()
        assert "continuing-review" in verdict.reason() or any(
            "continuing-review" in f.remedy for f in verdict.failures
        )

    def test_an_approval_that_has_not_taken_effect_refuses_separately(self, authorization):
        verdict = _validate(authorization, at=BEFORE_WINDOW_S)
        assert "AUTH_NOT_YET_VALID" in verdict.failure_codes
        assert "AUTH_EXPIRED" not in verdict.failure_codes

    def test_an_undated_request_is_not_assumed_current(self, authorization):
        verdict = _validate(authorization, at=None)
        assert not verdict.admitted
        assert "AUTH_TIME_UNDECLARED" in verdict.failure_codes

    def test_a_tms_record_does_not_authorise_tfus(self, authorization):
        """The load-bearing scope check: one class is not another."""
        verdict = _validate(authorization, cls="tfus")
        assert not verdict.admitted
        assert "AUTH_CLASS_NOT_AUTHORIZED" in verdict.failure_codes
        assert "AUTH_CONSENT_OUT_OF_SCOPE" in verdict.failure_codes
        assert "tfus" in verdict.reason()

    def test_consent_covering_data_but_not_intervention_refuses(
        self, make_authorization, authorization
    ):
        record = make_authorization(
            consent=authorization.consent.model_copy(
                update={"covers_prospective_intervention": False}
            )
        )
        verdict = _validate(record)
        assert not verdict.admitted
        assert "AUTH_CONSENT_OUT_OF_SCOPE" in verdict.failure_codes
        assert "not consent to be stimulated" in verdict.reason()

    def test_an_approval_wider_than_consent_refuses(self, make_authorization):
        record = make_authorization(authorized_intervention_classes=("tms", "tfus"))
        verdict = _validate(record)
        assert not verdict.admitted
        assert "AUTH_APPROVAL_EXCEEDS_CONSENT" in verdict.failure_codes

    def test_an_unresolved_withdrawal_refuses(self, make_authorization, authorization):
        record = make_authorization(
            consent=authorization.consent.model_copy(
                update={"withdrawal_status": "pending"}
            )
        )
        assert "AUTH_CONSENT_UNRESOLVED" in _validate(record).failure_codes

    def test_a_missing_device_regulatory_status_refuses(self, make_authorization):
        """IRB approval alone is not a universal regulatory description."""
        record = make_authorization(
            regulatory=RegulatoryStatus(
                jurisdiction="US",
                device_identifier="Example Model E8 figure-of-eight TMS coil",
                risk_determination="undeclared",
            )
        )
        verdict = _validate(record)
        assert not verdict.admitted
        assert "AUTH_DEVICE_STATUS_UNDECLARED" in verdict.failure_codes
        assert "fdaide" in verdict.reason() or "IRB approval alone" in verdict.reason()

    def test_a_significant_risk_device_without_an_ide_refuses(
        self, make_authorization, authorization
    ):
        record = make_authorization(
            regulatory=authorization.regulatory.model_copy(
                update={
                    "risk_determination": "significant_risk",
                    "ide_number": None,
                    "fda_approval_status": "pending",
                }
            )
        )
        verdict = _validate(record)
        codes = verdict.failure_codes
        assert codes.count("AUTH_REGULATORY_INCOMPLETE") == 2
        assert "IDE" in verdict.reason()

    def test_a_nonsignificant_risk_study_without_irb_concurrence_refuses(
        self, make_authorization, authorization
    ):
        record = make_authorization(
            regulatory=authorization.regulatory.model_copy(
                update={"irb_concurrence": False}
            )
        )
        assert "AUTH_REGULATORY_INCOMPLETE" in _validate(record).failure_codes

    @pytest.mark.parametrize(
        "value", ["", "   ", "TBD", "tbd", "n/a", "XXXX-1234", "placeholder", "TODO"]
    )
    def test_placeholder_fields_are_treated_as_missing(self, make_authorization, value):
        record = make_authorization(approval_identifier=value)
        verdict = _validate(record)
        assert not verdict.admitted
        assert set(verdict.failure_codes) & {
            "AUTH_FIELD_MISSING",
            "AUTH_FIELD_PLACEHOLDER",
        }

    def test_placeholder_detection_does_not_eat_real_identifiers(self):
        assert not is_placeholder("IRB-2026-0417")
        assert not is_placeholder("EX-TMS-DLPFC-01")
        assert is_placeholder(None)
        assert is_placeholder("  UNKNOWN ")

    def test_a_missing_protocol_version_refuses(self, make_authorization):
        record = make_authorization(protocol_version="")
        verdict = _validate(record)
        assert "AUTH_FIELD_MISSING" in verdict.failure_codes
        assert any(f.field == "protocol_version" for f in verdict.failures)

    def test_an_a_safe_inherited_from_a_generic_default_is_not_traceable(
        self, make_authorization, authorization
    ):
        record = make_authorization(
            a_safe=authorization.a_safe.model_copy(
                update={"derivation": "generic_default"}
            )
        )
        verdict = _validate(record)
        assert not verdict.admitted
        assert "AUTH_ASAFE_NOT_TRACEABLE" in verdict.failure_codes
        assert "generic default" in verdict.reason()

    def test_an_a_safe_attributed_to_another_protocol_version_refuses(
        self, make_authorization, authorization
    ):
        record = make_authorization(
            a_safe=authorization.a_safe.model_copy(
                update={"protocol_reference": "EX-TMS-DLPFC-01@2.0"}
            )
        )
        assert "AUTH_ASAFE_NOT_TRACEABLE" in _validate(record).failure_codes

    def test_an_a_safe_with_no_declared_axes_refuses(
        self, make_authorization, authorization
    ):
        record = make_authorization(
            a_safe=authorization.a_safe.model_copy(update={"constraint_axes": ()})
        )
        assert "AUTH_ASAFE_MISSING" in _validate(record).failure_codes

    def test_an_axis_the_protocol_does_not_bound_refuses(self, authorization):
        verdict = _validate(
            authorization, required_a_safe_axes=("tms.pulse_width_us",)
        )
        assert not verdict.admitted
        assert "AUTH_ASAFE_NOT_TRACEABLE" in verdict.failure_codes

    def test_operating_under_a_different_feasible_set_refuses(self, authorization):
        verdict = _validate(authorization, a_safe_id="some_other_a_safe")
        assert "AUTH_ASAFE_NOT_TRACEABLE" in verdict.failure_codes

    def test_an_a_safe_with_no_independent_validator_refuses(
        self, make_authorization, authorization
    ):
        record = make_authorization(
            a_safe=authorization.a_safe.model_copy(
                update={"independently_validated": False}
            )
        )
        assert "AUTH_ASAFE_NOT_TRACEABLE" in _validate(record).failure_codes

    def test_an_undeclared_enrollment_scope_refuses(self, make_authorization):
        record = make_authorization(enrollment=EnrollmentScope(declared_scope="some patients"))
        verdict = _validate(record)
        assert "AUTH_ENROLLMENT_UNDECLARED" in verdict.failure_codes

    def test_a_cohort_with_a_described_scope_is_enough(self, make_authorization):
        record = make_authorization(
            enrollment=EnrollmentScope(
                cohort_id="EX-TMS-DLPFC-01-cohortB",
                declared_scope="adults enrolled at site B under protocol v3.2",
            )
        )
        assert _validate(record).admitted

    def test_a_record_with_no_responsible_investigator_refuses(self, make_authorization):
        record = make_authorization(investigator=ResponsibleInvestigator(name="TBD"))
        assert "AUTH_INVESTIGATOR_UNDECLARED" in _validate(record).failure_codes

    def test_an_empty_record_is_not_admitted_merely_by_existing(self):
        """The point of the whole module: existence is not validity."""
        empty = AuthorizationRecord(
            id="empty",
            validity=ValidityWindow.between("2026-01-01", "2027-01-01"),
        )
        verdict = _validate(empty)
        assert not verdict.admitted
        assert len(verdict.failures) >= 8
        assert len(set(verdict.failure_codes)) >= 6

    def test_every_failure_names_a_field_and_a_reason(self, make_authorization):
        record = make_authorization(
            approval_identifier="TBD",
            a_safe=ASafeAttribution(),
            consent=ConsentScope(),
            regulatory=RegulatoryStatus(),
        )
        verdict = _validate(record)
        assert not verdict.admitted
        for failure in verdict.failures:
            assert failure.field
            assert failure.reason
            assert failure.remedy


# ---------------------------------------------------------------------------
# raising, and the shape of the refusal a consumer sees
# ---------------------------------------------------------------------------


class TestTheRefusalItRaises:
    def test_refusal_is_r11_and_carries_the_specific_failures(self, authorization):
        verdict = _validate(authorization, at=AFTER_WINDOW_S)
        with pytest.raises(CompilerRefusal) as excinfo:
            verdict.raise_if_refused()
        exc = excinfo.value
        assert exc.code == "R11"
        codes = [f["code"] for f in exc.evidence["authorization_failures"]]
        assert "AUTH_EXPIRED" in codes
        assert exc.evidence["authorization_record_hash"] == authorization.content_hash()

    def test_an_admitted_verdict_does_not_raise(self, authorization):
        assert _validate(authorization).raise_if_refused() is not None

    def test_an_unknown_intervention_class_is_a_programming_error(self, authorization):
        with pytest.raises(ValueError, match="unknown intervention class"):
            _validate(authorization, cls="telepathy")


# ---------------------------------------------------------------------------
# the compiler gate: R11 refuses unless a validated record admits
# ---------------------------------------------------------------------------


#: The fixture schema's impulse source declares this feasible set.
_FIXTURE_A_SAFE_ID = "A_safe_simulated_impulse"
_FIXTURE_A_SAFE_AXES = ("duty_cycle", "peak_current_density", "pulse_width")


def _aligned(record):
    """The fixture record, with its A_safe attribution pointing at the schema's.

    Alignment is the *workflow*: the protocol declares the limits, and the
    source card is compiled against those limits.  A record whose A_safe names
    a different feasible set is refused, which
    ``test_a_record_naming_another_feasible_set_refuses`` proves.
    """
    return record.model_copy(
        update={
            "a_safe": record.a_safe.model_copy(
                update={
                    "a_safe_id": _FIXTURE_A_SAFE_ID,
                    "constraint_axes": _FIXTURE_A_SAFE_AXES,
                }
            )
        }
    )


def _prospective_schema(claim_updates=None, intervention_updates=None):
    """The valid fixture, with its impulse source turned into prospective TMS."""
    schema, claim = build_valid()
    source = schema.source("impulse_sim_v1")
    updates = {"modality": "tms", "is_prospective_human": True}
    updates.update(intervention_updates or {})
    intervention = source.intervention.model_copy(update=updates)
    sources = [
        s.model_copy(update={"intervention": intervention}) if s.id == "impulse_sim_v1" else s
        for s in schema.sources
    ]
    schema = schema.model_copy(update={"sources": sources})
    if claim_updates:
        claim = claim.model_copy(update=claim_updates)
    return schema, claim


class TestTheCompilerGate:
    def test_prospective_human_tms_without_a_record_still_refuses(self):
        schema, claim = _prospective_schema()
        with pytest.raises(CompilerRefusal) as excinfo:
            compile(schema, claim=claim)
        exc = excinfo.value
        assert exc.code == "R11"
        assert exc.evidence["authorization_failure_codes"] == ["AUTH_ABSENT"]

    def test_prospective_human_tms_with_a_valid_record_compiles(self, authorization):
        schema, claim = _prospective_schema(
            {"authorization": _aligned(authorization), "request_time_s": WITHIN_WINDOW_S}
        )
        model = compile(schema, claim=claim)
        assert model.is_protocol_bound

    def test_a_record_naming_another_feasible_set_refuses(self, authorization):
        """The record must be attached to the limits actually in force."""
        schema, claim = _prospective_schema(
            {"authorization": authorization, "request_time_s": WITHIN_WINDOW_S}
        )
        with pytest.raises(CompilerRefusal) as excinfo:
            compile(schema, claim=claim)
        codes = excinfo.value.evidence["authorization_failure_codes"]
        assert set(codes) == {"AUTH_ASAFE_NOT_TRACEABLE"}

    def test_an_expired_record_refuses_at_compile_time(self, authorization):
        schema, claim = _prospective_schema(
            {"authorization": _aligned(authorization), "request_time_s": AFTER_WINDOW_S}
        )
        with pytest.raises(CompilerRefusal) as excinfo:
            compile(schema, claim=claim)
        assert "AUTH_EXPIRED" in excinfo.value.evidence["authorization_failure_codes"]

    def test_a_tms_record_does_not_admit_a_tfus_source(self, authorization):
        schema, claim = _prospective_schema(
            {"authorization": _aligned(authorization), "request_time_s": WITHIN_WINDOW_S},
            {"modality": "tfus"},
        )
        with pytest.raises(CompilerRefusal) as excinfo:
            compile(schema, claim=claim)
        codes = excinfo.value.evidence["authorization_failure_codes"]
        assert "AUTH_CLASS_NOT_AUTHORIZED" in codes

    def test_a_prospective_claim_with_no_prospective_source_refuses(self, authorization):
        schema, claim = build_valid()
        claim = claim.model_copy(
            update={
                "prospective_human": True,
                "authorization": authorization,
                "request_time_s": WITHIN_WINDOW_S,
            }
        )
        with pytest.raises(CompilerRefusal) as excinfo:
            compile(schema, claim=claim)
        assert "no intervention class to authorise" in excinfo.value.detail


class TestProvenanceRecordsTheAuthorization:
    def test_the_artifact_records_the_record_and_changes_its_claim(self, authorization):
        schema, claim = _prospective_schema(
            {"authorization": _aligned(authorization), "request_time_s": WITHIN_WINDOW_S}
        )
        model = compile(schema, claim=claim)
        prov = model.provenance

        assert prov.claim_scope == "protocol:EX-TMS-DLPFC-01@3.2"
        assert prov.is_protocol_bound
        assert prov.recorded_claim().endswith("@protocol:EX-TMS-DLPFC-01@3.2")
        assert _aligned(authorization).content_hash() in prov.authorization_hashes
        entry = prov.authorizations[0]
        assert entry["admitted"] is True
        assert entry["approving_body"].startswith("Example University")
        assert entry["approval_identifier"] == "IRB-2026-0417"
        assert entry["intervention_class"] == "tms"
        assert "DECLARATION, NOT VERIFICATION" in entry["notice"]

    def test_the_authorization_is_visible_in_the_summary_never_silent(self, authorization):
        schema, claim = _prospective_schema(
            {"authorization": _aligned(authorization), "request_time_s": WITHIN_WINDOW_S}
        )
        model = compile(schema, claim=claim)
        assert "protocol:EX-TMS-DLPFC-01@3.2" in model.provenance.summary()
        assert "IRB-2026-0417" in model.provenance.summary()

    def test_the_serialized_provenance_carries_it(self, authorization):
        schema, claim = _prospective_schema(
            {"authorization": _aligned(authorization), "request_time_s": WITHIN_WINDOW_S}
        )
        payload = compile(schema, claim=claim).provenance.as_dict()
        assert payload["claim_scope"] == "protocol:EX-TMS-DLPFC-01@3.2"
        assert (
            payload["authorizations"][0]["record_hash"]
            == _aligned(authorization).content_hash()
        )

    def test_an_unauthorized_artifact_is_a_different_artifact(self, authorization):
        """The change must be observable, or it is not a change."""
        schema, plain = build_valid()
        authorized = plain.model_copy(
            update={"authorization": authorization, "request_time_s": WITHIN_WINDOW_S}
        )
        a = compile(schema, claim=plain)
        b = compile(schema, claim=authorized)
        assert a.content_hash() != b.content_hash()
        assert a.provenance.as_dict() != b.provenance.as_dict()

    def test_a_record_attached_to_a_non_prospective_build_authorizes_nothing(
        self, authorization
    ):
        schema, claim = build_valid()
        claim = claim.model_copy(
            update={"authorization": authorization, "request_time_s": WITHIN_WINDOW_S}
        )
        model = compile(schema, claim=claim)
        assert model.claim_scope == "simulation_only"
        assert not model.is_protocol_bound
        assert model.provenance.authorizations[0]["admitted"] is False
        assert "nothing was authorized" in model.provenance.authorizations[0]["note"]


# ---------------------------------------------------------------------------
# time handling
# ---------------------------------------------------------------------------


class TestValidityWindows:
    def test_dates_are_read_as_utc_not_as_local_time(self):
        assert epoch_seconds("2026-01-01") == 1767225600.0
        assert epoch_seconds("2026-01-01T00:00:00+00:00") == 1767225600.0

    def test_an_inverted_window_is_not_constructible(self):
        with pytest.raises(ValueError, match="precedes start"):
            ValidityWindow.between("2027-01-01", "2026-01-01")

    def test_the_window_is_half_open(self):
        window = ValidityWindow.between("2026-01-01", "2027-01-01")
        assert window.contains(epoch_seconds("2026-01-01"))
        assert not window.contains(epoch_seconds("2027-01-01"))
