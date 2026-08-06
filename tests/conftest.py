"""Shared governance fixtures: a *fictional* authorization declaration.

The record built here is a test fixture.  It names no real committee, no real
protocol and no real person, and it authorises nothing: it exists so that the
R11 governance gate can be shown admitting a well-formed declaration and
refusing a defective one, each for its own specific reason.

Everything about the values is deliberately obvious fiction ("Example
University"), because a fixture that looked like a real approval would be the
first thing this repository is supposed to refuse.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

from scwbd.schema.authorization import (
    ASafeAttribution,
    AuthorizationRecord,
    ConsentScope,
    EnrollmentScope,
    RegulatoryStatus,
    ResponsibleInvestigator,
    ValidityWindow,
    epoch_seconds,
)

#: Every A_safe axis ``scwbd.runtime.targeting`` checks, plus the two the
#: intervention tests use.  A protocol that authorises operating "within
#: limits" has to declare limits on the axes actually checked.
PROTOCOL_AXES: tuple[str, ...] = (
    "protocol.session_duration_s",
    "tms.coil_scalp_distance_mm",
    "tms.frequency_hz",
    "tms.intertrain_interval_s",
    "tms.peak_efield_v_per_m",
    "tms.pulses_per_session",
)

#: A time inside the fixture's validity window.
WITHIN_WINDOW_S: float = epoch_seconds("2026-08-05")
#: A time after it expires.
AFTER_WINDOW_S: float = epoch_seconds("2027-06-01")
#: A time before it takes effect.
BEFORE_WINDOW_S: float = epoch_seconds("2025-11-01")


def build_authorization_record(**overrides: Any) -> AuthorizationRecord:
    """A complete, in-date, internally consistent *fictional* TMS declaration."""
    fields: dict[str, Any] = {
        "id": "auth_example_tms_2026",
        "approving_body": "Example University Institutional Review Board",
        "approving_body_kind": "irb",
        "approval_identifier": "IRB-2026-0417",
        "approval_date_label": "2026-01-04",
        "protocol_id": "EX-TMS-DLPFC-01",
        "protocol_version": "3.2",
        "protocol_title": "Offline comparison of candidate DLPFC coil poses",
        "authorized_intervention_classes": ("tms",),
        "consent": ConsentScope(
            document_id="ICF-EX-TMS-DLPFC-01",
            document_version="3.2",
            scope_statement=(
                "Participants consented to receive repetitive transcranial "
                "magnetic stimulation of left DLPFC within the protocol's "
                "declared exposure limits, and to modelling of their imaging "
                "data for coil-pose comparison."
            ),
            covered_intervention_classes=("tms",),
            covered_procedures=("mri", "eeg", "rtms"),
            covers_prospective_intervention=True,
            permits_data_reuse=True,
            withdrawal_status="none_pending",
        ),
        "enrollment": EnrollmentScope(
            cohort_id="EX-TMS-DLPFC-01-cohortA",
            participant_ids=("EX-P001", "EX-P002"),
            declared_scope=(
                "adults aged 22-65 enrolled at the Example University site "
                "under protocol version 3.2"
            ),
            population="adults with treatment-resistant depression",
            max_participants=24,
        ),
        "regulatory": RegulatoryStatus(
            jurisdiction="US",
            device_identifier="Example Model E8 figure-of-eight TMS coil",
            risk_determination="nonsignificant_risk",
            determined_by="sponsor (Example University) with IRB concurrence",
            ide_number=None,
            fda_approval_status="not_required",
            irb_concurrence=True,
            marketing_authorization="K000000 (fictional clearance reference)",
        ),
        "investigator": ResponsibleInvestigator(
            name="A. Example",
            role="principal investigator",
            institution="Example University",
            contact="pi@example.edu",
        ),
        "a_safe": ASafeAttribution(
            a_safe_id="EX-TMS-DLPFC-01-asafe",
            source="protocol EX-TMS-DLPFC-01 v3.2 appendix C, table C.1",
            protocol_reference="EX-TMS-DLPFC-01@3.2",
            derivation="declared_in_protocol",
            constraint_axes=PROTOCOL_AXES,
            independently_validated=True,
            validator="Example University clinical physics group",
        ),
        "validity": ValidityWindow.between("2026-01-04", "2027-01-04"),
        "declared_by": "test fixture",
        "declared_on_s": epoch_seconds("2026-01-05"),
    }
    fields.update(overrides)
    return AuthorizationRecord(**fields)


@pytest.fixture
def make_authorization() -> Callable[..., AuthorizationRecord]:
    """Factory for the fixture record; keyword arguments replace top-level fields."""
    return build_authorization_record


@pytest.fixture
def authorization(make_authorization) -> AuthorizationRecord:
    """The valid, in-date, TMS-scoped declaration."""
    return make_authorization()
