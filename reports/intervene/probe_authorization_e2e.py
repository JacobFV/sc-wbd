"""End-to-end probe: does a valid AuthorizationRecord actually admit a
prospective human TMS plan, or does something downstream refuse regardless?

Run from the faraday worktree with PYTHONPATH set to it.
"""
from __future__ import annotations

import json
import sys
import traceback

from scwbd.schema.authorization import (
    ASafeAttribution,
    AuthorizationRecord,
    ConsentScope,
    EnrollmentScope,
    RegulatoryStatus,
    ResponsibleInvestigator,
    ValidityWindow,
    epoch_seconds,
    validate_authorization,
)
from scwbd.schema.examples.three_region import (
    build_three_region_schema,
    build_three_region_claim,
)
from scwbd.compiler.compile import compile as compile_schema
from scwbd.schema.refusals import CompilerRefusal

AXES = (
    "protocol.session_duration_s",
    "tms.coil_scalp_distance_mm",
    "tms.frequency_hz",
    "tms.intertrain_interval_s",
    "tms.peak_efield_v_per_m",
    "tms.pulses_per_session",
)
AT = epoch_seconds("2026-08-05")


def record(**over) -> AuthorizationRecord:
    f = dict(
        id="auth_example_tms_2026",
        approving_body="Example University Institutional Review Board",
        approving_body_kind="irb",
        approval_identifier="IRB-2026-0417",
        approval_date_label="2026-01-04",
        protocol_id="EX-TMS-DLPFC-01",
        protocol_version="3.2",
        protocol_title="Prospective comparison of DLPFC coil poses",
        authorized_intervention_classes=("tms",),
        consent=ConsentScope(
            document_id="ICF-EX-TMS-DLPFC-01",
            document_version="3.2",
            scope_statement="Participants consented to rTMS of left DLPFC within declared limits.",
            covered_intervention_classes=("tms",),
            covered_procedures=("mri", "eeg", "rtms"),
            covers_prospective_intervention=True,
            permits_data_reuse=True,
            withdrawal_status="none_pending",
        ),
        enrollment=EnrollmentScope(
            cohort_id="EX-TMS-DLPFC-01-cohortA",
            participant_ids=("EX-P001", "EX-P002"),
            declared_scope="adults 22-65 at Example University under protocol v3.2",
            population="adults with treatment-resistant depression",
            max_participants=24,
        ),
        regulatory=RegulatoryStatus(
            jurisdiction="US",
            device_identifier="Example Model E8 figure-of-eight TMS coil",
            risk_determination="nonsignificant_risk",
            determined_by="sponsor with IRB concurrence",
            ide_number=None,
            fda_approval_status="not_required",
            irb_concurrence=True,
            marketing_authorization="K000000",
        ),
        investigator=ResponsibleInvestigator(
            name="A. Example",
            role="principal investigator",
            institution="Example University",
            contact="pi@example.edu",
        ),
        a_safe=ASafeAttribution(
            a_safe_id="EX-TMS-DLPFC-01-asafe",
            source="protocol EX-TMS-DLPFC-01 v3.2 appendix C",
            protocol_reference="EX-TMS-DLPFC-01@3.2",
            derivation="declared_in_protocol",
            constraint_axes=AXES,
            independently_validated=True,
            validator="Example University clinical physics group",
        ),
        validity=ValidityWindow.between("2026-01-04", "2027-01-04"),
        declared_by="faraday probe",
        declared_on_s=epoch_seconds("2026-01-05"),
    )
    f.update(over)
    return AuthorizationRecord(**f)


def out(k, v):
    print(f"[{k}] {v}")


# ---------------------------------------------------------------------------
print("=" * 72)
print("PROBE 1: validate_authorization on a complete, in-date, in-scope record")
print("=" * 72)
rec = record()
v = validate_authorization(
    rec,
    intervention_class="tms",
    at_time_s=AT,
    a_safe_id="EX-TMS-DLPFC-01-asafe",
    required_a_safe_axes=AXES,
    what="probe",
)
out("admitted", v.admitted)
out("claim_scope", v.claim_scope)
out("failures", [f.code for f in v.failures])
out("reason", v.reason())

# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("PROBE 2: AuthorizationGate.admit -- prospective human TMS proposal")
print("=" * 72)
from scwbd.intervene.safety import (  # noqa: E402
    AuthorizationGate,
    AuthorizedRequest,
    FeasibleSet,
    ProposedIntervention,
)

gate = AuthorizationGate(FeasibleSet(), a_safe_id="EX-TMS-DLPFC-01-asafe")
prop = ProposedIntervention(
    label="dlpfc_pose_a",
    modality="tms",
    exposure={
        "tms.peak_efield_v_per_m": 95.0,
        "tms.pulses_per_session": 600.0,
        "tms.coil_scalp_distance_mm": 4.0,
    },
    pose_certified=True,
    reversible=True,
)
req = AuthorizedRequest(record=rec, intervention_class="tms", at_time_s=AT)
try:
    adm = gate.admit(prop, req)
    out("ADMITTED", True)
    out("claim_scope", adm.claim_scope)
    out("provenance", json.dumps(adm.provenance(), indent=2, default=str)[:900])
except Exception as e:  # noqa: BLE001
    out("REFUSED", f"{type(e).__name__}: {e}")

# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("PROBE 3: full compiler on a schema declaring PROSPECTIVE HUMAN TMS")
print("=" * 72)
schema = build_three_region_schema()
base_claim = build_three_region_claim()

# turn the impulse source card into a prospective human TMS card
sources = list(schema.sources)
for i, s in enumerate(sources):
    if s.intervention is not None:
        iv = s.intervention
        new_safe = iv.a_safe.model_copy(
            update={
                "id": "EX-TMS-DLPFC-01-asafe",
                "constraints": {
                    "tms.peak_efield_v_per_m": (0.0, 130.0),
                    "tms.pulses_per_session": (0.0, 3000.0),
                    "tms.coil_scalp_distance_mm": (0.0, 30.0),
                },
                "constraint_units": {
                    "tms.peak_efield_v_per_m": iv.a_safe.constraint_units["peak_current_density"],
                    "tms.pulses_per_session": iv.a_safe.constraint_units["pulse_width"],
                    "tms.coil_scalp_distance_mm": iv.a_safe.constraint_units["duty_cycle"],
                },
            }
        )
        new_iv = iv.model_copy(
            update={
                "modality": "tms",
                "is_prospective_human": True,
                "a_safe": new_safe,
                "dose_independently_calibrated": True,
                "ethics_review": None,
            }
        )
        sources[i] = s.model_copy(update={"intervention": new_iv})
        out("rewrote source", s.id)
schema = schema.model_copy(update={"sources": sources})

for label, claim in [
    (
        "NO record (should refuse R11)",
        base_claim.model_copy(
            update={"prospective_human": True, "optimizes_intervention": True,
                    "request_time_s": AT, "authorization": None}
        ),
    ),
    (
        "VALID record (should COMPILE)",
        base_claim.model_copy(
            update={"prospective_human": True, "optimizes_intervention": True,
                    "request_time_s": AT, "authorization": rec}
        ),
    ),
]:
    print(f"\n--- {label} ---")
    try:
        cm = compile_schema(schema, claim=claim)
        out("COMPILED", True)
        out("claim_scope", cm.provenance.claim_scope)
        out("effective_claim_class", cm.provenance.effective_claim_class)
        out("authorizations", json.dumps(
            [dict(a) for a in cm.provenance.authorizations], indent=2, default=str)[:1200])
    except CompilerRefusal as e:
        out("REFUSED", f"{e.code}: {getattr(e, 'detail', '')[:400]}")
    except Exception as e:  # noqa: BLE001
        out("ERROR", f"{type(e).__name__}: {e}")
        traceback.print_exc(limit=3, file=sys.stdout)

# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("PROBE 4: downstream -- runtime ModelProvenance(prospective_human=True)")
print("=" * 72)
from scwbd.runtime.provenance import ModelProvenance  # noqa: E402

try:
    p = ModelProvenance(prospective_human=True)
    out("CONSTRUCTED", True)
except Exception as e:  # noqa: BLE001
    out("REFUSED", f"{type(e).__name__}: {e}")

print()
print("=" * 72)
print("PROBE 5: downstream -- bench D10 standing refusal")
print("=" * 72)
from scwbd.bench.leakage import audit_tms_tfus_decision_claim  # noqa: E402

rep = audit_tms_tfus_decision_claim()
out("status", rep.status)
out("subcheck", rep.subchecks[0].reason[:300] if rep.subchecks else "")
