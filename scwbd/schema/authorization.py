"""``AuthorizationRecord``: governance declared, recorded, and carried.

Why this module exists
----------------------
Refusal ``R11`` used to reject every prospective human TMS/tFUS protocol
*unconditionally*, on a hard-coded belief that no ethics approval, consent,
participants or device clearance existed anywhere in the world.  That is not a
safety property.  A constant cannot discriminate: it says the same thing to a
project with a current IRB approval and to one with nothing at all, so it
carries no information and cannot be wrong in a way anybody could observe.

The thesis's actual design is that governance is **declared, recorded, and
carried in provenance**: Appendix B makes governance a required source-card
field (\\(G_k\\)), and Table ``tab:compiler-refusals`` says an override
"changes the claim carried by the resulting artifact and remains visible in its
provenance".  ``thesis_contract.tex`` Sec. 0.5 adds the regulatory shape:
"the sponsor and IRB must document significant-risk versus nonsignificant-risk
status; a significant-risk device investigation requires FDA and IRB approval,
while an NSR study follows the abbreviated IDE requirements after IRB
concurrence \\citep{fdaide}. IRB approval alone is therefore not treated as a
universal regulatory description."

So ``R11`` now refuses **unless** a validated :class:`AuthorizationRecord` is
present for the requested intervention class at the requested time -- and
continues to refuse for every other reason it ever did.

WHAT VALIDATION DOES NOT ESTABLISH -- READ THIS
-----------------------------------------------
:func:`validate_authorization` checks that a **declaration** is well formed,
complete, in date, internally consistent, and in scope for what is being asked.

It does **not** contact an IRB, an REC, an institution, a sponsor, or a
regulator.  It cannot confirm that the approval exists, that it is still in
force, that the protocol version quoted is the approved one, that the
participants named consented, or that the device status was determined by
anyone entitled to determine it.  **No software can do that.**  A record that
passes every check here establishes exactly one thing: *a claim of
authorization was recorded, and that claim is internally coherent with what is
being requested.*  Whether the claim is true is a fact about the world, held by
the people who signed the protocol, and it is their responsibility -- not this
module's -- that it is true.

Nothing here authorises anything.  This package builds no stimulation
controller, no device command path, and no dosing computation for a person; an
:class:`AuthorizationRecord` is an input to a *refusal*, not a permission.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Literal

from pydantic import Field, JsonValue, model_validator

from .base import SchemaModel
from .refusals import CompilerRefusal

__all__ = [
    "AUTHORIZATION_DECLARATION_NOTICE",
    "AuthorizationFailure",
    "AuthorizationFailureCode",
    "AuthorizationRecord",
    "AuthorizationVerdict",
    "ASafeAttribution",
    "ConsentScope",
    "EnrollmentScope",
    "InterventionClass",
    "INTERVENTION_CLASSES",
    "RegulatoryStatus",
    "ResponsibleInvestigator",
    "ValidityWindow",
    "authorize_or_refuse",
    "epoch_seconds",
    "is_placeholder",
    "no_authorization_verdict",
    "validate_authorization",
]

#: Attached to every verdict.  A reader who sees only the verdict must still be
#: told what the verdict does and does not mean.
AUTHORIZATION_DECLARATION_NOTICE = (
    "DECLARATION, NOT VERIFICATION. This record states a claim of ethics, "
    "consent and regulatory authorization made by the people responsible for "
    "the protocol. Validation checks only that the claim is complete, in date, "
    "internally consistent and in scope for the request. It does not verify "
    "with any IRB, REC, institution, sponsor or regulator that the approval "
    "exists or covers this work, and no software can. A validated record means "
    "a claim of authorization was recorded and checked, never that "
    "authorization was granted."
)

#: Intervention classes an authorization can cover.  Deliberately the same
#: vocabulary as ``InterventionModel.modality`` so a record cannot authorise a
#: class the schema cannot express -- and so that a record authorising ``tms``
#: is visibly not a record authorising ``tfus``.
InterventionClass = Literal[
    "tms",
    "tfus",
    "tes",
    "sensory",
    "cognitive_task",
    "neurofeedback",
    "pharmacological",
    "body_state",
    "simulated_impulse",
]

INTERVENTION_CLASSES: tuple[str, ...] = (
    "tms",
    "tfus",
    "tes",
    "sensory",
    "cognitive_task",
    "neurofeedback",
    "pharmacological",
    "body_state",
    "simulated_impulse",
)

#: One code per *distinguishable* way a declaration can fail.  The refusal a
#: consumer sees is always ``R11`` (Table ``tab:compiler-refusals`` is fixed
#: and this module does not invent a twelfth refusal); these are the *reasons*,
#: carried in ``CompilerRefusal.evidence["authorization_failures"]`` so that
#: "expired" and "wrong modality" are never the same message.
AuthorizationFailureCode = Literal[
    "AUTH_ABSENT",
    "AUTH_FIELD_MISSING",
    "AUTH_FIELD_PLACEHOLDER",
    "AUTH_EXPIRED",
    "AUTH_NOT_YET_VALID",
    "AUTH_TIME_UNDECLARED",
    "AUTH_CLASS_NOT_AUTHORIZED",
    "AUTH_CONSENT_OUT_OF_SCOPE",
    "AUTH_CONSENT_UNRESOLVED",
    "AUTH_APPROVAL_EXCEEDS_CONSENT",
    "AUTH_DEVICE_STATUS_UNDECLARED",
    "AUTH_REGULATORY_INCOMPLETE",
    "AUTH_ASAFE_MISSING",
    "AUTH_ASAFE_NOT_TRACEABLE",
    "AUTH_ENROLLMENT_UNDECLARED",
    "AUTH_INVESTIGATOR_UNDECLARED",
]

#: Strings that look filled in but are not.  A required field holding one of
#: these is treated as missing, because "IRB: TBD" is not an approval.
_PLACEHOLDER_TOKENS: frozenset[str] = frozenset(
    {
        "",
        "-",
        "--",
        "?",
        "??",
        "n/a",
        "na",
        "nil",
        "none",
        "null",
        "tbd",
        "tba",
        "todo",
        "to do",
        "fixme",
        "xxx",
        "xxxx",
        "placeholder",
        "example",
        "unknown",
        "unspecified",
        "undeclared",
        "pending",
        "changeme",
        "your irb here",
        "0",
        "00000",
    }
)

_PLACEHOLDER_SUBSTRINGS: tuple[str, ...] = (
    "xxxx",
    "placeholder",
    "todo",
    "fixme",
    "<insert",
    "lorem ipsum",
)


def is_placeholder(value: str | None) -> bool:
    """True when ``value`` is absent or is filler standing in for a real one."""
    if value is None:
        return True
    stripped = value.strip().lower()
    if stripped in _PLACEHOLDER_TOKENS:
        return True
    return any(token in stripped for token in _PLACEHOLDER_SUBSTRINGS)


def epoch_seconds(when: str | date | datetime | float | int) -> float:
    """Seconds on the ``wall`` clock (UTC epoch) for a date, datetime or number.

    Accepts ISO-8601 (``"2026-01-15"`` or ``"2026-01-15T09:00:00+00:00"``) so a
    protocol's approval and expiry dates can be written the way they appear on
    the approval letter.  Naive inputs are read as UTC, explicitly, rather than
    silently picking up whatever timezone the machine happens to sit in.
    """
    if isinstance(when, bool):  # pragma: no cover - misuse guard
        raise TypeError("a bool is not a time")
    if isinstance(when, (int, float)):
        return float(when)
    if isinstance(when, datetime):
        dt = when if when.tzinfo is not None else when.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    if isinstance(when, date):
        return datetime(when.year, when.month, when.day, tzinfo=timezone.utc).timestamp()
    parsed = datetime.fromisoformat(str(when))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _iso(seconds: float | None) -> str:
    if seconds is None:
        return "unbounded"
    return datetime.fromtimestamp(float(seconds), tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# components
# ---------------------------------------------------------------------------


class ValidityWindow(SchemaModel):
    """Half-open ``[start_s, end_s)`` on a named clock. **Approvals expire.**

    ``end_s`` is required and may not be ``None``: an approval with no end date
    is not a thing this repository will represent, because an unbounded
    approval is indistinguishable from an unchecked one.  The clock is named
    for the same reason ``scwbd.transforms.se3.ValidityInterval`` names one --
    "valid until the 14th" must not silently mean a different clock's 14th.
    """

    start_s: float
    end_s: float
    clock: str = "wall"
    #: Human-readable dates, carried for the report and the provenance entry.
    start_label: str = ""
    end_label: str = ""

    @model_validator(mode="after")
    def _check(self) -> "ValidityWindow":
        if self.end_s < self.start_s:
            raise ValueError(
                f"validity window end {_iso(self.end_s)} precedes start "
                f"{_iso(self.start_s)}"
            )
        return self

    @classmethod
    def between(
        cls,
        start: str | date | datetime | float,
        end: str | date | datetime | float,
        *,
        clock: str = "wall",
    ) -> "ValidityWindow":
        return cls(
            start_s=epoch_seconds(start),
            end_s=epoch_seconds(end),
            clock=clock,
            start_label=str(start),
            end_label=str(end),
        )

    def contains(self, t: float) -> bool:
        return self.start_s <= t < self.end_s

    def __str__(self) -> str:
        lo = self.start_label or _iso(self.start_s)
        hi = self.end_label or _iso(self.end_s)
        return f"[{lo}, {hi}) on {self.clock}"


class ConsentScope(SchemaModel):
    """What the participant actually consented to.

    ``covered_intervention_classes`` is the load-bearing field: a consent
    document covering TMS does not cover tFUS, and this is the type that says
    so.  ``covers_prospective_intervention`` is separate because consent to
    *reuse of data* is not consent to *be stimulated*.
    """

    document_id: str = ""
    document_version: str = ""
    #: Free text quoting/summarising the scope as the participant saw it.
    scope_statement: str = ""
    covered_intervention_classes: tuple[InterventionClass, ...] = ()
    covered_procedures: tuple[str, ...] = ()
    #: Whether the consented scope includes receiving a prospective
    #: intervention at all, as opposed to data collection or reuse.
    covers_prospective_intervention: bool = False
    permits_data_reuse: bool = False
    permits_model_release: bool = False
    withdrawal_status: Literal["none_pending", "pending", "honored", "unknown"] = "unknown"

    def covers(self, intervention_class: str) -> bool:
        return (
            self.covers_prospective_intervention
            and intervention_class in self.covered_intervention_classes
        )


class EnrollmentScope(SchemaModel):
    """Who the approval covers: named participants, or a declared cohort.

    One of the two must be present.  "Some patients" is not an enrollment
    scope; neither is a cohort id with no description of who is in it.
    """

    cohort_id: str = ""
    participant_ids: tuple[str, ...] = ()
    declared_scope: str = ""
    population: str = ""
    max_participants: int | None = Field(default=None, ge=1)

    @property
    def is_declared(self) -> bool:
        if any(not is_placeholder(p) for p in self.participant_ids):
            return True
        return not is_placeholder(self.cohort_id) and not is_placeholder(
            self.declared_scope
        )


class RegulatoryStatus(SchemaModel):
    """Device regulatory status. IRB approval alone is not this.

    ``thesis_contract.tex`` Sec. 0.5: in the United States the sponsor and IRB
    must document significant-risk versus nonsignificant-risk status; an SR
    device investigation requires FDA **and** IRB approval, while an NSR study
    follows the abbreviated IDE requirements after IRB concurrence
    (``fdaide``).  The vocabulary below is US-shaped and says so; a record from
    another jurisdiction must still declare a determination and who made it,
    and must name the instrument it was made under.
    """

    jurisdiction: str = ""
    device_identifier: str = ""
    risk_determination: Literal[
        "significant_risk", "nonsignificant_risk", "exempt", "not_a_device", "undeclared"
    ] = "undeclared"
    #: Who documented the determination. FDA expects the sponsor and the IRB.
    determined_by: str = ""
    ide_number: str | None = None
    fda_approval_status: Literal["approved", "not_required", "pending", "unknown"] = (
        "unknown"
    )
    irb_concurrence: bool = False
    #: 510(k)/De Novo/PMA/CE reference, if the device is marketed.
    marketing_authorization: str = ""
    #: Required when ``risk_determination`` is ``exempt`` or ``not_a_device``.
    exemption_basis: str = ""
    #: The instrument the determination was made under, for non-US records.
    regulatory_instrument: str = ""


class ResponsibleInvestigator(SchemaModel):
    """The named human who is answerable for the protocol."""

    name: str = ""
    role: str = ""
    institution: str = ""
    contact: str = ""

    @property
    def is_declared(self) -> bool:
        return not is_placeholder(self.name) and not is_placeholder(self.institution)


class ASafeAttribution(SchemaModel):
    """The link from this protocol to *its* ``A_safe``, not a generic default.

    ``scwbd.intervene.safety`` keeps :math:`\\mathcal A_{\\rm safe}` outside the
    learned objective and outside this record: an authorization **never widens
    the feasible set**.  What this attribution does is the opposite -- it makes
    the record name the limits the protocol itself declares, so that "we are
    approved" cannot quietly inherit whatever bounds happen to ship in
    ``limits/a_safe.toml``.  ``derivation`` must be ``declared_in_protocol``;
    ``generic_default`` is refused by name.
    """

    a_safe_id: str = ""
    #: Where the protocol's limits are written down (document, section, path).
    source: str = ""
    #: ``"<protocol_id>@<protocol_version>"``; must match the record's protocol.
    protocol_reference: str = ""
    derivation: Literal["declared_in_protocol", "generic_default", "unknown"] = "unknown"
    #: Axis names the protocol bounds, e.g. ``"tms.peak_efield_v_per_m"``.
    constraint_axes: tuple[str, ...] = ()
    #: Hash of the limits document, if it is a file under version control.
    limits_content_hash: str | None = None
    independently_validated: bool = False
    validator: str = ""


# ---------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------


class AuthorizationRecord(SchemaModel):
    """A recorded, content-addressed claim of authorization for a protocol.

    Frozen, ``extra="forbid"``, and hashable via :meth:`content_hash` like every
    other ``scwbd.schema`` object, so that the exact record a run was compiled
    under is pinned in provenance and cannot be edited after the fact without
    changing the hash the artifact carries.

    Construction is deliberately permissive: an incomplete or expired record is
    *representable*, because the refusal paths have to be exercisable by tests.
    Completeness, currency and scope are decided by
    :func:`validate_authorization` against a specific request, never by the
    record's mere existence.
    """

    id: str
    #: IRB / REC / ethics committee name.
    approving_body: str = ""
    approving_body_kind: Literal["irb", "rec", "ethics_committee", "other"] = "other"
    #: The approval number as issued by that body.
    approval_identifier: str = ""
    approval_date_label: str = ""

    protocol_id: str = ""
    protocol_version: str = ""
    protocol_title: str = ""

    #: Intervention classes this approval covers. A record authorising TMS does
    #: not authorise tFUS, and this tuple is where that is written.
    authorized_intervention_classes: tuple[InterventionClass, ...] = ()

    consent: ConsentScope = Field(default_factory=ConsentScope)
    enrollment: EnrollmentScope = Field(default_factory=EnrollmentScope)
    regulatory: RegulatoryStatus = Field(default_factory=RegulatoryStatus)
    investigator: ResponsibleInvestigator = Field(default_factory=ResponsibleInvestigator)
    a_safe: ASafeAttribution = Field(default_factory=ASafeAttribution)
    validity: ValidityWindow

    #: Who entered this declaration into the repository, and when.
    declared_by: str = ""
    declared_on_s: float | None = None
    notes: dict[str, JsonValue] = Field(default_factory=dict)
    notice: str = AUTHORIZATION_DECLARATION_NOTICE

    @property
    def protocol_reference(self) -> str:
        return f"{self.protocol_id}@{self.protocol_version}"

    def authorizes_class(self, intervention_class: str) -> bool:
        """Membership only. Not a verdict -- see :func:`validate_authorization`."""
        return intervention_class in self.authorized_intervention_classes

    def label(self) -> str:
        return (
            f"{self.approving_body} {self.approval_identifier} "
            f"protocol {self.protocol_reference}"
        ).strip()

    def summary(self) -> str:
        return (
            f"AuthorizationRecord[{self.id}] {self.label()} "
            f"classes={list(self.authorized_intervention_classes)} "
            f"valid {self.validity}"
        )


# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------


class AuthorizationFailure(SchemaModel):
    """One specific, named reason a declaration did not admit a request."""

    code: AuthorizationFailureCode
    field: str
    reason: str
    remedy: str = ""

    def __str__(self) -> str:
        return f"{self.code} [{self.field}]: {self.reason}"


class AuthorizationVerdict(SchemaModel):
    """The result of checking one declaration against one request.

    ``claim_scope`` is the part that must reach provenance.  An admitted
    request produces artifacts scoped ``protocol:<id>@<version>`` instead of
    ``simulation_only``: the artifact is now one produced under a *named*
    protocol, and downstream consumers can see which.
    """

    admitted: bool
    record_id: str = ""
    record_hash: str = ""
    protocol: str = ""
    approving_body: str = ""
    approval_identifier: str = ""
    intervention_class: str = ""
    at_time_s: float | None = None
    checks_passed: tuple[str, ...] = ()
    failures: tuple[AuthorizationFailure, ...] = ()
    claim_scope: str = "simulation_only"
    notice: str = AUTHORIZATION_DECLARATION_NOTICE

    @property
    def failure_codes(self) -> tuple[str, ...]:
        return tuple(f.code for f in self.failures)

    @property
    def primary_code(self) -> str | None:
        return self.failures[0].code if self.failures else None

    def reason(self) -> str:
        if self.admitted:
            return (
                f"declared authorization {self.record_id!r} ({self.approving_body} "
                f"{self.approval_identifier}) covers {self.intervention_class} "
                f"under protocol {self.protocol} at "
                f"{_iso(self.at_time_s)}"
            )
        return "; ".join(str(f) for f in self.failures) or "no reason recorded"

    def as_provenance(self) -> dict[str, Any]:
        """The entry recorded in ``CompiledModel.provenance`` and in artifacts."""
        return {
            "admitted": self.admitted,
            "record_id": self.record_id,
            "record_hash": self.record_hash,
            "protocol": self.protocol,
            "approving_body": self.approving_body,
            "approval_identifier": self.approval_identifier,
            "intervention_class": self.intervention_class,
            "at_time_s": self.at_time_s,
            "at_time_iso": _iso(self.at_time_s),
            "checks_passed": list(self.checks_passed),
            "failures": [f.model_dump(mode="json") for f in self.failures],
            "claim_scope": self.claim_scope,
            "notice": self.notice,
        }

    def raise_if_refused(self, *, offending_object: Any = None) -> "AuthorizationVerdict":
        """Return the verdict, or raise ``CompilerRefusal(code="R11")``."""
        if self.admitted:
            return self
        raise CompilerRefusal(
            "R11",
            remedy=(
                "supply a complete, in-date AuthorizationRecord whose consent "
                "scope covers the requested intervention class and whose A_safe "
                "is attributable to the named protocol; "
                + "; ".join(f.remedy for f in self.failures if f.remedy)
            ),
            offending_object=offending_object if offending_object is not None else self.record_id,
            detail=(
                f"no validated authorization admits a {self.intervention_class} "
                f"request: {self.reason()}"
            ),
            evidence={
                "authorization_failures": [f.model_dump(mode="json") for f in self.failures],
                "authorization_record_id": self.record_id,
                "authorization_record_hash": self.record_hash,
                "intervention_class": self.intervention_class,
                "claim_scope": self.claim_scope,
            },
        )


def no_authorization_verdict(
    intervention_class: str, *, at_time_s: float | None = None, what: str = "request"
) -> AuthorizationVerdict:
    """The verdict for "there is no record at all".

    This is the ordinary case for SC-WBD-001-beta and it still refuses -- what
    changed is *why*: the refusal now names a missing, supplyable artifact
    instead of asserting a belief about the world.
    """
    return AuthorizationVerdict(
        admitted=False,
        intervention_class=intervention_class,
        at_time_s=at_time_s,
        failures=(
            AuthorizationFailure(
                code="AUTH_ABSENT",
                field="authorization",
                reason=(
                    f"{what} declares a prospective {intervention_class} "
                    "intervention but carries no AuthorizationRecord"
                ),
                remedy=(
                    "attach an AuthorizationRecord (IRB/REC identifier, protocol "
                    "id and version, consent scope, enrollment scope, device "
                    "regulatory status, validity window, responsible "
                    "investigator, protocol-attributable A_safe)"
                ),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def _require(
    failures: list[AuthorizationFailure],
    value: str | None,
    field: str,
    what: str,
    remedy: str,
) -> bool:
    """Record a missing/placeholder failure for a required text field."""
    if value is None or not str(value).strip():
        failures.append(
            AuthorizationFailure(
                code="AUTH_FIELD_MISSING",
                field=field,
                reason=f"{what} is missing",
                remedy=remedy,
            )
        )
        return False
    if is_placeholder(value):
        failures.append(
            AuthorizationFailure(
                code="AUTH_FIELD_PLACEHOLDER",
                field=field,
                reason=f"{what} is placeholder text {value.strip()!r}, not a value",
                remedy=remedy,
            )
        )
        return False
    return True


def validate_authorization(
    record: AuthorizationRecord | None,
    *,
    intervention_class: str,
    at_time_s: float | str | date | datetime | None,
    a_safe_id: str | None = None,
    required_a_safe_axes: tuple[str, ...] = (),
    what: str = "request",
) -> AuthorizationVerdict:
    """Decide whether ``record`` admits a request, and say exactly why not.

    Parameters
    ----------
    record:
        The declaration, or ``None`` when there is none (the ordinary case).
    intervention_class:
        What is actually being asked for.  A record authorising ``"tms"`` does
        not admit a ``"tfus"`` request; that is a refusal with its own code.
    at_time_s:
        When the request applies.  Approvals expire, so a request with no time
        cannot be checked against a validity window and is refused rather than
        assumed current.
    a_safe_id / required_a_safe_axes:
        The feasible set the caller intends to operate under.  The record's
        ``A_safe`` attribution must name it (and cover those axes), so that the
        limits in force are the protocol's, not a generic default's.

    Returns
    -------
    AuthorizationVerdict
        ``admitted=True`` only when *every* check passes.  This function never
        raises for an invalid record; call
        :meth:`AuthorizationVerdict.raise_if_refused` to turn a refusal into
        ``CompilerRefusal(code="R11")``.

    Claim limit
    -----------
    See the module docstring.  This validates a **declaration**; it does not
    establish that an IRB approval exists, and nothing in this repository can.
    """
    if intervention_class not in INTERVENTION_CLASSES:
        raise ValueError(
            f"unknown intervention class {intervention_class!r}; "
            f"expected one of {INTERVENTION_CLASSES}"
        )

    t: float | None
    if at_time_s is None:
        t = None
    else:
        t = epoch_seconds(at_time_s)

    if record is None:
        return no_authorization_verdict(intervention_class, at_time_s=t, what=what)

    failures: list[AuthorizationFailure] = []
    passed: list[str] = []

    # -- 1. identity of the approving body and the protocol -----------------
    ok = True
    ok &= _require(
        failures,
        record.approving_body,
        "approving_body",
        "the approving body (IRB/REC) name",
        "name the committee that issued the approval",
    )
    ok &= _require(
        failures,
        record.approval_identifier,
        "approval_identifier",
        "the IRB/REC approval identifier",
        "record the approval number as issued",
    )
    ok &= _require(
        failures,
        record.protocol_id,
        "protocol_id",
        "the protocol identifier",
        "record the protocol identifier the approval refers to",
    )
    ok &= _require(
        failures,
        record.protocol_version,
        "protocol_version",
        "the protocol version",
        "record the approved protocol version; approvals attach to versions",
    )
    if ok:
        passed.append("approval_identity")

    # -- 2. responsible investigator ----------------------------------------
    if record.investigator.is_declared:
        passed.append("responsible_investigator")
    else:
        failures.append(
            AuthorizationFailure(
                code="AUTH_INVESTIGATOR_UNDECLARED",
                field="investigator",
                reason=(
                    "no responsible investigator is declared (a name and an "
                    "institution are both required)"
                ),
                remedy="name the investigator answerable for the protocol",
            )
        )

    # -- 3. validity window: approvals expire --------------------------------
    if t is None:
        failures.append(
            AuthorizationFailure(
                code="AUTH_TIME_UNDECLARED",
                field="at_time_s",
                reason=(
                    "the request declares no time, so it cannot be checked "
                    "against the approval's validity window; an undated request "
                    "is not assumed current"
                ),
                remedy="declare the time the request applies to",
            )
        )
    elif t < record.validity.start_s:
        failures.append(
            AuthorizationFailure(
                code="AUTH_NOT_YET_VALID",
                field="validity",
                reason=(
                    f"the request at {_iso(t)} precedes the approval's validity "
                    f"window {record.validity}"
                ),
                remedy="wait for the approval to take effect, or correct the date",
            )
        )
    elif t >= record.validity.end_s:
        failures.append(
            AuthorizationFailure(
                code="AUTH_EXPIRED",
                field="validity",
                reason=(
                    f"the approval expired at {_iso(record.validity.end_s)}; the "
                    f"request applies at {_iso(t)}"
                ),
                remedy="obtain and record a current continuing-review approval",
            )
        )
    else:
        passed.append("validity_window")

    # -- 4. the requested class must be inside the approved scope ------------
    if not record.authorized_intervention_classes:
        failures.append(
            AuthorizationFailure(
                code="AUTH_CLASS_NOT_AUTHORIZED",
                field="authorized_intervention_classes",
                reason="the record authorises no intervention class at all",
                remedy="declare the intervention classes the approval covers",
            )
        )
    elif intervention_class not in record.authorized_intervention_classes:
        failures.append(
            AuthorizationFailure(
                code="AUTH_CLASS_NOT_AUTHORIZED",
                field="authorized_intervention_classes",
                reason=(
                    f"the approval covers "
                    f"{list(record.authorized_intervention_classes)} and the "
                    f"request is {intervention_class!r}; an approval for one "
                    "intervention class is not an approval for another"
                ),
                remedy=(
                    f"obtain an approval covering {intervention_class}, or "
                    "request only an authorized class"
                ),
            )
        )
    else:
        passed.append("intervention_class_authorized")

    # -- 5. consent scope ----------------------------------------------------
    consent = record.consent
    consent_ok = True
    consent_ok &= _require(
        failures,
        consent.document_id,
        "consent.document_id",
        "the consent document identifier",
        "record which consent document the participants signed",
    )
    consent_ok &= _require(
        failures,
        consent.document_version,
        "consent.document_version",
        "the consent document version",
        "record the consent version; consent attaches to versions",
    )
    consent_ok &= _require(
        failures,
        consent.scope_statement,
        "consent.scope_statement",
        "the statement of what participants consented to",
        "state the consented scope in the participants' terms",
    )
    if not consent.covers_prospective_intervention:
        consent_ok = False
        failures.append(
            AuthorizationFailure(
                code="AUTH_CONSENT_OUT_OF_SCOPE",
                field="consent.covers_prospective_intervention",
                reason=(
                    "the consent scope does not include receiving a prospective "
                    "intervention; consent to data collection or reuse is not "
                    "consent to be stimulated"
                ),
                remedy="record consent that explicitly covers the intervention",
            )
        )
    elif intervention_class not in consent.covered_intervention_classes:
        consent_ok = False
        failures.append(
            AuthorizationFailure(
                code="AUTH_CONSENT_OUT_OF_SCOPE",
                field="consent.covered_intervention_classes",
                reason=(
                    f"the consent covers "
                    f"{list(consent.covered_intervention_classes)} and the "
                    f"request is {intervention_class!r}; a participant who "
                    "consented to TMS has not consented to tFUS"
                ),
                remedy=(
                    f"re-consent participants for {intervention_class} before "
                    "requesting it"
                ),
            )
        )
    if consent.withdrawal_status != "none_pending":
        consent_ok = False
        failures.append(
            AuthorizationFailure(
                code="AUTH_CONSENT_UNRESOLVED",
                field="consent.withdrawal_status",
                reason=(
                    f"withdrawal status is {consent.withdrawal_status!r}; only "
                    "'none_pending' is an admissible state for a prospective "
                    "request"
                ),
                remedy="resolve the withdrawal before proceeding",
            )
        )
    extra = tuple(
        c
        for c in record.authorized_intervention_classes
        if c not in consent.covered_intervention_classes
    )
    if extra:
        consent_ok = False
        failures.append(
            AuthorizationFailure(
                code="AUTH_APPROVAL_EXCEEDS_CONSENT",
                field="authorized_intervention_classes",
                reason=(
                    f"the record claims approval for {list(extra)}, which the "
                    "declared consent scope does not cover; an approval cannot "
                    "exceed what participants agreed to"
                ),
                remedy="align the approved classes with the consented classes",
            )
        )
    if consent_ok:
        passed.append("consent_scope")

    # -- 6. device regulatory status ----------------------------------------
    reg = record.regulatory
    reg_ok = True
    if reg.risk_determination == "undeclared":
        reg_ok = False
        failures.append(
            AuthorizationFailure(
                code="AUTH_DEVICE_STATUS_UNDECLARED",
                field="regulatory.risk_determination",
                reason=(
                    "no device risk determination is recorded; IRB approval "
                    "alone is not a universal regulatory description "
                    "(thesis_contract Sec. 0.5, fdaide)"
                ),
                remedy=(
                    "record the sponsor/IRB significant-risk vs "
                    "nonsignificant-risk determination, or the basis for exemption"
                ),
            )
        )
    if not _require(
        failures,
        reg.jurisdiction,
        "regulatory.jurisdiction",
        "the regulatory jurisdiction",
        "name the jurisdiction whose rules the determination was made under",
    ):
        reg_ok = False
    if not _require(
        failures,
        reg.device_identifier,
        "regulatory.device_identifier",
        "the device identifier",
        "name the device the protocol uses",
    ):
        reg_ok = False
    if reg.risk_determination != "undeclared" and not _require(
        failures,
        reg.determined_by,
        "regulatory.determined_by",
        "who documented the risk determination",
        "record the sponsor and IRB that documented the determination",
    ):
        reg_ok = False

    if reg.risk_determination == "significant_risk":
        if is_placeholder(reg.ide_number):
            reg_ok = False
            failures.append(
                AuthorizationFailure(
                    code="AUTH_REGULATORY_INCOMPLETE",
                    field="regulatory.ide_number",
                    reason=(
                        "a significant-risk device investigation requires an IDE; "
                        "no IDE number is recorded"
                    ),
                    remedy="record the IDE number and its FDA approval status",
                )
            )
        if reg.fda_approval_status != "approved":
            reg_ok = False
            failures.append(
                AuthorizationFailure(
                    code="AUTH_REGULATORY_INCOMPLETE",
                    field="regulatory.fda_approval_status",
                    reason=(
                        f"a significant-risk investigation requires FDA and IRB "
                        f"approval; FDA status is {reg.fda_approval_status!r}"
                    ),
                    remedy="record FDA approval of the IDE before requesting",
                )
            )
        if not reg.irb_concurrence:
            reg_ok = False
            failures.append(
                AuthorizationFailure(
                    code="AUTH_REGULATORY_INCOMPLETE",
                    field="regulatory.irb_concurrence",
                    reason="IRB concurrence with the risk determination is not recorded",
                    remedy="record the IRB's concurrence",
                )
            )
    elif reg.risk_determination == "nonsignificant_risk":
        if not reg.irb_concurrence:
            reg_ok = False
            failures.append(
                AuthorizationFailure(
                    code="AUTH_REGULATORY_INCOMPLETE",
                    field="regulatory.irb_concurrence",
                    reason=(
                        "an NSR study follows the abbreviated IDE requirements "
                        "only after IRB concurrence with the NSR determination; "
                        "no concurrence is recorded"
                    ),
                    remedy="record the IRB's concurrence with the NSR determination",
                )
            )
    elif reg.risk_determination in ("exempt", "not_a_device"):
        if not _require(
            failures,
            reg.exemption_basis,
            "regulatory.exemption_basis",
            f"the basis for the {reg.risk_determination!r} determination",
            "state the regulatory basis for the exemption",
        ):
            reg_ok = False
    if reg_ok:
        passed.append("device_regulatory_status")

    # -- 7. A_safe attributable to *this* protocol ---------------------------
    a = record.a_safe
    a_ok = True
    if not _require(
        failures,
        a.a_safe_id,
        "a_safe.a_safe_id",
        "the identifier of the protocol's A_safe",
        "name the feasible set the protocol declares",
    ):
        a_ok = False
    if not a.constraint_axes:
        a_ok = False
        failures.append(
            AuthorizationFailure(
                code="AUTH_ASAFE_MISSING",
                field="a_safe.constraint_axes",
                reason=(
                    "the record names no exposure axes the protocol bounds; an "
                    "authorization with no limits attached is not an "
                    "authorization to operate within limits"
                ),
                remedy="list the axes the protocol bounds, with the source",
            )
        )
    if a.derivation != "declared_in_protocol":
        a_ok = False
        failures.append(
            AuthorizationFailure(
                code="AUTH_ASAFE_NOT_TRACEABLE",
                field="a_safe.derivation",
                reason=(
                    f"A_safe derivation is {a.derivation!r}; the limits in force "
                    "must be traceable to this protocol, not inherited from a "
                    "generic default"
                ),
                remedy="cite the protocol section that declares the limits",
            )
        )
    if a.protocol_reference != record.protocol_reference:
        a_ok = False
        failures.append(
            AuthorizationFailure(
                code="AUTH_ASAFE_NOT_TRACEABLE",
                field="a_safe.protocol_reference",
                reason=(
                    f"A_safe is attributed to {a.protocol_reference!r} but the "
                    f"record's protocol is {record.protocol_reference!r}"
                ),
                remedy="attribute the limits to the protocol version approved",
            )
        )
    if not a.independently_validated or is_placeholder(a.validator):
        a_ok = False
        failures.append(
            AuthorizationFailure(
                code="AUTH_ASAFE_NOT_TRACEABLE",
                field="a_safe.independently_validated",
                reason=(
                    "the protocol's A_safe is not recorded as independently "
                    "validated by a named party; R11 requires an independently "
                    "validated feasible set"
                ),
                remedy="record the independent validator of the limits",
            )
        )
    if a_safe_id is not None and a.a_safe_id != a_safe_id:
        a_ok = False
        failures.append(
            AuthorizationFailure(
                code="AUTH_ASAFE_NOT_TRACEABLE",
                field="a_safe.a_safe_id",
                reason=(
                    f"the request operates under A_safe {a_safe_id!r} but the "
                    f"authorization attributes limits to {a.a_safe_id!r}"
                ),
                remedy="operate under the feasible set the protocol declares",
            )
        )
    missing_axes = tuple(ax for ax in required_a_safe_axes if ax not in a.constraint_axes)
    if missing_axes:
        a_ok = False
        failures.append(
            AuthorizationFailure(
                code="AUTH_ASAFE_NOT_TRACEABLE",
                field="a_safe.constraint_axes",
                reason=(
                    f"the request is checked on axes {list(missing_axes)} that "
                    "the protocol's declared limits do not cover"
                ),
                remedy="declare protocol limits for every axis being checked",
            )
        )
    if a_ok:
        passed.append("a_safe_traceable")

    # -- 8. enrollment scope -------------------------------------------------
    if record.enrollment.is_declared:
        passed.append("enrollment_declared")
    else:
        failures.append(
            AuthorizationFailure(
                code="AUTH_ENROLLMENT_UNDECLARED",
                field="enrollment",
                reason=(
                    "no participant identifiers and no declared cohort scope; "
                    "'some patients' is not an enrollment scope"
                ),
                remedy="record participant identifiers or a described cohort",
            )
        )

    admitted = not failures
    return AuthorizationVerdict(
        admitted=admitted,
        record_id=record.id,
        record_hash=record.content_hash(),
        protocol=record.protocol_reference,
        approving_body=record.approving_body,
        approval_identifier=record.approval_identifier,
        intervention_class=intervention_class,
        at_time_s=t,
        checks_passed=tuple(passed),
        failures=tuple(failures),
        claim_scope=(
            f"protocol:{record.protocol_reference}" if admitted else "simulation_only"
        ),
    )


def authorize_or_refuse(
    record: AuthorizationRecord | None,
    *,
    intervention_class: str,
    at_time_s: float | str | date | datetime | None,
    a_safe_id: str | None = None,
    required_a_safe_axes: tuple[str, ...] = (),
    what: str = "request",
    offending_object: Any = None,
) -> AuthorizationVerdict:
    """:func:`validate_authorization`, raising ``CompilerRefusal(R11)`` on refusal."""
    verdict = validate_authorization(
        record,
        intervention_class=intervention_class,
        at_time_s=at_time_s,
        a_safe_id=a_safe_id,
        required_a_safe_axes=required_a_safe_axes,
        what=what,
    )
    return verdict.raise_if_refused(offending_object=offending_object)
