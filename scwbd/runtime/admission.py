"""The export edge: what may leave this repository, for what purpose.

This module is the single enforced gate between SC-WBD and any consumer of it,
including ``~/Documents/robotics`` (``tms-robotics``).  It exists because run 1
shipped and was demonstrated without one.

Two independent things are checked here and both must pass:

**1. Does this checkpoint support the purpose it is being loaded for?**
    The run-1 artifact is the equal-capacity generic-operator *control arm* of
    ``body.tex`` Sec. 11.4's first required ablation (``reports/scope_gap.md``),
    its anatomy is ``is_biological: false``, and its claim gates are
    ``COULD_NOT_RUN``.  Each of those is a reason a consumer may not use it for
    a purpose that assumes otherwise, and each is checked by name.

**2. Is live application authorized?**
    Everything computational inside this repository -- simulation, modelling,
    intervention physics, planning against simulated or recorded open data,
    training, benchmarks -- is approved work and is **not** gated here.  What is
    gated is *live application*: driving real hardware, or informing a real
    person's stimulation, in production in the consumer repository.  That is
    pending a preliminary review.

    The gate does **not** open on a date.  A calendar comparison encodes "the
    review was scheduled", and a scheduled review is not a completed one; such
    a check also silently opens on its own the day after.  What opens this gate
    is a :class:`LiveUseAuthorization` -- a record that a review *occurred* and
    what its *outcome* was.  :data:`EARLIEST_CREDIBLE_REVIEW` is a floor on
    when such a record could exist, used only to reject records that claim an
    outcome from before the review could have happened.  It is not the gate.

Orthogonality, stated because it will otherwise be misread
----------------------------------------------------------
``sim2real_ready``, ``promotion_eligible`` and ``robot_command_authority``
remain ``False`` standing invariants of the consumer, unconditionally.  They are
**not** what this gate governs and no authorization record here relaxes them.
An approving review of live application is not a promotion decision; anyone
reading "authorized" here as "promotion eligible" has crossed two unrelated
boundaries.  :data:`CONSUMER_STANDING_INVARIANTS` is checked on every admission,
for every purpose, and cannot be widened -- see :class:`ConsumerInvariants`.

Claim limits
------------
Admission is a *refusal* mechanism.  Passing it asserts that no checked
condition failed; it asserts nothing about whether the model is accurate, and
nothing whatsoever about whether an intervention is safe for a person.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

__all__ = [
    "ExportPurpose",
    "EXPORT_PURPOSES",
    "LIVE_PURPOSES",
    "EARLIEST_CREDIBLE_REVIEW",
    "CONSUMER_STANDING_INVARIANTS",
    "ConsumerInvariants",
    "ConsumerInvariantViolation",
    "LiveUseAuthorization",
    "AuthorizationInvalid",
    "AdmissionCondition",
    "AdmissionVerdict",
    "CheckpointRefused",
    "CheckpointClaims",
    "admit",
    "SIDECAR_NAME",
]

# --------------------------------------------------------------------------
# purposes
# --------------------------------------------------------------------------

ExportPurpose = Literal[
    #: numbers that stay inside a simulation and are reported as simulation.
    "simulation",
    #: offline ranking of hypotheses against recorded or simulated data; no
    #: hardware, no person.  ``TargetingService.evaluate_pose`` in its
    #: research use.
    "research_offline",
    #: the prediction reaches a physical robot.
    "live_hardware",
    #: the prediction informs what is done to a real person.
    "patient_directed",
]

EXPORT_PURPOSES: tuple[ExportPurpose, ...] = (
    "simulation",
    "research_offline",
    "live_hardware",
    "patient_directed",
)

#: Purposes that constitute *live application* and therefore require a
#: :class:`LiveUseAuthorization`.
LIVE_PURPOSES: frozenset[str] = frozenset({"live_hardware", "patient_directed"})

#: The earliest date on which a record of the pending preliminary review could
#: exist.  **This is a floor, not the gate.**  A record dated before this did
#: not come from that review; a record dated after it still only admits
#: anything if its ``outcome`` is approving.  Nothing opens because this date
#: passes.
EARLIEST_CREDIBLE_REVIEW = date(2026, 8, 25)

#: The consumer's standing invariants.  Unconditional, and unrelated to any
#: authorization recorded here.
CONSUMER_STANDING_INVARIANTS: Mapping[str, bool] = {
    "sim2real_ready": False,
    "promotion_eligible": False,
    "robot_command_authority": False,
}

#: Filename of the admission sidecar beside a checkpoint.
SIDECAR_NAME = "claim_manifest.json"


# --------------------------------------------------------------------------
# standing invariants
# --------------------------------------------------------------------------

class ConsumerInvariantViolation(RuntimeError):
    """Someone tried to construct a consumer state with an invariant widened."""


@dataclass(frozen=True)
class ConsumerInvariants:
    """The three flags, which may be read but not set true.

    They are carried as a value so a consumer can record them in its own
    evidence graph, and they refuse construction in any other state.  This is
    the same shape the consumer's own ``BRIDGE_INVARIANTS`` uses; keeping it
    identical on both sides means the two cannot drift into disagreeing.
    """

    sim2real_ready: bool = False
    promotion_eligible: bool = False
    robot_command_authority: bool = False

    def __post_init__(self) -> None:
        for name, required in CONSUMER_STANDING_INVARIANTS.items():
            actual = getattr(self, name)
            if actual is not required:
                raise ConsumerInvariantViolation(
                    f"{name}={actual!r} but this repository's standing "
                    f"invariant is {name}={required!r}. This is not relaxed by "
                    "any authorization record: an approving review of live "
                    "application is a compliance outcome, not a promotion "
                    "decision, and the two are unrelated boundaries"
                )

    def as_dict(self) -> dict[str, bool]:
        return dict(asdict(self))


# --------------------------------------------------------------------------
# live-use authorization
# --------------------------------------------------------------------------

class AuthorizationInvalid(RuntimeError):
    """A live-use record that does not establish what it claims."""


#: Outcomes that admit live application.  Everything else -- including
#: ``"pending"``, ``"scheduled"`` and ``"deferred"`` -- does not.
APPROVING_OUTCOMES: frozenset[str] = frozenset(
    {"approved", "approved_with_conditions"}
)


@dataclass(frozen=True)
class LiveUseAuthorization:
    """A record that a review **occurred**, and what it decided.

    The point of the type is that there is no way to express "the date has
    passed" in it.  It carries a body, a reference, an outcome, the date the
    outcome was reached, the purposes it covers, and who recorded it.  A
    consumer cannot manufacture admission by waiting.
    """

    review_body: str
    reference: str
    outcome: str
    reviewed_on: date
    #: Which :data:`EXPORT_PURPOSES` this record covers.  A record approving
    #: ``live_hardware`` does not thereby approve ``patient_directed``.
    scope: tuple[str, ...]
    recorded_by: str
    #: Conditions attached to an ``approved_with_conditions`` outcome.  Recorded
    #: so that they travel with the artifact; this module does not verify them.
    conditions: tuple[str, ...] = ()
    expires_on: date | None = None
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.review_body or not self.reference:
            raise AuthorizationInvalid(
                "a live-use authorization must name the reviewing body and its "
                "reference; an unattributed approval is not a record of one"
            )
        if not self.recorded_by:
            raise AuthorizationInvalid(
                "a live-use authorization must name who recorded it"
            )
        if not self.scope:
            raise AuthorizationInvalid(
                "a live-use authorization must name the purposes it covers; an "
                "unscoped approval would admit every purpose, which is not "
                "what any review grants"
            )
        bad = [s for s in self.scope if s not in EXPORT_PURPOSES]
        if bad:
            raise AuthorizationInvalid(
                f"unknown purposes in scope: {bad}; known: {list(EXPORT_PURPOSES)}"
            )
        if self.outcome in APPROVING_OUTCOMES and self.reviewed_on < EARLIEST_CREDIBLE_REVIEW:
            raise AuthorizationInvalid(
                f"record claims outcome {self.outcome!r} reached on "
                f"{self.reviewed_on.isoformat()}, but the preliminary review it "
                f"would have to come from could not have concluded before "
                f"{EARLIEST_CREDIBLE_REVIEW.isoformat()}. This is a floor on "
                "when such a record could exist, not a date on which anything "
                "becomes permitted"
            )
        if self.outcome == "approved_with_conditions" and not self.conditions:
            raise AuthorizationInvalid(
                "outcome 'approved_with_conditions' with no conditions listed; "
                "the conditions are the substance of that outcome"
            )

    # -- the question the gate asks ----------------------------------------
    def refusal_for(self, purpose: str, *, as_of: date) -> str | None:
        """``None`` if this record admits ``purpose`` on ``as_of``, else why not."""
        if self.outcome not in APPROVING_OUTCOMES:
            return (
                f"the recorded outcome of {self.review_body} {self.reference} is "
                f"{self.outcome!r}, which does not approve live application "
                f"(approving outcomes: {sorted(APPROVING_OUTCOMES)})"
            )
        if self.reviewed_on > as_of:
            return (
                f"the record is dated {self.reviewed_on.isoformat()}, which is "
                f"after the date being evaluated ({as_of.isoformat()}): it "
                "describes a review that has not happened yet"
            )
        if self.expires_on is not None and self.expires_on < as_of:
            return (
                f"the authorization expired on {self.expires_on.isoformat()} "
                f"(evaluating as of {as_of.isoformat()})"
            )
        if purpose not in self.scope:
            return (
                f"purpose {purpose!r} is not in the record's scope "
                f"{list(self.scope)}; approval of one purpose is not approval "
                "of another"
            )
        return None

    def canonical(self) -> dict[str, Any]:
        return {
            "review_body": self.review_body,
            "reference": self.reference,
            "outcome": self.outcome,
            "reviewed_on": self.reviewed_on.isoformat(),
            "scope": list(self.scope),
            "recorded_by": self.recorded_by,
            "conditions": list(self.conditions),
            "expires_on": self.expires_on.isoformat() if self.expires_on else None,
            "notes": dict(self.notes),
        }

    def record_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.canonical(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "LiveUseAuthorization":
        exp = raw.get("expires_on")
        return cls(
            review_body=str(raw["review_body"]),
            reference=str(raw["reference"]),
            outcome=str(raw["outcome"]),
            reviewed_on=date.fromisoformat(str(raw["reviewed_on"])),
            scope=tuple(str(s) for s in raw.get("scope", ())),
            recorded_by=str(raw["recorded_by"]),
            conditions=tuple(str(c) for c in raw.get("conditions", ())),
            expires_on=date.fromisoformat(str(exp)) if exp else None,
            notes=dict(raw.get("notes", {})),
        )


# --------------------------------------------------------------------------
# what a checkpoint says about itself
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CheckpointClaims:
    """The admission-relevant facts, read from the sidecar beside a checkpoint.

    Every field defaults to the **refusing** value.  A fact that is not stated
    is not thereby true: an artifact with no anatomy record is treated as one
    whose anatomy is not biological, because that is the state we can defend.
    """

    manifest_id: str = "absent"
    claim_class: str = "surrogate"
    posterior_class: str = "pseudo"
    #: ``True`` when this checkpoint is an ablation control rather than the
    #: model.  ``reports/scope_gap.md``: run 1 is the equal-capacity
    #: generic-operator control of ``body.tex`` Sec. 11.4.
    is_control_arm: bool = True
    control_arm_of: str = "unstated"
    #: From the checkpoint's ``extra.anatomy``.
    anatomy_is_biological: bool = False
    anatomy_provenance: str = "unstated"
    #: Gate name -> status. ``COULD_NOT_RUN`` blocks every non-simulation use.
    gates: Mapping[str, str] = field(default_factory=dict)
    #: Whether real weights were found and hashed.
    weights_trained: bool = False
    #: Digest of the port contract this checkpoint declares (see
    #: :mod:`scwbd.runtime.ports`).
    port_contract_digest: str = ""
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def unrun_gates(self) -> tuple[str, ...]:
        return tuple(
            sorted(k for k, v in self.gates.items() if str(v).upper() == "COULD_NOT_RUN")
        )

    @property
    def failed_gates(self) -> tuple[str, ...]:
        return tuple(sorted(k for k, v in self.gates.items() if str(v).upper() == "FAIL"))

    @classmethod
    def absent(cls) -> "CheckpointClaims":
        """The claims of a checkpoint that shipped without a manifest."""
        return cls()

    @classmethod
    def from_manifest(cls, raw: Mapping[str, Any]) -> "CheckpointClaims":
        """Read a sidecar.  Unstated facts stay at their refusing defaults."""
        notes = dict(raw.get("notes") or {})
        anat = dict(raw.get("anatomy") or notes.get("anatomy") or {})
        gates = dict(raw.get("gates") or notes.get("gates") or {})
        arm = raw.get("arm", notes.get("arm"))
        is_control = raw.get("is_control_arm", notes.get("is_control_arm"))
        if is_control is None:
            # An artifact that does not say is treated as the control arm: that
            # is the state run 1 was actually in while claiming otherwise.
            is_control = True if arm is None else ("control" in str(arm).lower())
        return cls(
            manifest_id=str(raw.get("id", "unnamed")),
            claim_class=str(raw.get("claim_class", "surrogate")),
            posterior_class=str(raw.get("posterior_class", "pseudo")),
            is_control_arm=bool(is_control),
            control_arm_of=str(arm) if arm is not None else "unstated",
            anatomy_is_biological=bool(anat.get("is_biological", False)),
            anatomy_provenance=str(anat.get("provenance", "unstated")),
            gates=gates,
            weights_trained=bool(raw.get("weights_trained", False)),
            port_contract_digest=str(raw.get("port_contract_digest", "")),
            raw=dict(raw),
        )


# --------------------------------------------------------------------------
# the verdict
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class AdmissionCondition:
    """One named check, its outcome, and what it looked at."""

    code: str
    name: str
    passed: bool
    detail: str
    #: Purposes for which this condition is required.  A condition not required
    #: for the requested purpose is reported with ``required=False`` rather than
    #: omitted, so the record shows what was *not* asked.
    required: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "passed": self.passed,
            "required": self.required,
            "detail": self.detail,
        }


class CheckpointRefused(RuntimeError):
    """Admission refused.  Names every condition that failed.

    Raised at **load**, before a consumer can hold a service object at all, so
    that there is no window in which an inadmissible checkpoint is usable.
    """

    def __init__(
        self, purpose: str, failed: Sequence[AdmissionCondition], *,
        designation: str = "",
    ) -> None:
        lines = [f"  - [{c.code}] {c.name}: {c.detail}" for c in failed]
        super().__init__(
            f"SC-WBD refuses to serve {designation or 'this checkpoint'} for "
            f"purpose {purpose!r}; {len(failed)} condition(s) failed:\n"
            + "\n".join(lines)
        )
        self.purpose = purpose
        self.failed = tuple(failed)
        self.codes = tuple(c.code for c in failed)


@dataclass(frozen=True)
class AdmissionVerdict:
    """The full record of an admission decision, pass or fail."""

    purpose: str
    admitted: bool
    conditions: tuple[AdmissionCondition, ...]
    claims: CheckpointClaims
    invariants: ConsumerInvariants
    authorization: LiveUseAuthorization | None
    as_of: date

    @property
    def failed(self) -> tuple[AdmissionCondition, ...]:
        return tuple(c for c in self.conditions if c.required and not c.passed)

    def canonical(self) -> dict[str, Any]:
        return {
            "purpose": self.purpose,
            "admitted": self.admitted,
            "as_of": self.as_of.isoformat(),
            "conditions": [c.as_dict() for c in self.conditions],
            "manifest_id": self.claims.manifest_id,
            "invariants": self.invariants.as_dict(),
            "authorization": (
                self.authorization.canonical() if self.authorization else None
            ),
            "authorization_hash": (
                self.authorization.record_hash() if self.authorization else ""
            ),
        }

    def content_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.canonical(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def raise_if_refused(self, designation: str = "") -> "AdmissionVerdict":
        if not self.admitted:
            raise CheckpointRefused(
                self.purpose, self.failed, designation=designation
            )
        return self


# Which conditions each purpose requires.  ``A0`` is required by every purpose
# and is not listed.
_REQUIRED_BY_PURPOSE: Mapping[str, frozenset[str]] = {
    "simulation": frozenset(),
    "research_offline": frozenset({"A1"}),
    "live_hardware": frozenset({"A1", "A2", "A3", "A4", "A5", "A6"}),
    "patient_directed": frozenset({"A1", "A2", "A3", "A4", "A5", "A6"}),
}


def admit(
    claims: CheckpointClaims,
    *,
    purpose: str,
    live_use_authorization: LiveUseAuthorization | None = None,
    invariants: ConsumerInvariants | None = None,
    as_of: date | None = None,
    designation: str = "",
    raise_on_refusal: bool = True,
) -> AdmissionVerdict:
    """Decide whether ``claims`` may be served for ``purpose``.

    Refuses by raising :class:`CheckpointRefused` unless ``raise_on_refusal``
    is ``False``, in which case the verdict is returned with
    ``admitted=False``.  The non-raising form exists so a caller can *report*
    admissibility (a dashboard, a report generator) without the report itself
    becoming a load; every serving path uses the raising form.
    """
    if purpose not in EXPORT_PURPOSES:
        raise ValueError(
            f"unknown export purpose {purpose!r}; known: {list(EXPORT_PURPOSES)}. "
            "The runtime will not admit a purpose it has no rules for"
        )
    as_of = as_of or date.today()
    inv = invariants if invariants is not None else ConsumerInvariants()
    required = _REQUIRED_BY_PURPOSE[purpose]

    conditions: list[AdmissionCondition] = []

    # A0 -- standing invariants. Required for every purpose, including
    # "simulation". Construction of ConsumerInvariants already refuses a widened
    # flag, so reaching here means they hold; the condition is recorded anyway
    # so the verdict shows it was checked rather than assumed.
    conditions.append(
        AdmissionCondition(
            "A0",
            "consumer standing invariants are false",
            True,
            "sim2real_ready=False, promotion_eligible=False, "
            "robot_command_authority=False; unconditional and not relaxed by "
            "any authorization record",
            required=True,
        )
    )

    # A1 -- the checkpoint states its claims at all.
    conditions.append(
        AdmissionCondition(
            "A1",
            "claim manifest present",
            claims.manifest_id != "absent",
            (
                f"manifest {claims.manifest_id!r} (claim_class="
                f"{claims.claim_class!r}, posterior_class="
                f"{claims.posterior_class!r})"
                if claims.manifest_id != "absent"
                else (
                    f"no {SIDECAR_NAME} beside the checkpoint. An artifact that "
                    "does not state its claim class cannot be checked against a "
                    "purpose, and absence is not permission"
                )
            ),
            required="A1" in required,
        )
    )

    # A2 -- not an ablation control arm.
    conditions.append(
        AdmissionCondition(
            "A2",
            "not an ablation control arm",
            not claims.is_control_arm,
            (
                f"declared control arm of {claims.control_arm_of!r}; "
                "reports/scope_gap.md records that SC-WBD-001-beta is the "
                "equal-capacity generic-operator control of body.tex Sec. 11.4's "
                "first ablation, and a control arm's measurements are "
                "measurements of the control, not of the model"
                if claims.is_control_arm
                else f"treatment arm ({claims.control_arm_of})"
            ),
            required="A2" in required,
        )
    )

    # A3 -- the anatomy is anatomy.
    conditions.append(
        AdmissionCondition(
            "A3",
            "anatomy is biological",
            claims.anatomy_is_biological,
            (
                f"anatomy provenance {claims.anatomy_provenance!r} with "
                "is_biological=False: the connectome is a geometry-respecting "
                "synthetic graph, not anatomy, so no prediction it produces is "
                "about any head"
                if not claims.anatomy_is_biological
                else f"anatomy provenance {claims.anatomy_provenance!r}"
            ),
            required="A3" in required,
        )
    )

    # A4 -- the gates actually ran.
    unrun = claims.unrun_gates
    failed_gates = claims.failed_gates
    conditions.append(
        AdmissionCondition(
            "A4",
            "claim gates ran and did not fail",
            not unrun and not failed_gates and bool(claims.gates),
            (
                "no gate statuses recorded; an artifact with no gate results is "
                "not an artifact whose gates passed"
                if not claims.gates
                else "; ".join(
                    p
                    for p in (
                        f"COULD_NOT_RUN: {list(unrun)}" if unrun else "",
                        f"FAIL: {list(failed_gates)}" if failed_gates else "",
                    )
                    if p
                )
                or f"{len(claims.gates)} gate(s) recorded, none COULD_NOT_RUN or FAIL"
            ),
            required="A4" in required,
        )
    )

    # A5 -- there are trained weights behind the prediction.
    conditions.append(
        AdmissionCondition(
            "A5",
            "trained weights are loaded",
            claims.weights_trained,
            (
                "predictions come from a closed-form field model and "
                "prior-specified surrogate propagators; no trained "
                "SC-WBD-001-beta weights back them"
                if not claims.weights_trained
                else "trained checkpoint discovered and hashed"
            ),
            required="A5" in required,
        )
    )

    # A6 -- live application is authorized, by record and not by date.
    if live_use_authorization is None:
        a6_pass, a6_detail = False, (
            f"purpose {purpose!r} is live application and no LiveUseAuthorization "
            "was supplied. Live use in the consumer repository is pending a "
            "preliminary review; what opens this gate is a record that the "
            "review occurred with an approving outcome, not the passing of "
            f"{EARLIEST_CREDIBLE_REVIEW.isoformat()}"
        )
    else:
        why = live_use_authorization.refusal_for(purpose, as_of=as_of)
        a6_pass = why is None
        a6_detail = why or (
            f"{live_use_authorization.review_body} "
            f"{live_use_authorization.reference}: outcome "
            f"{live_use_authorization.outcome!r} recorded "
            f"{live_use_authorization.reviewed_on.isoformat()}, scope "
            f"{list(live_use_authorization.scope)}"
        )
    conditions.append(
        AdmissionCondition(
            "A6", "live application authorized by record", a6_pass, a6_detail,
            required="A6" in required,
        )
    )

    verdict = AdmissionVerdict(
        purpose=purpose,
        admitted=all(c.passed for c in conditions if c.required),
        conditions=tuple(conditions),
        claims=claims,
        invariants=inv,
        authorization=live_use_authorization,
        as_of=as_of,
    )
    if raise_on_refusal:
        verdict.raise_if_refused(designation)
    return verdict


# --------------------------------------------------------------------------
# sidecar
# --------------------------------------------------------------------------

def read_sidecar(path: Path | str) -> CheckpointClaims:
    """Read an admission sidecar, or return the refusing defaults."""
    p = Path(path)
    if not p.is_file():
        return CheckpointClaims.absent()
    return CheckpointClaims.from_manifest(json.loads(p.read_text()))


def sidecar_from_checkpoint(
    checkpoint_path: Path | str,
    *,
    trust_checkpoint_pickle: bool = False,
    gates: Mapping[str, str] | None = None,
    is_control_arm: bool | None = None,
    control_arm_of: str = "unstated",
) -> dict[str, Any]:
    """Derive a sidecar from a checkpoint's own metadata.

    A foundation checkpoint carries everything admission needs -- ``extra
    .anatomy`` (with ``is_biological``), ``state_layout``, ``config`` -- but it
    is a pickle, and reading it requires ``weights_only=False``, i.e. executing
    whatever is inside it.  That decision is made **once, by a person, at
    emission time**, not on every consumer load: hence ``trust_checkpoint_pickle``
    must be passed explicitly, and the serving path never calls this.

    Gate statuses are **not** in the checkpoint and must be supplied; omitting
    them leaves ``gates`` empty, which condition A4 refuses.
    """
    if not trust_checkpoint_pickle:
        raise PermissionError(
            "reading a foundation checkpoint requires torch.load(..., "
            "weights_only=False), which executes the pickle. Pass "
            "trust_checkpoint_pickle=True to state that you accept that for "
            "this file. The serving path never does this: consumers read the "
            f"{SIDECAR_NAME} sidecar, which is data"
        )
    import torch  # local: keep the import cost off the serving path

    from .ports import PortContract

    ck = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    extra = ck.get("extra") or {}
    anat = dict(extra.get("anatomy") or {})
    layout = ck.get("state_layout")
    try:
        digest = PortContract.from_state_layout(layout).digest()
    except Exception:
        digest = ""
    return {
        "id": str(ck.get("model_id", "unnamed")),
        "claim_class": "surrogate",
        "posterior_class": "pseudo",
        "is_control_arm": (
            is_control_arm
            if is_control_arm is not None
            else bool(ck.get("config", {}).get("model", {}).get("local_core"))
        ),
        "arm": control_arm_of,
        "anatomy": {
            "is_biological": bool(anat.get("is_biological", False)),
            "provenance": str(anat.get("provenance", "unstated")),
            "frame": str(anat.get("frame", "unstated")),
            "n_regions": anat.get("n_regions"),
        },
        "gates": dict(gates or {}),
        "weights_trained": True,
        # The layout itself, so a consumer gets ports from data rather than
        # from the pickle. This is the whole reason the sidecar exists.
        "state_layout": layout,
        "port_contract_digest": digest,
        "derived_from": str(checkpoint_path),
        "notes": {
            "stage": ck.get("stage"),
            "step": ck.get("step"),
            "git_sha": ck.get("git_sha"),
        },
    }
