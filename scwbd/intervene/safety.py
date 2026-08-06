""":math:`\\mathcal A_{\\rm safe}` as a refusal mechanism, not a permission.

**SIMULATION ONLY.**  Nothing in this module authorises stimulating anybody.
:math:`\\mathcal A_{\\rm safe}` here is a *feasible set that blocks a simulated
optimizer*.  Passing every check in this module means only that the optimizer
was not blocked on those axes; it does not mean an intervention is safe,
approved, or applicable to a person.  Applying a plan to a person or to real
hardware is gated by :mod:`scwbd.intervene.deployment`, which is checked here
whenever a proposal declares ``application="live"``.

Design, following thesis Sec. 7.4:

* Limits live in a **declarative, citable file** (``limits/a_safe.toml``) and
  are **never learned**.  :class:`SafetyLimits` refuses mutation, refuses to
  hand parameters to an optimizer, and refuses to load a limit that has no
  citation.
* The feasible set sits **outside the learned objective**.  The optimizer
  never sees :math:`\\mathcal A_{\\rm safe}` as a penalty it can trade away; it
  sees a hard filter that raises :class:`CompilerRefusal` with ``code="R11"``.
* The controller can answer :class:`Defer` -- a safer measurement or a
  reversible probe -- and, per ``thesis_contract.tex`` Sec. 0.5 step 6, *must*
  do so when model disagreement or transform uncertainty dominates the
  estimated benefit difference.
* :class:`NoRecommendation` is a first-class outcome.

Governance (added with the R11 gate)
------------------------------------
:class:`AuthorizationGate` admits a request only when a validated
:class:`~scwbd.schema.authorization.AuthorizationRecord` covers the requested
intervention class at the requested time **and** the proposal is inside
:math:`\\mathcal A_{\\rm safe}`.  The ordering matters and is one-way:

* an authorization **never widens** :math:`\\mathcal A_{\\rm safe}`.  The
  feasible set is loaded from the same declarative, citable file either way,
  and :meth:`FeasibleSet.contains` does not take an authorization argument, so
  there is no code path by which a record could relax a bound.  What an
  authorization does is permit operating *within* limits the protocol itself
  declares;
* an authorized proposal outside :math:`\\mathcal A_{\\rm safe}` still refuses
  ``R11``.  ``tests/intervene/test_authorization_gate.py`` proves this
  explicitly, because a gate nobody has seen refuse is indistinguishable from
  one that cannot.

A validated record is a *recorded declaration* of authorization, never
verification that one exists; see
:mod:`scwbd.schema.authorization` for the full claim limit.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

import torch
from torch import Tensor

from ..schema.authorization import (
    AUTHORIZATION_DECLARATION_NOTICE,
    AuthorizationRecord,
    AuthorizationVerdict,
    validate_authorization,
)
from .base import SIMULATION_ONLY_NOTICE, InterventionRefusal, Ledger
from .deployment import (
    PRELIMINARY_REVIEW_SCHEDULED,
    ApplicationMode,
    LiveApplicationVerdict,
    PreliminaryReviewRecord,
    authorize_live_application,
)

__all__ = [
    "AUTHORIZATION_DECLARATION_NOTICE",
    "PRELIMINARY_REVIEW_SCHEDULED",
    "ApplicationMode",
    "LiveApplicationVerdict",
    "PreliminaryReviewRecord",
    "authorize_live_application",
    "AuthorizationGate",
    "AuthorizationRecord",
    "AuthorizationVerdict",
    "AuthorizedRequest",
    "AuthorizedProposal",
    "CompilerRefusal",
    "SafetyLimits",
    "LimitSpec",
    "Violation",
    "SafetyVerdict",
    "ProposedIntervention",
    "FeasibleSet",
    "Defer",
    "NoRecommendation",
    "SimulatedRanking",
    "RiskSensitiveController",
    "DEFAULT_LIMITS_PATH",
]

DEFAULT_LIMITS_PATH = Path(__file__).with_name("limits") / "a_safe.toml"


class CompilerRefusal(InterventionRefusal):
    """Refusal raised when a program leaves :math:`\\mathcal A_{\\rm safe}`.

    Named to match ``thesis_contract.tex`` Table ``tab:compiler-refusals`` and
    ``scwbd.compiler``: ``R11`` is *intervention optimization outside an
    independently validated feasible set*.
    """

    def __init__(
        self,
        code: str = "R11",
        message: str = "intervention optimization outside A_safe",
        *,
        remedy: str = (
            "restrict the search to A_safe, obtain independent safety review, "
            "and the applicable ethics and regulatory approval"
        ),
        offending_object: Any = None,
        violations: Sequence["Violation"] = (),
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(code, message, remedy=remedy, offending_object=offending_object)
        self.violations = tuple(violations)
        #: Structured detail a consumer can branch on.  The governance gate
        #: puts the specific ``AUTH_*`` failures here so that "expired" and
        #: "wrong modality" are never the same message.
        self.evidence: dict[str, Any] = dict(evidence or {})


# ---------------------------------------------------------------------------
# declarative limits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LimitSpec:
    """One declared bound on one named quantity. Immutable and citable."""

    modality: str
    quantity: str
    minimum: float | None
    maximum: float | None
    units: str
    basis: str
    citation: str

    def check(self, value: float) -> "Violation | None":
        if self.minimum is not None and value < self.minimum:
            return Violation(self, float(value), "below_minimum")
        if self.maximum is not None and value > self.maximum:
            return Violation(self, float(value), "above_maximum")
        return None

    @property
    def key(self) -> str:
        return f"{self.modality}.{self.quantity}"


@dataclass(frozen=True)
class Violation:
    limit: LimitSpec
    value: float
    kind: Literal[
        "below_minimum",
        "above_maximum",
        "missing",
        "unknown_axis",
        "undeclared_by_proposal",
    ]

    def __str__(self) -> str:
        if self.kind == "undeclared_by_proposal":
            return (
                f"{self.limit.key} is a declared limit that this proposal "
                f"supplies no value for, so it was never checked "
                f"({self.limit.citation})"
            )
        bound = self.limit.minimum if self.kind == "below_minimum" else self.limit.maximum
        return (
            f"{self.limit.key} = {self.value:g} {self.limit.units} violates "
            f"{self.kind} {bound} ({self.limit.citation})"
        )


class SafetyLimits:
    """Loaded, immutable, never-learned limits.

    Refuses in three ways:

    * ``__setattr__`` after construction -> ``CompilerRefusal`` (limits are not
      state an optimizer may write).
    * a limit entry without a ``citation`` -> refused at load time.
    * :meth:`parameters` -> refuses; there is nothing here for an optimizer or
      an autograd graph to touch.
    """

    _frozen = False

    def __init__(self, limits: Mapping[str, LimitSpec], meta: Mapping[str, Any]):
        object.__setattr__(self, "_limits", dict(limits))
        object.__setattr__(self, "_meta", dict(meta))
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name: str, value: Any) -> None:  # pragma: no cover - guard
        raise CompilerRefusal(
            "R11",
            f"safety limits are declarative and never learned; refused write to {name!r}",
            remedy="edit limits/a_safe.toml with a citation, and re-review",
            offending_object=name,
        )

    def parameters(self) -> Iterable[Tensor]:  # pragma: no cover - guard
        raise CompilerRefusal(
            "R11",
            "SafetyLimits exposes no learnable parameters; A_safe sits outside "
            "the learned objective (thesis Sec. 7.4)",
            remedy="do not place A_safe inside the optimization graph",
            offending_object="SafetyLimits.parameters",
        )

    # -- loading ------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path | None = None) -> "SafetyLimits":
        p = Path(path) if path is not None else DEFAULT_LIMITS_PATH
        with open(p, "rb") as fh:
            raw = tomllib.load(fh)

        meta = {
            k: v for k, v in raw.items() if not isinstance(v, dict)
        }
        meta["source_path"] = str(p)
        if meta.get("human_use_authorized", False):
            # This refusal stays exactly as strict under a valid
            # AuthorizationRecord.  A limits file is where *bounds* are
            # declared; it is not where authorization lives, and a file that
            # asserts its own authorization is asserting something no limits
            # file is in a position to know.  Authorization is carried by an
            # AuthorizationRecord, validated against a specific request, and
            # recorded in provenance -- see scwbd.schema.authorization.
            raise CompilerRefusal(
                "R11",
                "limits file declares human_use_authorized=true; a declarative "
                "bounds file cannot authorise anything, and A_safe is never "
                "widened by an authorization",
                remedy=(
                    "remove the flag; carry authorization in an "
                    "AuthorizationRecord validated against the specific request"
                ),
                offending_object=str(p),
            )

        limits: dict[str, LimitSpec] = {}
        for modality, block in raw.items():
            if not isinstance(block, dict):
                continue
            if modality == "decision":
                # The one namespace that is explicitly rules-for-the-controller
                # rather than bounds-on-an-exposure.  Read into meta below.
                continue
            for quantity, entry in block.items():
                if not isinstance(entry, dict):
                    continue
                if "min" not in entry and "max" not in entry:
                    # Previously this was `continue`, and it is how
                    # `protocol.reversibility` sat in this file for the whole
                    # project without ever becoming a LimitSpec: declared,
                    # cited, never loaded, never checkable, never able to fail.
                    # A bound nothing can check is decoration
                    # (reports/decorative_guards.md).  Refuse it at load so the
                    # failure is at startup rather than invisible forever.
                    raise CompilerRefusal(
                        "R11",
                        f"limit {modality}.{quantity} declares neither `min` nor "
                        "`max`, so nothing can ever check it; a bound that "
                        "cannot fire is not a bound",
                        remedy=(
                            "give it a numeric `min`/`max`, or move it under "
                            "[decision] where controller rules live"
                        ),
                        offending_object=f"{modality}.{quantity}",
                    )
                if not entry.get("citation"):
                    raise CompilerRefusal(
                        "R11",
                        f"limit {modality}.{quantity} has no citation; an "
                        "uncited limit is not an independently validated limit",
                        remedy="add `citation` and `basis` to the limits file",
                        offending_object=f"{modality}.{quantity}",
                    )
                spec = LimitSpec(
                    modality=modality,
                    quantity=quantity,
                    minimum=entry.get("min"),
                    maximum=entry.get("max"),
                    units=entry.get("units", "dimensionless"),
                    basis=entry.get("basis", ""),
                    citation=entry["citation"],
                )
                limits[spec.key] = spec
        # non-numeric declarative rules kept in meta for the controller
        meta["decision"] = raw.get("decision", {})
        return cls(limits, meta)

    # -- access -------------------------------------------------------------

    @property
    def meta(self) -> Mapping[str, Any]:
        return dict(self._meta)  # type: ignore[attr-defined]

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._limits))  # type: ignore[attr-defined]

    def get(self, key: str) -> LimitSpec:
        try:
            return self._limits[key]  # type: ignore[attr-defined]
        except KeyError as exc:
            raise CompilerRefusal(
                "R11",
                f"no declared limit for axis {key!r}; an undeclared exposure "
                "axis cannot be certified as inside A_safe",
                remedy="declare the axis in limits/a_safe.toml with a citation",
                offending_object=key,
            ) from exc

    def all_specs(self) -> tuple[LimitSpec, ...]:
        """Every loaded bound.

        Exists so a test can sweep the file rather than a hardcoded list: a
        bound added tomorrow is covered tomorrow, and one that cannot be made
        to fire fails the suite instead of passing it quietly.
        """
        return tuple(v for _, v in sorted(self._limits.items()))  # type: ignore[attr-defined]

    def for_modality(self, modality: str) -> tuple[LimitSpec, ...]:
        return tuple(
            v for k, v in sorted(self._limits.items())  # type: ignore[attr-defined]
            if v.modality == modality
        )

    def require_reversible_for_live(self) -> bool:
        """Whether a live plan must be reversible (``[decision.reversibility]``).

        Read here rather than declared and ignored: this rule previously sat in
        the file as ``[protocol.reversibility] required = true``, where the
        loader skipped it for having no numeric bound, so nothing ever read it.
        """
        return bool(
            self._meta.get("decision", {})  # type: ignore[attr-defined]
            .get("reversibility", {})
            .get("require_for_live_application", False)
        )

    def citations(self) -> tuple[str, ...]:
        return tuple(sorted({v.citation for v in self._limits.values()}))  # type: ignore[attr-defined]

    def defer_ratio(self) -> float:
        return float(
            self._meta.get("decision", {})  # type: ignore[attr-defined]
            .get("defer", {})
            .get("disagreement_dominance_ratio", 1.0)
        )

    def min_candidate_models(self) -> int:
        return int(
            self._meta.get("decision", {})  # type: ignore[attr-defined]
            .get("epistemic", {})
            .get("min_candidate_models", 2)
        )


# ---------------------------------------------------------------------------
# proposals and the feasible set
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProposedIntervention:
    """A candidate :math:`u` from a simulated optimizer.

    ``exposure`` maps declared axis names (``"tms.peak_efield_v_per_m"``, ...)
    to simulated values.  ``pose_certified`` records whether the pose passed
    :mod:`scwbd.intervene.tms.pose` (a scalp label alone is not a pose).

    ``application`` declares what the plan is *for*.  The default,
    ``"computational"``, is the ordinary path for everything in this
    repository and is not gated.  ``"live"`` declares an intent to drive real
    hardware or to apply the plan to a person, and pulls in
    :mod:`scwbd.intervene.deployment` -- which refuses until a record exists
    that the preliminary review occurred with an approving outcome.
    """

    label: str
    modality: str
    exposure: Mapping[str, float]
    pose_certified: bool = False
    reversible: bool = False
    provenance: Mapping[str, Any] = field(default_factory=dict)
    notice: str = SIMULATION_ONLY_NOTICE
    application: ApplicationMode = "computational"

    @property
    def is_live_application(self) -> bool:
        return self.application == "live"


@dataclass(frozen=True)
class SafetyVerdict:
    feasible: bool
    violations: tuple[Violation, ...]
    checked_axes: tuple[str, ...]
    unchecked_declared_axes: tuple[str, ...]
    notice: str = SIMULATION_ONLY_NOTICE

    def raise_if_infeasible(self, offending: Any = None) -> None:
        if not self.feasible:
            raise CompilerRefusal(
                "R11",
                "proposal leaves A_safe: " + "; ".join(str(v) for v in self.violations),
                offending_object=offending,
                violations=self.violations,
            )


class FeasibleSet:
    """:math:`\\mathcal A_{\\rm safe}`. A filter that refuses, never a scorer.

    Deliberately has **no** ``score``/``penalty`` method: turning the feasible
    set into a soft penalty is exactly the failure mode R11 names.
    """

    def __init__(
        self,
        limits: SafetyLimits | None = None,
        *,
        require_pose_certification: bool = True,
        require_complete_coverage: bool = False,
    ) -> None:
        self.limits = limits or SafetyLimits.load()
        self.require_pose_certification = require_pose_certification
        #: When set, an axis declared for this modality that the proposal does
        #: not supply is a violation rather than a note.  Forced on for any
        #: proposal declaring ``application="live"``, independently of this
        #: flag, so it cannot be switched off for the case that matters.
        self.require_complete_coverage = require_complete_coverage

    def contains(self, proposal: ProposedIntervention) -> SafetyVerdict:
        violations: list[Violation] = []
        declared = {s.key for s in self.limits.for_modality(proposal.modality)}
        declared |= {s.key for s in self.limits.for_modality("protocol")}
        checked: list[str] = []

        for axis, value in proposal.exposure.items():
            spec = self.limits.get(axis)  # refuses on undeclared axis
            checked.append(axis)
            v = spec.check(float(value))
            if v is not None:
                violations.append(v)

        unchecked = tuple(sorted(declared - set(checked)))

        # A declared axis the proposal simply omits is reported, not violated
        # -- for a simulated study that is right, because most axes have no
        # producer for most proposals.  For a *live* plan it is exactly wrong:
        # a plan could pass by supplying the three axes it is comfortable with
        # and omitting the thermal ones.  `tfus.cem43_minutes` and
        # `tfus.temperature_rise_c` have no producer anywhere in `scwbd`, so
        # under the old rule a live tFUS plan was silently unchecked on
        # thermal dose. Requiring complete coverage turns that silence into a
        # refusal that names the missing axes.
        if (self.require_complete_coverage or proposal.is_live_application) and unchecked:
            for axis in unchecked:
                spec = self.limits.get(axis)
                violations.append(Violation(spec, float("nan"), "undeclared_by_proposal"))

        if (
            proposal.is_live_application
            and self.limits.require_reversible_for_live()
            and not proposal.reversible
        ):
            violations.append(
                Violation(
                    LimitSpec(
                        modality=proposal.modality,
                        quantity="reversibility",
                        minimum=1.0,
                        maximum=None,
                        units="dimensionless",
                        basis=(
                            "a plan applied to a person must be reversible; the "
                            "controller must retain the option of a reversible "
                            "probe or a safer measurement"
                        ),
                        citation="body.tex Sec. 7.4; thesis_contract.tex Sec. 0.5 step 6",
                    ),
                    0.0,
                    "below_minimum",
                )
            )

        if self.require_pose_certification and not proposal.pose_certified:
            violations.append(
                Violation(
                    LimitSpec(
                        modality=proposal.modality,
                        quantity="pose_certified",
                        minimum=1.0,
                        maximum=None,
                        units="dimensionless",
                        basis=(
                            "a target is not a scalp label; the full device pose "
                            "and its transform chain must be certified"
                        ),
                        citation="body.tex Sec. 2.8 / 7.2; thesis_contract Sec. 0.5 step 2",
                    ),
                    0.0,
                    "below_minimum",
                )
            )

        return SafetyVerdict(
            feasible=not violations,
            violations=tuple(violations),
            checked_axes=tuple(checked),
            unchecked_declared_axes=unchecked,
        )

    def guard(self, proposal: ProposedIntervention) -> ProposedIntervention:
        """Pass a proposal through, or raise ``CompilerRefusal(code='R11')``."""
        self.contains(proposal).raise_if_infeasible(offending=proposal.label)
        return proposal

    def guard_optimizer(
        self,
        propose: Callable[[], ProposedIntervention],
        *,
        max_attempts: int = 1,
    ) -> ProposedIntervention:
        """Wrap a simulated optimizer so a proposal outside A_safe is blocked.

        There is no repair loop by design: silently projecting an infeasible
        proposal back into the set would hide that the objective wanted to
        leave it.  ``max_attempts > 1`` only re-queries the optimizer and the
        final failure still refuses.
        """
        last: SafetyVerdict | None = None
        for _ in range(max(1, max_attempts)):
            proposal = propose()
            last = self.contains(proposal)
            if last.feasible:
                return proposal
        assert last is not None
        last.raise_if_infeasible(offending="optimizer proposal")
        raise AssertionError("unreachable")


# ---------------------------------------------------------------------------
# the governance gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuthorizedRequest:
    """A request to compare hypotheses *under a named protocol*.

    Carries the declaration, what is being asked for, and when.  All three are
    needed: a record authorising ``tms`` does not admit a ``tfus`` request, and
    an approval that has expired does not admit a request today.
    """

    record: AuthorizationRecord | None
    intervention_class: str
    at_time_s: float | None
    purpose: str = "offline hypothesis comparison"
    #: Present only when the request is for live application.  Absent is the
    #: normal case and is why a live proposal refuses by default.
    review: PreliminaryReviewRecord | None = None

    def __str__(self) -> str:
        rid = self.record.id if self.record is not None else "<no record>"
        return f"AuthorizedRequest({self.intervention_class} under {rid})"


@dataclass(frozen=True)
class AuthorizedProposal:
    """A proposal that passed *both* gates, with the provenance it now carries.

    Existence of one of these means: a declaration validated for this class at
    this time, **and** the exposure was inside :math:`\\mathcal A_{\\rm safe}`.
    It is still not permission to stimulate anybody -- this repository builds
    no stimulation controller, no device command path and no dosing
    computation.  It is an offline hypothesis comparison labelled with the
    protocol it was performed under.
    """

    proposal: ProposedIntervention
    verdict: SafetyVerdict
    authorization: AuthorizationVerdict
    notice: str = SIMULATION_ONLY_NOTICE
    #: The live-application verdict.  For the ordinary computational path this
    #: is an admitted ``mode="computational"`` verdict, which records in
    #: provenance that the plan was *not* for live use -- rather than leaving a
    #: reader to infer it from an absence.
    application: LiveApplicationVerdict | None = None

    @property
    def claim_scope(self) -> str:
        return self.authorization.claim_scope

    def provenance(self) -> dict[str, Any]:
        """What any emitted artifact must record about this comparison."""
        return {
            "claim_scope": self.claim_scope,
            "authorization": self.authorization.as_provenance(),
            "application": (
                self.application.as_provenance() if self.application else None
            ),
            "a_safe_axes_checked": list(self.verdict.checked_axes),
            "a_safe_axes_unchecked": list(self.verdict.unchecked_declared_axes),
            "proposal_label": self.proposal.label,
            "modality": self.proposal.modality,
            "notice": self.notice,
            "authorization_notice": AUTHORIZATION_DECLARATION_NOTICE,
        }


class AuthorizationGate:
    """Three gates in series, in this order, none able to excuse another.

    1. **Governance.** A validated :class:`AuthorizationRecord` must cover the
       requested intervention class at the requested time, with consent that
       covers that class, a declared device regulatory status, a declared
       enrollment scope, a named responsible investigator, and an ``A_safe``
       attributable to *that* protocol.  Anything missing, expired or
       out-of-scope refuses ``R11`` with its own specific reason.
    2. **:math:`\\mathcal A_{\\rm safe}`.** The proposal must be inside the
       independently declared feasible set.  This check is *identical* whether
       or not an authorization is present -- :meth:`FeasibleSet.contains` is
       not passed the record and could not use it -- so an authorized request
       outside the set refuses exactly as an unauthorized one does.
    3. **Live application.** If and only if the proposal declares
       ``application="live"``, :mod:`scwbd.intervene.deployment` must admit it:
       there must be a record that the preliminary review *occurred* with an
       approving outcome covering this class.  Gate 1 is necessary here and
       explicitly not sufficient -- an authorization covering computational
       work cannot unlock a live plan, because the two records answer different
       questions and this gate asks both.

    Ordering is one-way in a second sense worth stating: relaxing gate 1 (which
    is what happened when ``R11`` stopped being an unconditional constant)
    cannot relax gates 2 or 3.  Neither of them is passed the authorization
    record, and neither could use it.

    What this gate cannot do, stated so nobody has to infer it: it cannot
    verify that an IRB approval exists, it cannot verify that a preliminary
    review occurred, it cannot make a limit safe, and it cannot turn a
    simulation into evidence about a person.  Both records it consumes are
    *declarations*.
    """

    def __init__(
        self,
        feasible_set: FeasibleSet | None = None,
        *,
        a_safe_id: str | None = None,
    ) -> None:
        self.feasible_set = feasible_set or FeasibleSet()
        #: When set, the record's A_safe attribution must name this feasible
        #: set, so "we are approved" cannot inherit a generic default's bounds.
        self.a_safe_id = a_safe_id

    # -- governance only ----------------------------------------------------
    def check_authorization(
        self, request: AuthorizedRequest, *, required_axes: Sequence[str] = ()
    ) -> AuthorizationVerdict:
        """Validate the declaration. Returns a verdict; never raises for invalidity."""
        return validate_authorization(
            request.record,
            intervention_class=request.intervention_class,
            at_time_s=request.at_time_s,
            a_safe_id=self.a_safe_id,
            required_a_safe_axes=tuple(required_axes),
            what=request.purpose,
        )

    # -- both gates ---------------------------------------------------------
    def admit(
        self, proposal: ProposedIntervention, request: AuthorizedRequest
    ) -> AuthorizedProposal:
        """Admit, or raise ``CompilerRefusal(code="R11")`` naming which gate refused.

        Governance is checked first so that an unauthorized request is refused
        for being unauthorized rather than for whatever else happens to be
        wrong with it; the feasible set is then checked unconditionally.
        """
        if proposal.modality != request.intervention_class:
            raise CompilerRefusal(
                "R11",
                (
                    f"proposal modality {proposal.modality!r} does not match the "
                    f"authorized request class {request.intervention_class!r}; an "
                    "authorization is checked against the class actually proposed"
                ),
                remedy="request the class you propose",
                offending_object=proposal.label,
            )

        verdict = self.check_authorization(
            request, required_axes=tuple(sorted(proposal.exposure))
        )
        if not verdict.admitted:
            # Raised as this module's R11 flavour so that a caller catching
            # ``scwbd.intervene.safety.CompilerRefusal`` sees the governance
            # refusal and the A_safe refusal through the same door, each
            # carrying its own specific reason.
            raise CompilerRefusal(
                "R11",
                (
                    f"no validated authorization admits a "
                    f"{request.intervention_class} request: {verdict.reason()}"
                ),
                remedy=(
                    "supply a complete, in-date AuthorizationRecord whose consent "
                    "scope covers the requested intervention class and whose "
                    "A_safe is attributable to the named protocol; "
                    + "; ".join(f.remedy for f in verdict.failures if f.remedy)
                ),
                offending_object=proposal.label,
                evidence={
                    "authorization_failures": [
                        f.model_dump(mode="json") for f in verdict.failures
                    ],
                    "authorization_failure_codes": list(verdict.failure_codes),
                    "authorization_record_id": verdict.record_id,
                    "authorization_record_hash": verdict.record_hash,
                    "intervention_class": request.intervention_class,
                    "claim_scope": verdict.claim_scope,
                },
            )

        # A_safe second, and identically to the unauthorized path.
        safety = self.feasible_set.contains(proposal)
        safety.raise_if_infeasible(offending=proposal.label)

        # Live application third.  For the ordinary computational path this
        # admits without checking anything -- simulation is not gated here, and
        # a gate that fired on simulation would be a gate everyone learns to
        # route around.
        application = authorize_live_application(
            mode=proposal.application,
            intervention_class=request.intervention_class,
            at_time_s=request.at_time_s,
            review=request.review,
            authorization=request.record,
            a_safe_id=self.a_safe_id,
            required_a_safe_axes=tuple(sorted(proposal.exposure)),
        )
        application.raise_if_refused(offending=proposal.label)

        return AuthorizedProposal(
            proposal=proposal,
            verdict=safety,
            authorization=verdict,
            application=application,
        )


# ---------------------------------------------------------------------------
# controller outcomes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Defer:
    """The controller declines to act and names a cheaper, safer next step."""

    reason: str
    suggested_action: Literal[
        "additional_calibration_measurement", "reversible_probe", "no_action"
    ]
    detail: Mapping[str, float] = field(default_factory=dict)
    notice: str = SIMULATION_ONLY_NOTICE

    def __str__(self) -> str:
        return f"Defer({self.suggested_action}): {self.reason}"


@dataclass(frozen=True)
class NoRecommendation:
    """No candidate is distinguishable; the honest output is nothing."""

    reason: str
    detail: Mapping[str, float] = field(default_factory=dict)
    notice: str = SIMULATION_ONLY_NOTICE


@dataclass(frozen=True)
class SimulatedRanking:
    """A ranking over *simulated* candidates. Never a protocol for a person.

    ``human_use_authorized`` is hard-wired ``False``; the field exists so that
    any consumer that forgets to check it will at least carry the flag.
    """

    ordered_labels: tuple[str, ...]
    objective_values: Tensor
    benefit_gap: float
    epistemic_uncertainty: float
    limits_citations: tuple[str, ...]
    ledger: Ledger = field(default_factory=Ledger)
    human_use_authorized: bool = False
    notice: str = SIMULATION_ONLY_NOTICE

    def __post_init__(self) -> None:
        if self.human_use_authorized:  # pragma: no cover - guard
            raise CompilerRefusal(
                "R11",
                "SimulatedRanking cannot be marked authorized for human use",
                offending_object=self.ordered_labels,
            )


Decision = SimulatedRanking | Defer | NoRecommendation


# ---------------------------------------------------------------------------
# risk-sensitive controller (thesis Sec. 7.4)
# ---------------------------------------------------------------------------


class RiskSensitiveController:
    """Selects across the posterior, inside :math:`\\mathcal A_{\\rm safe}` only.

    Implements

    .. math::

        u^* \\in \\arg\\max_{u\\in\\mathcal A_{\\rm safe}}
          \\mathbb E\\left[B(X_T) - \\lambda C(u) - \\beta L_{\\rm harm}\\right]
          - \\gamma\\,\\mathcal U_{\\rm epi}(u)

    with three properties the thesis requires:

    1. :math:`\\mathcal A_{\\rm safe}` is a **filter applied before** the
       objective is evaluated, not a term inside it.
    2. :math:`L_{\\rm harm}` is averaged **once** under the posterior and
       :math:`\\mathcal U_{\\rm epi}` penalises only *reducible* model
       disagreement -- aleatoric outcome risk is not double counted.
    3. If disagreement or transform uncertainty dominates the benefit
       difference, the output is :class:`Defer` or :class:`NoRecommendation`
       (``thesis_contract.tex`` Sec. 0.5 step 6).
    """

    def __init__(
        self,
        feasible_set: FeasibleSet | None = None,
        *,
        lam: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
    ) -> None:
        self.feasible_set = feasible_set or FeasibleSet()
        self.lam = lam
        self.beta = beta
        self.gamma = gamma

    def decide(
        self,
        candidates: Sequence[ProposedIntervention],
        *,
        benefit: Tensor,  # [n_models, n_candidates] posterior samples of B(X_T)
        burden: Tensor,  # [n_candidates]
        harm: Tensor,  # [n_models, n_candidates]
        model_log_weights: Tensor,  # [n_models]
        transform_uncertainty: Tensor | None = None,  # [n_candidates]
        reversible_probe_available: bool = True,
    ) -> Decision:
        if len(candidates) == 0:
            return NoRecommendation(reason="no candidates supplied")

        n_models = int(benefit.shape[0])
        if n_models < self.feasible_set.limits.min_candidate_models():
            return Defer(
                reason=(
                    f"only {n_models} response model(s) considered; the "
                    "mechanism is unresolved and comparison under a single "
                    "model is not admissible"
                ),
                suggested_action="additional_calibration_measurement",
                detail={"n_models": float(n_models)},
            )

        # 1. A_safe filter -- BEFORE the objective. Infeasible proposals do not
        #    get a score they could trade against.
        feasible: list[int] = []
        for i, c in enumerate(candidates):
            if self.feasible_set.contains(c).feasible:
                feasible.append(i)
        if not feasible:
            return NoRecommendation(
                reason="every candidate lies outside A_safe",
                detail={"n_candidates": float(len(candidates))},
            )

        idx = torch.tensor(feasible, dtype=torch.long)
        b = benefit.to(torch.float64)[:, idx]
        h = harm.to(torch.float64)[:, idx]
        c_burden = burden.to(torch.float64)[idx]
        w = torch.softmax(model_log_weights.to(torch.float64), dim=0).reshape(-1, 1)

        # 2. posterior expectation, harm averaged exactly once
        exp_b = (w * b).sum(0)
        exp_h = (w * h).sum(0)

        # reducible model disagreement only (across-model spread of the benefit)
        mean_b = exp_b.unsqueeze(0)
        u_epi = ((w * (b - mean_b) ** 2).sum(0)).sqrt()
        if transform_uncertainty is not None:
            u_epi = u_epi + transform_uncertainty.to(torch.float64)[idx]

        objective = exp_b - self.lam * c_burden - self.beta * exp_h - self.gamma * u_epi

        order = torch.argsort(objective, descending=True)
        labels = tuple(candidates[feasible[int(j)]].label for j in order)
        sorted_obj = objective[order]

        # 3. Sec. 0.5 step 6: disagreement or transform uncertainty dominating
        #    the benefit difference -> defer.
        if sorted_obj.numel() == 1:
            gap = float("inf")
        else:
            gap = float(sorted_obj[0] - sorted_obj[1])
        top_u = float(u_epi[order[0]])
        ratio = self.feasible_set.limits.defer_ratio()

        if sorted_obj.numel() > 1 and top_u >= ratio * abs(gap):
            reason = (
                f"epistemic uncertainty {top_u:.4g} dominates the benefit "
                f"difference {gap:.4g} between the top two admissible "
                f"candidates (threshold ratio {ratio:g})"
            )
            if reversible_probe_available:
                return Defer(
                    reason=reason,
                    suggested_action="reversible_probe",
                    detail={"benefit_gap": gap, "epistemic": top_u},
                )
            return Defer(
                reason=reason,
                suggested_action="additional_calibration_measurement",
                detail={"benefit_gap": gap, "epistemic": top_u},
            )

        if not math.isfinite(gap) or gap <= 0.0:
            return NoRecommendation(
                reason="admissible candidates are not distinguishable",
                detail={"benefit_gap": gap},
            )

        return SimulatedRanking(
            ordered_labels=labels,
            objective_values=sorted_obj,
            benefit_gap=gap,
            epistemic_uncertainty=top_u,
            limits_citations=self.feasible_set.limits.citations(),
            ledger=Ledger(
                variance={"model": float(u_epi.mean() ** 2)},
                bias_status="prior_specified_sensitivity",
                validity_domain={"scope": "simulation_only"},
            ),
        )
