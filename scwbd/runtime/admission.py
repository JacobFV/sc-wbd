"""The export edge: what a consumer is told about the checkpoint it holds.

Posture (``ARCHITECTURE.md`` Sec. 7a)
-------------------------------------
**Ship the artifact and label it. Never refuse to produce it.** Refusals belong
on claims in papers, not on files on disk. A precondition blocks only if it
changes what a number *means*.

This module therefore separates two things that an earlier version of it
conflated:

**Refusals (A0, A1).** Correctness, not policy.

* **A0** -- the consumer's standing invariants are false. Widening one is a
  contradiction in terms, not a judgement call.
* **A1** -- there is a readable claim manifest. Not "the manifest says good
  things": that it *parses*. A consumer that cannot read what it is holding
  cannot label it, and every label below would be a guess.

**Labels (L1, L2, L3).** Facts about the artifact that a consumer must be
*told*, loudly, and then allowed to proceed on.

* **L1** -- this is an ablation control arm (``reports/scope_gap.md``: run 1 is
  the equal-capacity generic-operator control of ``body.tex`` Sec. 11.4's first
  required ablation).
* **L2** -- the anatomy is ``is_biological: false``, a geometry-respecting
  synthetic connectome rather than anatomy.
* **L3** -- the claim gates are ``COULD_NOT_RUN`` or ``FAIL``, or no gate
  results were recorded at all.

None of L1--L3 blocks a load. Each of them changes what a number *means*, which
is exactly why the consumer has to carry them rather than be protected from
them: a control-arm number correctly labelled as a control-arm number is
useful, and the same number silently refused is not.

Every label defaults to its **worst** value. A fact the artifact does not state
is not thereby true: a checkpoint with no anatomy record is labelled as having
non-biological anatomy, and one that does not say which arm it is is labelled a
control arm -- because that is precisely the state run 1 was in while claiming
otherwise. Silence is not a clean bill of health; it is an unlabelled artifact,
and the label says so.

What is *not* here
------------------
There is no live-application gate in this module. There was one, delegating to
``scwbd.intervene.deployment``, which has since been removed from the
repository. Nothing replaced it here, deliberately: driving hardware or
informing a person's stimulation is governed in the consumer repository, whose
three standing invariants (below) are false and stay false.

Orthogonality, stated because it will otherwise be misread
----------------------------------------------------------
``sim2real_ready``, ``promotion_eligible`` and ``robot_command_authority``
remain ``False`` standing invariants of the consumer, unconditionally. No label
here relaxes them, and neither does any record anywhere else. Reading a
successful admission as "promotion eligible" crosses two unrelated boundaries.
:data:`CONSUMER_STANDING_INVARIANTS` is checked on every admission, for every
purpose, and cannot be widened -- see :class:`ConsumerInvariants`.

Claim limits
------------
Passing admission asserts that the invariants hold and the manifest parsed. It
asserts nothing about whether the model is accurate, and nothing whatsoever
about whether an intervention is safe for a person. The labels it attaches are
the artifact's own statements about itself, carried forward -- not verified.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

__all__ = [
    "ExportPurpose",
    "EXPORT_PURPOSES",
    "CONSUMER_STANDING_INVARIANTS",
    "ConsumerInvariants",
    "ConsumerInvariantViolation",
    "AdmissionCondition",
    "AdmissionLabel",
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
    #: offline ranking of hypotheses against recorded or simulated data.
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

#: The consumer's standing invariants.  Unconditional.
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
# what a checkpoint says about itself
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CheckpointClaims:
    """The admission-relevant facts, read from the sidecar beside a checkpoint.

    Every field defaults to its **worst** value.  A fact that is not stated is
    not thereby true: an artifact with no anatomy record is labelled as having
    non-biological anatomy, because that is the state we can defend.  Under the
    Sec. 7a posture these defaults drive *labels*, not refusals -- so the effect
    is that an unlabelled artifact gets loudly pessimistic labels and still
    loads, rather than being withheld.
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
    #: Gate name -> status.  ``COULD_NOT_RUN`` and ``FAIL`` are **labelled**
    #: (L3), not refused: an unsupported claim is the consumer's to carry.
    gates: Mapping[str, str] = field(default_factory=dict)
    #: Whether real weights were found and hashed.
    weights_trained: bool = False
    #: Digest of the port contract this checkpoint declares (see
    #: :mod:`scwbd.runtime.ports`).
    port_contract_digest: str = ""
    #: Whether a sidecar was found *and parsed*.  This is the only claims-side
    #: input to a refusal (A1), and it is deliberately about readability rather
    #: than content: a manifest that says bad things is fine, a manifest that
    #: cannot be read means every label would be a guess.
    manifest_readable: bool = False
    #: Whether there is a checkpoint here at all.  A1 binds only when there is:
    #: with no checkpoint the service is the analytic backend, which has no
    #: claims to mislabel and says so through ``weights_status`` and label L4.
    #: Refusing that case would break the load path's deliberate design of not
    #: failing when there is no trained artifact.
    checkpoint_present: bool = False
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
            manifest_readable=True,
            raw=dict(raw),
        )


# --------------------------------------------------------------------------
# the verdict: two refusals, three labels
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class AdmissionCondition:
    """A **refusal** check: correctness, and it blocks."""

    code: str
    name: str
    passed: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class AdmissionLabel:
    """A **label**: a fact about the artifact that does not block a load.

    ``clean`` is ``True`` when the artifact is in the unremarkable state for
    this label.  A label that is not clean is not an error -- it is the thing
    the consumer has to carry, and :meth:`AdmissionVerdict.warnings` is what
    makes it loud.
    """

    code: str
    name: str
    clean: bool
    detail: str
    #: Why this changes what a number means.  Present on every label, because a
    #: label a consumer cannot act on is decoration.
    consequence: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "clean": self.clean,
            "detail": self.detail,
            "consequence": self.consequence,
        }


class CheckpointRefused(RuntimeError):
    """Admission refused.  Names every condition that failed.

    Raised at **load**, before a consumer can hold a service object at all.
    Only A0 and A1 can produce this: an impossible invariant, or a manifest
    that cannot be read.  Nothing about the *quality* of the artifact refuses
    -- see :class:`AdmissionLabel`.
    """

    def __init__(
        self, purpose: str, failed: Sequence[AdmissionCondition], *,
        designation: str = "",
    ) -> None:
        lines = [f"  - [{c.code}] {c.name}: {c.detail}" for c in failed]
        super().__init__(
            f"SC-WBD cannot serve {designation or 'this checkpoint'} for "
            f"purpose {purpose!r}; {len(failed)} condition(s) failed:\n"
            + "\n".join(lines)
        )
        self.purpose = purpose
        self.failed = tuple(failed)
        self.codes = tuple(c.code for c in failed)


@dataclass(frozen=True)
class AdmissionVerdict:
    """The full record: what blocked, and what the consumer is being told."""

    purpose: str
    admitted: bool
    conditions: tuple[AdmissionCondition, ...]
    labels: tuple[AdmissionLabel, ...]
    claims: CheckpointClaims
    invariants: ConsumerInvariants

    @property
    def failed(self) -> tuple[AdmissionCondition, ...]:
        return tuple(c for c in self.conditions if not c.passed)

    @property
    def flagged(self) -> tuple[AdmissionLabel, ...]:
        """Labels that are not in their unremarkable state."""
        return tuple(l for l in self.labels if not l.clean)

    @property
    def label_codes(self) -> tuple[str, ...]:
        return tuple(l.code for l in self.flagged)

    @property
    def is_clean(self) -> bool:
        """True only when every label is unremarkable.  Rarely true, honestly."""
        return not self.flagged

    def warnings(self) -> tuple[str, ...]:
        """One line per flagged label, for a consumer to log verbatim."""
        return tuple(
            f"[{l.code}] {l.name}: {l.detail} -- {l.consequence}"
            for l in self.flagged
        )

    def banner(self) -> str:
        """The loud form.  Empty when there is nothing to say."""
        if not self.flagged:
            return ""
        return (
            f"SC-WBD served {self.claims.manifest_id!r} for purpose "
            f"{self.purpose!r} with {len(self.flagged)} label(s) that change "
            "what its numbers mean:\n  - " + "\n  - ".join(self.warnings())
        )

    def canonical(self) -> dict[str, Any]:
        return {
            "purpose": self.purpose,
            "admitted": self.admitted,
            "conditions": [c.as_dict() for c in self.conditions],
            "labels": [l.as_dict() for l in self.labels],
            "flagged": list(self.label_codes),
            "manifest_id": self.claims.manifest_id,
            "invariants": self.invariants.as_dict(),
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


def admit(
    claims: CheckpointClaims,
    *,
    purpose: str,
    invariants: ConsumerInvariants | None = None,
    designation: str = "",
    raise_on_refusal: bool = True,
) -> AdmissionVerdict:
    """Admit ``claims`` for ``purpose``, refusing only on A0/A1 and labelling
    everything else.

    ``ARCHITECTURE.md`` Sec. 7a: ship the artifact and label it. A precondition
    blocks only if it changes what a number *means*, and neither "which
    ablation arm is this" nor "did the gates run" changes that -- they change
    what the number **is about**, which is the consumer's problem to carry, not
    ours to hide.

    Refuses by raising :class:`CheckpointRefused` unless ``raise_on_refusal``
    is ``False``, in which case the verdict is returned with
    ``admitted=False``.
    """
    if purpose not in EXPORT_PURPOSES:
        raise ValueError(
            f"unknown export purpose {purpose!r}; known: {list(EXPORT_PURPOSES)}"
        )
    inv = invariants if invariants is not None else ConsumerInvariants()

    # -- refusals ----------------------------------------------------------
    conditions: list[AdmissionCondition] = [
        AdmissionCondition(
            "A0",
            "consumer standing invariants are false",
            True,
            "sim2real_ready=False, promotion_eligible=False, "
            "robot_command_authority=False; unconditional",
        ),
        AdmissionCondition(
            "A1",
            "a checkpoint that exists can be labelled",
            claims.manifest_readable or not claims.checkpoint_present,
            (
                f"manifest {claims.manifest_id!r} parsed"
                if claims.manifest_readable
                else "no checkpoint present; the analytic backend has no claims "
                     "to mislabel (see label L4)"
                if not claims.checkpoint_present
                else (
                    f"a checkpoint is present with no readable {SIDECAR_NAME} "
                    "beside it. This refuses -- not because an unlabelled "
                    "artifact is unsafe, but because every label below would be "
                    "a guess, and a guessed label is worse than none"
                )
            ),
        ),
    ]

    # -- labels ------------------------------------------------------------
    unrun = claims.unrun_gates
    failed_gates = claims.failed_gates
    labels: list[AdmissionLabel] = [
        AdmissionLabel(
            "L1",
            "ablation arm",
            not claims.is_control_arm,
            (
                f"control arm ({claims.control_arm_of})"
                if claims.is_control_arm
                else f"treatment arm ({claims.control_arm_of})"
            ),
            consequence=(
                "this artifact is the equal-capacity generic-operator control "
                "of body.tex Sec. 11.4's first ablation (reports/scope_gap.md); "
                "its numbers measure the control, and may not be reported as a "
                "test of the thesis"
            ) if claims.is_control_arm else "",
        ),
        AdmissionLabel(
            "L2",
            "anatomy provenance",
            claims.anatomy_is_biological,
            f"anatomy provenance {claims.anatomy_provenance!r}, "
            f"is_biological={claims.anatomy_is_biological}",
            consequence=(
                "the connectome is a geometry-respecting synthetic graph, not "
                "anatomy; no prediction it produces is about any particular "
                "head, and none is subject-specific"
            ) if not claims.anatomy_is_biological else "",
        ),
        AdmissionLabel(
            "L3",
            "claim gates",
            bool(claims.gates) and not unrun and not failed_gates,
            (
                "no gate statuses recorded"
                if not claims.gates
                else "; ".join(
                    part for part in (
                        f"COULD_NOT_RUN: {list(unrun)}" if unrun else "",
                        f"FAIL: {list(failed_gates)}" if failed_gates else "",
                    ) if part
                ) or f"{len(claims.gates)} gate(s) recorded, all clear"
            ),
            consequence=(
                "the claims this artifact would support are unsupported rather "
                "than supported: COULD_NOT_RUN is not a pending PASS, and a "
                "recorded FAIL is a measurement"
            ) if (not claims.gates or unrun or failed_gates) else "",
        ),
        AdmissionLabel(
            "L4",
            "weights",
            claims.weights_trained,
            (
                "trained checkpoint discovered and hashed"
                if claims.weights_trained
                else "no trained weights; predictions come from a closed-form "
                     "field model and prior-specified surrogate propagators"
            ),
            consequence=(
                "nothing behind these numbers was fitted to data"
            ) if not claims.weights_trained else "",
        ),
    ]

    verdict = AdmissionVerdict(
        purpose=purpose,
        admitted=all(c.passed for c in conditions),
        conditions=tuple(conditions),
        labels=tuple(labels),
        claims=claims,
        invariants=inv,
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
        # A single global `local_core` string *is* the control arm: one operator
        # for all regions is exactly `body.tex` Sec. 11.4's equal-capacity
        # generic-operator control (reports/scope_gap.md G-1), and refusal R12
        # enforces the same thing at checkpoint emission. So the derivation is
        # not a heuristic about truthiness -- it reads the one config field that
        # decides which arm the artifact is. An explicit `is_control_arm=`
        # overrides it; an artifact that declares neither stays the control arm.
        "is_control_arm": (
            is_control_arm
            if is_control_arm is not None
            else isinstance(
                ck.get("config", {}).get("model", {}).get("local_core"), str
            )
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
