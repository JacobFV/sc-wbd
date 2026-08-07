"""What a patient actually has -- declared, never imputed.

``body.tex`` sec. 6.3: *"A training episode need only illuminate a slice of the
model... Unobserved interior modules are frozen, marginalized, or sampled from a
population posterior according to whether gradients have a defensible causal
path to them."*  The first thing a slice needs is an honest statement of what
was measured.

Two rules are enforced here and nowhere else in this package, because a rule
enforced in two places is a rule that can disagree with itself:

**Rule 1 -- missing is missing.**
    There is no accessor that returns a default, a zero, an empty tensor, or a
    population mean for a modality the patient does not have.
    :meth:`ModalityAvailability.require` raises :class:`MissingModalityError`.
    A caller that wants to branch on presence must say so with ``in``.

**Rule 2 -- a present modality is a declared modality.**
    ``R01`` of ``thesis_contract.tex`` rejects unknown units / clock / support /
    frame / calibration lineage.  A :class:`ModalityRecord` therefore cannot be
    constructed without a source card, a spatial support, a clock and a
    calibration record; :class:`UndeclaredModalityError` is raised otherwise.
    Whether those four are *objects* or merely *names* is recorded in
    :attr:`ModalityRecord.declaration_kind` and surfaces in the patient report,
    because a string naming a calibration manifest is not a calibration
    manifest and the reader must be able to tell the difference.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal, Mapping, Sequence

__all__ = [
    "MODALITIES",
    "Modality",
    "MissingModalityError",
    "ModalityAvailability",
    "ModalityRecord",
    "UndeclaredModalityError",
    "ZeroImputationRefused",
    "channel_contribution",
]

#: The modalities the clinic can present us with.  ``"none"`` is not a member:
#: a patient with no data is an availability with no records, which is a
#: different (and representable) thing from a modality called "none".
Modality = Literal[
    "structural_mri",
    "dmri",
    "fmri",
    "eeg",
    "meg",
    "behavior",
]

MODALITIES: tuple[Modality, ...] = (
    "structural_mri",
    "dmri",
    "fmri",
    "eeg",
    "meg",
    "behavior",
)


class MissingModalityError(KeyError):
    """Asked for a modality the patient does not have.  Rule 1."""


class UndeclaredModalityError(ValueError):
    """A modality declared present without units/support/clock/calibration (R01)."""


class ZeroImputationRefused(ValueError):
    """Something tried to substitute zeros/means for an absent modality.  Rule 1."""


#: ``modality -> (likelihood channel of the reference slice, why)``.
#:
#: The reference forward model of ``scwbd.infer.linear_gaussian`` has exactly
#: two observation operators, ``H_eeg`` (T2) and ``H_bold`` (T3).  Everything
#: else a patient may have is *real data about a real thing* that this
#: particular likelihood cannot read.  Saying so explicitly is the point of this
#: table: "contributes no channel" is a statement about the reference model, not
#: a claim that the modality is uninformative.
_CHANNEL_MAP: dict[Modality, tuple[str | None, str]] = {
    "eeg": (
        "eeg",
        "read by H_eeg (T2): instantaneous lead-field mixing of the fast state",
    ),
    "meg": (
        "eeg",
        "mapped onto the T2 fast-electrophysiology channel as a PROXY; the "
        "reference slice has no separate magnetic lead field, so a MEG-only "
        "profile is an EEG-shaped profile and is labelled as such",
    ),
    "fmri": (
        "bold",
        "read by H_bold (T3): hemodynamic cascade on the slow clock",
    ),
    "structural_mri": (
        None,
        "no observation operator in the reference dynamical slice; structural "
        "MRI individualises geometry, lead fields and anatomy-derived priors, "
        "which are outside theta = (a21, a32, a13, tau)",
    ),
    "dmri": (
        None,
        "no observation operator in the reference dynamical slice; dMRI "
        "constrains structural connectivity and tract lengths, which enter the "
        "coupling PRIOR rather than the coupling likelihood",
    ),
    "behavior": (
        None,
        "no observation operator in the reference dynamical slice; behaviour "
        "carries information about the system but this likelihood cannot read "
        "it, so it must not be counted as evidence about theta here",
    ),
}


def channel_contribution(modality: Modality) -> tuple[str | None, str]:
    """``(channel or None, reason)`` for one modality in the reference slice."""
    if modality not in _CHANNEL_MAP:
        raise KeyError(f"unknown modality {modality!r}; known: {list(MODALITIES)}")
    return _CHANNEL_MAP[modality]


def _declaration(value: Any, what: str, modality: str) -> tuple[Any, str]:
    """Normalise one declaration slot, refusing an empty one (R01)."""
    if value is None:
        raise UndeclaredModalityError(
            f"[R01] modality {modality!r} declared present without a {what}; "
            "an undeclared unit/support/clock/calibration lineage is a rejection "
            "condition, and a missing declaration must not be filled in by "
            "default. Supply the object, or a documented identifier for it."
        )
    if isinstance(value, str):
        if not value.strip():
            raise UndeclaredModalityError(
                f"[R01] modality {modality!r}: {what} is an empty string"
            )
        return value, "name"
    return value, "object"


@dataclass(frozen=True)
class ModalityRecord:
    """One modality the patient actually has, with its four declarations.

    ``source_card``/``support``/``clock``/``calibration`` accept either the
    corresponding ``scwbd.schema`` object (:class:`~scwbd.schema.sources.SourceCard`,
    :class:`~scwbd.schema.supports.Support`,
    :class:`~scwbd.schema.supports.TemporalSupport`,
    :class:`~scwbd.schema.frames.CalibrationManifest`) or a string identifying
    it.  Which one was supplied is recorded per slot.
    """

    modality: Modality
    source_card: Any
    support: Any
    clock: Any
    calibration: Any
    n_samples: int | None = None
    duration_seconds: float | None = None
    session_id: str = "s0"
    notes: tuple[str, ...] = ()
    declaration_kind: Mapping[str, str] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if self.modality not in MODALITIES:
            raise ValueError(
                f"unknown modality {self.modality!r}; known: {list(MODALITIES)}"
            )
        kinds: dict[str, str] = {}
        for slot, what in (
            ("source_card", "source card"),
            ("support", "spatial support"),
            ("clock", "clock / temporal support"),
            ("calibration", "calibration record"),
        ):
            _v, kind = _declaration(getattr(self, slot), what, self.modality)
            kinds[slot] = kind
        object.__setattr__(self, "declaration_kind", kinds)

    @property
    def fully_objectified(self) -> bool:
        """True when every declaration is a real object, not just a name."""
        return all(k == "object" for k in self.declaration_kind.values())

    @property
    def channel(self) -> str | None:
        return channel_contribution(self.modality)[0]

    def _ident(self, slot: str) -> str:
        v = getattr(self, slot)
        if isinstance(v, str):
            return v
        return str(getattr(v, "id", None) or type(v).__name__)

    def to_dict(self) -> dict[str, Any]:
        chan, why = channel_contribution(self.modality)
        return {
            "modality": self.modality,
            "session_id": self.session_id,
            "source_card": self._ident("source_card"),
            "support": self._ident("support"),
            "clock": self._ident("clock"),
            "calibration": self._ident("calibration"),
            "declaration_kind": dict(self.declaration_kind),
            "fully_objectified": self.fully_objectified,
            "n_samples": self.n_samples,
            "duration_seconds": self.duration_seconds,
            "likelihood_channel": chan,
            "channel_reason": why,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ModalityAvailability:
    """The modalities one patient actually has.

    Constructed from records; an absent modality is simply not present.  There
    is deliberately **no** ``get(modality, default)``.
    """

    patient_id: str
    records: tuple[ModalityRecord, ...] = ()
    group: str = "population"
    session_ids: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        seen: dict[tuple[str, str], int] = {}
        for r in self.records:
            key = (r.modality, r.session_id)
            seen[key] = seen.get(key, 0) + 1
        dupes = [k for k, n in seen.items() if n > 1]
        if dupes:
            raise ValueError(f"duplicate (modality, session) records: {dupes}")
        if not self.session_ids:
            ids = tuple(dict.fromkeys(r.session_id for r in self.records)) or ("s0",)
            object.__setattr__(self, "session_ids", ids)

    # -- presence ---------------------------------------------------------
    def __contains__(self, modality: object) -> bool:
        return any(r.modality == modality for r in self.records)

    def __iter__(self) -> Iterator[ModalityRecord]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def present(self) -> tuple[Modality, ...]:
        return tuple(dict.fromkeys(r.modality for r in self.records))

    @property
    def absent(self) -> tuple[Modality, ...]:
        have = set(self.present)
        return tuple(m for m in MODALITIES if m not in have)

    @property
    def is_empty(self) -> bool:
        return not self.records

    def require(self, modality: Modality, session_id: str | None = None) -> ModalityRecord:
        """The record, or :class:`MissingModalityError`.  Never a default."""
        for r in self.records:
            if r.modality == modality and (session_id is None or r.session_id == session_id):
                return r
        raise MissingModalityError(
            f"patient {self.patient_id!r} has no {modality!r}"
            + (f" for session {session_id!r}" if session_id else "")
            + f"; present: {list(self.present)}. Missing data is missing: it is "
            "not zero, not the population mean, and not an empty record. Branch "
            "on presence (`modality in availability`) and record the absence."
        )

    def records_for(self, modality: Modality) -> tuple[ModalityRecord, ...]:
        return tuple(r for r in self.records if r.modality == modality)

    # -- the reference-slice design this availability implies --------------
    @property
    def channels(self) -> tuple[str, ...]:
        """Likelihood channels of the reference slice, in canonical order."""
        chans = {r.channel for r in self.records if r.channel is not None}
        return tuple(c for c in ("eeg", "bold") if c in chans)

    @property
    def design(self) -> str:
        """The ``scwbd.infer`` design name this availability corresponds to.

        ``"prior"`` -- the design with *no* channels -- is returned when the
        patient has no modality the reference likelihood can read.  That is the
        honest encoding of an MRI-only patient: there is no fMRI record full of
        zeros, there is no record at all, and the expected information about
        ``theta`` is identically the zero matrix.
        """
        c = self.channels
        if c == ("eeg",):
            return "eeg_only"
        if c == ("bold",):
            return "fmri_only"
        if c == ("eeg", "bold"):
            return "joint_native"
        return "prior"

    @property
    def uses_meg_as_eeg_proxy(self) -> bool:
        return "meg" in self and "eeg" not in self

    def digest(self) -> str:
        """Stable content hash, so a profile can be pinned to an availability."""
        payload = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "group": self.group,
            "session_ids": list(self.session_ids),
            "present": list(self.present),
            "absent": list(self.absent),
            "channels": list(self.channels),
            "design": self.design,
            "meg_as_eeg_proxy": self.uses_meg_as_eeg_proxy,
            "records": [r.to_dict() for r in self.records],
            "notes": list(self.notes),
        }

    # -- construction helpers ---------------------------------------------
    @classmethod
    def from_modalities(
        cls,
        patient_id: str,
        modalities: Sequence[Modality],
        *,
        group: str = "population",
        session_id: str = "s0",
        prefix: str = "declared",
        **kw: Any,
    ) -> "ModalityAvailability":
        """Convenience constructor with *named* (not objectified) declarations.

        Every record it builds has ``fully_objectified == False``, so a report
        built from it says so.  This exists for profiling and for tests; a
        clinical run should pass real schema objects.
        """
        recs = tuple(
            ModalityRecord(
                modality=m,
                source_card=f"{prefix}:{patient_id}:{m}:card",
                support=f"{prefix}:{patient_id}:{m}:support",
                clock=f"{prefix}:{patient_id}:{m}:clock",
                calibration=f"{prefix}:{patient_id}:{m}:calibration",
                session_id=session_id,
            )
            for m in modalities
        )
        return cls(patient_id=patient_id, records=recs, group=group, **kw)


def refuse_zero_imputation(availability: ModalityAvailability, modality: Modality) -> None:
    """Raise if a caller is about to fabricate a record for an absent modality.

    Call this at the top of any function that would otherwise happily build an
    all-zero tensor for a channel the patient does not have.
    """
    if modality not in availability:
        raise ZeroImputationRefused(
            f"refusing to synthesise a {modality!r} record for patient "
            f"{availability.patient_id!r}, which does not have it; zero- or "
            "mean-imputing an absent modality makes an uninformative dataset "
            "look like an informative one and is forbidden."
        )
