"""Immutable lineage keys for leakage-safe grouping.

Appendix D, row "Participant or family leakage" requires that *all* sessions,
derivatives, relatives and duplicate archive records are grouped **before**
splitting.  This module defines the lineage tuple used everywhere in
``scwbd.sources`` and the refusal raised when parentage cannot be resolved
(compiler refusal ``R10`` in ``ARCHITECTURE.md`` §2).

The lineage is a strict hierarchy::

    family > participant > site > device > session > run > trial

``family`` sits above ``participant`` because relatives must not be split
apart.  ``site`` and ``device`` are *not* above ``participant`` in the
ownership sense (a participant may in principle visit two sites) but they are
recorded so that leave-site-out / leave-device-out evaluation is expressible.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping

UNKNOWN = "unknown"

#: Ordered from most inclusive (never split apart) to most granular.
LINEAGE_ORDER: tuple[str, ...] = (
    "family",
    "participant",
    "site",
    "device",
    "session",
    "run",
    "trial",
)


class LineageError(RuntimeError):
    """Raised when lineage/parentage cannot be resolved.

    This is refusal ``R10`` ("derived scans/sessions/relatives/replicas
    crossing a parent-level holdout").  It is deliberately loud: a split whose
    parentage is unresolved is not a split, it is a guess.
    """

    code = "R10"

    def __init__(self, message: str, *, offending_object: Any = None, remedy: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.offending_object = offending_object
        self.remedy = remedy or (
            "Populate the missing lineage key on the record (or declare it in the "
            "source card's population block) before calling any splitter."
        )

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"[{self.code}] {self.message} (remedy: {self.remedy})"


@dataclass(frozen=True)
class Lineage:
    """Immutable provenance of one record.

    Parameters
    ----------
    participant, family, site, device, session, run, trial
        Hierarchy keys.  ``None`` means "not applicable at this level"
        (e.g. a recording that is not divided into trials);
        the string ``"unknown"`` means "applicable but unresolved" and will
        trigger :class:`LineageError` when that level (or a level above it) is
        used as a grouping key.
    derived_from
        Identifier of the *parent record* this one was derived from (a
        tractogram from a scan, a re-reference of a raw EEG file, an
        augmentation).  Derived records inherit their parent's group.
    content_hash
        Hash of the underlying bytes.  Two records with the same
        ``content_hash`` are the same datum even if their ids differ; the
        leakage audit uses this to catch duplicate archive records.
    """

    participant: str | None = None
    family: str | None = None
    site: str | None = None
    device: str | None = None
    session: str | None = None
    run: str | None = None
    trial: str | None = None
    derived_from: str | None = None
    content_hash: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    # -- construction ----------------------------------------------------
    @classmethod
    def from_mapping(cls, m: Mapping[str, Any]) -> "Lineage":
        known = {k: m.get(k) for k in (*LINEAGE_ORDER, "derived_from", "content_hash")}
        extra = {k: v for k, v in m.items() if k not in known}
        return cls(**known, extra=extra)

    def as_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in LINEAGE_ORDER}
        d["derived_from"] = self.derived_from
        d["content_hash"] = self.content_hash
        return d

    def with_(self, **kw: Any) -> "Lineage":
        return replace(self, **kw)

    # -- grouping --------------------------------------------------------
    def value_key(self, level: str) -> str:
        """Return the *bare* value at ``level`` (no ancestry prefix).

        Used for holdout over entities that are not children of a
        participant - a site or a device is shared across participants, so
        prefixing it with the participant would make every record its own
        group and silently turn leave-site-out into leave-one-run-out.
        """
        if level not in LINEAGE_ORDER:
            raise ValueError(f"unknown lineage level {level!r}; expected one of {LINEAGE_ORDER}")
        value = getattr(self, level)
        if value is None:
            raise LineageError(
                f"lineage level {level!r} is absent on this record; cannot hold out a level "
                "the source does not define",
                offending_object=self.as_dict(),
            )
        if str(value) == UNKNOWN:
            raise LineageError(
                f"lineage level {level!r} is 'unknown'; refusing to hold out an unresolved "
                f"{level}",
                offending_object=self.as_dict(),
            )
        return str(value)

    def group_key(self, level: str) -> str:
        """Return the immutable group key at ``level``.

        The key is the tuple of all lineage entries from the top of the
        hierarchy down to ``level``, so that grouping at ``session`` never
        merges two participants' sessions that happen to share a session
        label ("ses-01").

        Raises
        ------
        LineageError
            If ``level`` or any level above it is ``"unknown"`` or missing
            while being required.
        """
        if level not in LINEAGE_ORDER:
            raise ValueError(f"unknown lineage level {level!r}; expected one of {LINEAGE_ORDER}")
        idx = LINEAGE_ORDER.index(level)
        parts: list[str] = []
        for name in LINEAGE_ORDER[: idx + 1]:
            value = getattr(self, name)
            if value is None:
                # Not applicable at this level.  Permitted for levels ABOVE the
                # requested one only if the requested level itself resolves;
                # we record a sentinel so keys stay unambiguous.
                if name == level:
                    raise LineageError(
                        f"lineage level {level!r} is absent on this record; "
                        "cannot group by a level the source does not define",
                        offending_object=self.as_dict(),
                    )
                parts.append("-")
                continue
            if str(value) == UNKNOWN:
                raise LineageError(
                    f"lineage level {name!r} is 'unknown'; refusing to split records "
                    f"whose parentage at or above {level!r} is unresolved",
                    offending_object=self.as_dict(),
                )
            parts.append(str(value))
        return "/".join(parts)


@dataclass(frozen=True)
class Record:
    """A splittable unit of data (one run, one epoch file, one scan).

    ``stimulus_ids`` lists the stimuli/conditions presented in this record;
    stimulus-level holdout (Appendix D, "Stimulus memorization") groups on it.
    """

    id: str
    source_id: str
    lineage: Lineage
    stimulus_ids: tuple[str, ...] = ()
    path: str | None = None
    n_bytes: int | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)


def resolve_parentage(records: Iterable[Record]) -> dict[str, str]:
    """Map every record id to the id of its ultimate (non-derived) ancestor.

    Raises
    ------
    LineageError
        If a record declares ``derived_from`` pointing at an id that is not
        present in ``records`` (unresolved parentage -> R10), or if the
        derivation graph contains a cycle.
    """
    recs = {r.id: r for r in records}
    if len(recs) != len(list(recs)):  # pragma: no cover - dict dedups
        pass
    root: dict[str, str] = {}
    for rid, rec in recs.items():
        seen: list[str] = [rid]
        cur = rec
        while cur.lineage.derived_from is not None:
            parent_id = cur.lineage.derived_from
            if parent_id not in recs:
                raise LineageError(
                    f"record {rid!r} is derived from {parent_id!r}, which is not in the "
                    "record set; the parent may already sit in another fold",
                    offending_object=rec,
                    remedy=(
                        "Include the parent record in the split input, or record the "
                        "parent's participant/family keys on the derivative."
                    ),
                )
            if parent_id in seen:
                raise LineageError(
                    f"cycle in derivation graph: {' -> '.join(seen + [parent_id])}",
                    offending_object=rec,
                )
            seen.append(parent_id)
            cur = recs[parent_id]
        root[rid] = cur.id
    return root
