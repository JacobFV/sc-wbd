"""Immutable lineage (Appendix B row 1, ARCHITECTURE.md sec. 2, rule 6).

"Group by immutable lineage identifiers before splitting and fail the run when
parentage is unresolved" is refusal R10's remedy, so lineage must be a real
graph query and not a free-text note.  ``Identity`` records persistent id,
version, file hashes, parent ids and software/container hashes;
``lineage_closure`` walks the ancestor set; ``LineageGraph`` unions everything
that shares an ancestor into the group a split must respect.
"""

from __future__ import annotations

import re
from typing import Iterable, Literal, Mapping

from pydantic import Field, model_validator

from .base import SchemaModel

__all__ = [
    "Identity",
    "LineageUnit",
    "LineageGraph",
    "LineageError",
    "UnitKind",
]

_HASH_RE = re.compile(r"^[a-f0-9]{8,64}$")


class LineageError(ValueError):
    """Raised when parentage cannot be resolved."""


UnitKind = Literal[
    "participant",
    "family",
    "site",
    "device",
    "session",
    "run",
    "trial",
    "derivative",
    "simulator_replica",
    "stimulus",
]


class Identity(SchemaModel):
    """Identity, version and provenance of a schema object or dataset snapshot.

    Appendix B rejection conditions implemented here: an unversioned mutable
    download, an unknown parent, or irreproducible preprocessing.
    """

    persistent_id: str
    version: str
    #: Release/snapshot label (DOI, tag, accession).
    release: str = ""
    #: filename -> sha256 (or other content hash) of every file in the snapshot.
    file_hashes: dict[str, str] = Field(default_factory=dict)
    #: Immutable ids of the objects this one was derived from.
    parent_ids: tuple[str, ...] = ()
    #: Hash of the code that produced the object.
    software_hash: str | None = None
    #: Hash of the execution environment (container digest).
    container_hash: str | None = None
    #: Hash of the preprocessing configuration.
    preprocessing_hash: str | None = None
    #: True when the source is fetched from a mutable location (a rejection
    #: condition: "unversioned mutable download").
    mutable_download: bool = False
    label: str = ""

    @model_validator(mode="after")
    def _check(self) -> "Identity":
        if not self.persistent_id.strip():
            raise ValueError("persistent_id must be non-empty")
        if not self.version.strip():
            raise ValueError(f"{self.persistent_id!r} must declare a version")
        if self.persistent_id in self.parent_ids:
            raise ValueError(f"{self.persistent_id!r} is listed as its own parent")
        for name, h in self.file_hashes.items():
            if not _HASH_RE.match(h):
                raise ValueError(
                    f"file hash for {name!r} is not a hex digest: {h!r}"
                )
        return self

    @property
    def is_reproducible(self) -> bool:
        """Whether the snapshot can be re-obtained bit-for-bit."""
        return (
            not self.mutable_download
            and bool(self.file_hashes)
            and self.software_hash is not None
            and self.container_hash is not None
        )

    def missing_reproducibility_fields(self) -> tuple[str, ...]:
        missing: list[str] = []
        if self.mutable_download:
            missing.append("mutable_download")
        if not self.file_hashes:
            missing.append("file_hashes")
        if self.software_hash is None:
            missing.append("software_hash")
        if self.container_hash is None:
            missing.append("container_hash")
        return tuple(missing)

    def lineage_closure(
        self,
        registry: Mapping[str, "Identity"] | None = None,
        *,
        strict: bool = False,
    ) -> frozenset[str]:
        """All ancestor ids of this object, transitively.

        ``registry`` maps ``persistent_id -> Identity``; without it only the
        declared direct parents are returned.  With ``strict=True`` an
        unresolved parent raises :class:`LineageError` - that is the
        "fail the run when parentage is unresolved" half of R10.
        """
        registry = dict(registry or {})
        seen: set[str] = set()
        stack = list(self.parent_ids)
        while stack:
            pid = stack.pop()
            if pid in seen:
                continue  # also breaks cycles
            seen.add(pid)
            parent = registry.get(pid)
            if parent is None:
                if strict:
                    raise LineageError(
                        f"parent {pid!r} of {self.persistent_id!r} is not in the "
                        "lineage registry; parentage is unresolved"
                    )
                continue
            stack.extend(parent.parent_ids)
        return frozenset(seen)

    def unresolved_parents(
        self, registry: Mapping[str, "Identity"]
    ) -> tuple[str, ...]:
        return tuple(sorted(p for p in self.parent_ids if p not in registry))


class LineageUnit(SchemaModel):
    """One groupable unit of evidence: a participant, session, run, replica...

    A simulator replica or a derived scan is *not* an independent subject.  The
    unit graph is what the split policy must group on.
    """

    id: str
    kind: UnitKind
    parent_ids: tuple[str, ...] = ()
    #: For families/relatives: relatedness is a lineage edge too.
    related_ids: tuple[str, ...] = ()
    #: True for synthetic replicas that must never be counted as independent.
    is_synthetic_replica: bool = False
    n_observations: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check(self) -> "LineageUnit":
        if self.id in self.parent_ids or self.id in self.related_ids:
            raise ValueError(f"lineage unit {self.id!r} references itself")
        return self

    @property
    def links(self) -> tuple[str, ...]:
        return tuple(self.parent_ids) + tuple(self.related_ids)


class LineageGraph:
    """Union-find over lineage units: the groups a split must not break.

    Not a pydantic model - it is a derived index built by the compiler from the
    declared units, and it is rebuilt rather than serialized.
    """

    def __init__(self, units: Iterable[LineageUnit]) -> None:
        self.units: dict[str, LineageUnit] = {u.id: u for u in units}
        self._parent: dict[str, str] = {uid: uid for uid in self.units}
        for unit in self.units.values():
            for other in unit.links:
                if other in self._parent:
                    self.union(unit.id, other)

    # -- union-find --------------------------------------------------------
    def find(self, a: str) -> str:
        if a not in self._parent:
            raise LineageError(f"unknown lineage unit {a!r}")
        root = a
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[a] != root:  # path compression
            self._parent[a], a = root, self._parent[a]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Deterministic merge order keeps group ids stable across runs.
            lo, hi = sorted((ra, rb))
            self._parent[hi] = lo

    # -- queries -----------------------------------------------------------
    def group_of(self, unit_id: str) -> str:
        return self.find(unit_id)

    def groups(self) -> dict[str, tuple[str, ...]]:
        out: dict[str, list[str]] = {}
        for uid in self.units:
            out.setdefault(self.find(uid), []).append(uid)
        return {k: tuple(sorted(v)) for k, v in sorted(out.items())}

    def unresolved_parents(self) -> tuple[tuple[str, str], ...]:
        """(unit, missing_parent) pairs. Any entry fails the run under R10."""
        bad: list[tuple[str, str]] = []
        for unit in self.units.values():
            for other in unit.links:
                if other not in self.units:
                    bad.append((unit.id, other))
        return tuple(sorted(bad))

    def ancestors(self, unit_id: str) -> frozenset[str]:
        seen: set[str] = set()
        stack = list(self.units[unit_id].parent_ids)
        while stack:
            pid = stack.pop()
            if pid in seen:
                continue
            seen.add(pid)
            parent = self.units.get(pid)
            if parent is not None:
                stack.extend(parent.parent_ids)
        return frozenset(seen)

    def crossing_assignments(
        self, assignments: Mapping[str, str]
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Lineage groups whose units land in more than one fold.

        Returns (group_id, sorted folds) for every violating group.
        """
        by_group: dict[str, set[str]] = {}
        for uid, fold in assignments.items():
            if uid not in self.units:
                raise LineageError(
                    f"fold assignment references unknown lineage unit {uid!r}"
                )
            by_group.setdefault(self.find(uid), set()).add(fold)
        return tuple(
            (g, tuple(sorted(folds)))
            for g, folds in sorted(by_group.items())
            if len(folds) > 1
        )

    def unassigned(self, assignments: Mapping[str, str]) -> tuple[str, ...]:
        return tuple(sorted(uid for uid in self.units if uid not in assignments))

    def __len__(self) -> int:
        return len(self.units)
