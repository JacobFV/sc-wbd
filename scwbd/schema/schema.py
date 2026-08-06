"""``BrainSchema``: the declarative program the compiler consumes.

Thesis sec. 2.3: the schema describes "anatomical entities, nested regions,
state fields, ports, pathways, clocks, observation roles, interventions, and
confidence", and a compiler emits packed latent memory, resolution contracts,
block-sparse masks and typed dispatch, multirate synchronization, loss roles,
observation/intervention adapters, and a serialized state schema.
"""

from __future__ import annotations

from typing import Iterator, Mapping

from pydantic import Field, JsonValue, model_validator

from .base import SchemaModel
from .clocks import ClockEdge, ClockSpec
from .frames import FrameGraphSpec
from .lineage import Identity, LineageGraph, LineageUnit
from .operators import OperatorSpec
from .poset import ResolutionPoset
from .regions import Region
from .sources import SourceCard
from .state import Port

__all__ = ["BrainSchema", "SCHEMA_VERSION"]

#: ARCHITECTURE.md header: schema version of this contract.
SCHEMA_VERSION = "scwbd-schema/1.0.0"


class BrainSchema(SchemaModel):
    """The full declaration: regions, operators, scales, clocks, frames, sources."""

    version: str = SCHEMA_VERSION
    regions: list[Region] = Field(default_factory=list)
    operators: list[OperatorSpec] = Field(default_factory=list)
    resolution_poset: ResolutionPoset = Field(default_factory=ResolutionPoset)
    clocks: list[ClockSpec] = Field(default_factory=list)
    frames: FrameGraphSpec
    sources: list[SourceCard] = Field(default_factory=list)

    # -- additions used by the compiler --------------------------------------
    id: str = "unnamed_schema"
    label: str = ""
    #: Lineage of the schema object itself (not of its data sources).
    identity: Identity | None = None
    #: Explicit clock-to-clock synchronization edges beyond each clock's own
    #: ``reference``.
    clock_edges: list[ClockEdge] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> "BrainSchema":
        rids = [r.id for r in self.regions]
        dupes = {i for i in rids if rids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate region ids: {sorted(dupes)}")
        known = set(rids)
        for op in self.operators:
            if op.src not in known:
                raise ValueError(f"operator {op.key} has unknown source region {op.src!r}")
            if op.dst not in known:
                raise ValueError(f"operator {op.key} has unknown target region {op.dst!r}")
        cids = [str(c.id) for c in self.clocks]
        cdupes = {i for i in cids if cids.count(i) > 1}
        if cdupes:
            raise ValueError(f"duplicate clock ids: {sorted(cdupes)}")
        sids = [s.id for s in self.sources]
        sdupes = {i for i in sids if sids.count(i) > 1}
        if sdupes:
            raise ValueError(f"duplicate source card ids: {sorted(sdupes)}")
        for region in self.regions:
            if region.parent is not None and region.parent not in known:
                raise ValueError(
                    f"region {region.id!r} declares unknown parent {region.parent!r}"
                )
        return self

    # -- indices ------------------------------------------------------------
    def region(self, rid: str) -> Region:
        for r in self.regions:
            if r.id == rid:
                return r
        raise KeyError(f"unknown region {rid!r}")

    def clock(self, cid: str) -> ClockSpec:
        for c in self.clocks:
            if str(c.id) == str(cid):
                return c
        raise KeyError(f"unknown clock {cid!r}")

    def source(self, sid: str) -> SourceCard:
        for s in self.sources:
            if s.id == sid:
                return s
        raise KeyError(f"unknown source card {sid!r}")

    def operator(self, key: str) -> OperatorSpec:
        for op in self.operators:
            if op.key == key:
                return op
        raise KeyError(f"unknown operator {key!r}")

    def region_ids(self) -> tuple[str, ...]:
        return tuple(r.id for r in self.regions)

    def clock_ids(self) -> tuple[str, ...]:
        return tuple(str(c.id) for c in self.clocks)

    def has_clock(self, cid: str) -> bool:
        return str(cid) in self.clock_ids()

    def iter_ports(self) -> Iterator[tuple[Region, Port]]:
        for region in self.regions:
            for port in region.ports:
                yield region, port

    def ports_with_role(self, role: str) -> tuple[tuple[Region, Port], ...]:
        return tuple((r, p) for r, p in self.iter_ports() if p.role == role)

    def port(self, qualified: str) -> tuple[Region, Port]:
        """Look up ``"region_id.port_name"``."""
        rid, _, pname = qualified.partition(".")
        region = self.region(rid)
        return region, region.port(pname)

    # -- lineage ------------------------------------------------------------
    def lineage_registry(self) -> Mapping[str, Identity]:
        """persistent_id -> Identity over sources and the schema itself."""
        reg: dict[str, Identity] = {}
        if self.identity is not None:
            reg[self.identity.persistent_id] = self.identity
        for s in self.sources:
            reg[s.identity.persistent_id] = s.identity
        return reg

    def lineage_units(self) -> tuple[LineageUnit, ...]:
        seen: dict[str, LineageUnit] = {}
        for s in self.sources:
            for u in s.population.units:
                seen.setdefault(u.id, u)
        return tuple(seen[k] for k in sorted(seen))

    def lineage_graph(self) -> LineageGraph:
        return LineageGraph(self.lineage_units())

    # -- convenience --------------------------------------------------------
    def numel(self) -> int:
        return sum(r.numel() for r in self.regions)

    def nbytes(self) -> int:
        return sum(r.nbytes() for r in self.regions)

    def all_ledgers(self) -> tuple[tuple[str, object], ...]:
        """(object path, ledger) for every ledger in the schema."""
        out: list[tuple[str, object]] = []
        for r in self.regions:
            if r.ledger is not None:
                out.append((f"region:{r.id}", r.ledger))
            for name, comp in r.state.components.items():
                if comp.ledger is not None:
                    out.append((f"region:{r.id}.state.{name}", comp.ledger))
            for p in r.ports:
                if p.ledger is not None:
                    out.append((f"port:{r.id}.{p.name}", p.ledger))
        for op in self.operators:
            out.append((f"operator:{op.key}", op.ledger))
        for s in self.sources:
            out.append((f"source:{s.id}", s.ledger))
        for pair in self.resolution_poset.maps:
            for slot in ("restriction", "prolongation"):
                m = getattr(pair, slot)
                if m is not None and m.ledger is not None:
                    out.append((f"scale_map:{pair.fine}->{pair.coarse}:{slot}", m.ledger))
        for e in self.frames.edges:
            if e.ledger is not None:
                out.append((f"frame_edge:{e.src}->{e.dst}", e.ledger))
        return tuple(out)
