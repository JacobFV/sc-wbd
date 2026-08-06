"""Compiled frame and clock graphs.

"The transform graph stores each edge separately.  Paths are checked for units,
handedness, epoch, support, round-trip residual, and calibration validity;
their composed result never replaces the original edges" (thesis sec. 0.4).

These compiled views are query indices over the declared graphs.  They do not
compose numeric transforms or propagate covariance - that is agent D's
``scwbd.transforms``, which must retain the cross terms of (T5).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schema.clocks import ClockEdge, ClockSpec
from ..schema.frames import FrameEdge, FrameGraphSpec, FrameNode
from ..schema.schema import BrainSchema

__all__ = ["CompiledFrameGraph", "CompiledClockGraph", "build_frame_graph", "build_clock_graph"]


@dataclass(frozen=True)
class CompiledFrameGraph:
    """Query index over the declared frame graph."""

    spec: FrameGraphSpec
    #: Frames actually referenced by a support in this schema.
    used_frames: frozenset[str]
    root: str

    def node(self, frame: str) -> FrameNode:
        return self.spec.node(frame)

    def has(self, frame: str) -> bool:
        return self.spec.has(frame)

    def path(self, src: str, dst: str) -> tuple[tuple[FrameEdge, bool], ...] | None:
        """Edge list from ``src`` to ``dst``; None when no defensible path."""
        return self.spec.path(src, dst)

    def path_is_valid(self, src: str, dst: str) -> bool:
        """A path exists and every edge on it has a complete calibration."""
        path = self.path(src, dst)
        if path is None:
            return False
        for edge, _ in path:
            if edge.transform == "identity":
                continue
            if edge.calibration is None or not edge.calibration.is_complete:
                return False
            if not edge.roundtrip_ok:
                return False
        return True

    def edges(self) -> tuple[FrameEdge, ...]:
        return self.spec.edges

    def n_nodes(self) -> int:
        return len(self.spec.nodes)

    def summary(self) -> str:
        return (
            f"CompiledFrameGraph({self.n_nodes()} frames, "
            f"{len(self.spec.edges)} edges, root={self.root})"
        )


@dataclass(frozen=True)
class CompiledClockGraph:
    """Query index over clocks and their synchronization relations."""

    clocks: dict[str, ClockSpec]
    edges: tuple[ClockEdge, ...]
    master: str | None
    _parent: dict[str, str] = field(repr=False, default_factory=dict)

    def has(self, clock: str) -> bool:
        return str(clock) in self.clocks

    def spec(self, clock: str) -> ClockSpec:
        return self.clocks[str(clock)]

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.clocks))

    def chain_to_master(self, clock: str) -> tuple[str, ...]:
        """Clock ids from ``clock`` up to the master, inclusive."""
        out = [str(clock)]
        seen = {str(clock)}
        cur = str(clock)
        while (parent := self._parent.get(cur)) is not None:
            if parent in seen:  # cycle guard; validator forbids self-reference
                break
            out.append(parent)
            seen.add(parent)
            cur = parent
        return tuple(out)

    def to_master(self, clock: str, t: float) -> float:
        """Map a timestamp on ``clock`` onto the master clock."""
        for cid in self.chain_to_master(clock):
            spec = self.clocks[cid]
            if spec.is_master:
                break
            t = spec.to_reference(t)
        return t

    def between(self, src: str, dst: str, t: float) -> float:
        """Map a timestamp from ``src`` to ``dst`` via the master."""
        t_master = self.to_master(src, t)
        chain = [c for c in self.chain_to_master(dst) if not self.clocks[c].is_master]
        for cid in reversed(chain):
            t_master = self.clocks[cid].from_reference(t_master)
        return t_master

    def unverified(self) -> tuple[str, ...]:
        """Non-master clocks whose relation to their reference is not evidenced."""
        from ..schema.clocks import UNVERIFIED_SYNC

        return tuple(
            sorted(
                cid
                for cid, c in self.clocks.items()
                if not c.is_master and c.sync_evidence in UNVERIFIED_SYNC
            )
        )

    def orphans(self) -> tuple[str, ...]:
        """Clocks whose reference is not declared."""
        return tuple(
            sorted(
                cid
                for cid, c in self.clocks.items()
                if c.reference is not None and str(c.reference) not in self.clocks
            )
        )

    def summary(self) -> str:
        return f"CompiledClockGraph({len(self.clocks)} clocks, master={self.master})"


def build_frame_graph(schema: BrainSchema) -> CompiledFrameGraph:
    used: set[str] = set()
    for region in schema.regions:
        for comp in region.state.components.values():
            used.add(str(comp.support.frame))
        for port in region.ports:
            used.add(str(port.support.frame))
    for source in schema.sources:
        used.add(str(source.spatial.frame))
    for ref in (r for region in schema.regions for r in region.atlas_refs):
        used.add(str(ref.frame))
    return CompiledFrameGraph(
        spec=schema.frames,
        used_frames=frozenset(used),
        root=str(schema.frames.root),
    )


def build_clock_graph(schema: BrainSchema) -> CompiledClockGraph:
    clocks = {str(c.id): c for c in schema.clocks}
    parent = {
        cid: str(c.reference) for cid, c in clocks.items() if c.reference is not None
    }
    masters = sorted(cid for cid, c in clocks.items() if c.is_master)
    return CompiledClockGraph(
        clocks=clocks,
        edges=tuple(schema.clock_edges),
        master=masters[0] if masters else None,
        _parent=parent,
    )
