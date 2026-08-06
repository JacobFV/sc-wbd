"""Typed operator dispatch.

"Self-dynamics and cross-region communication are dispatched from the same
typed operator registry" (thesis sec. 2.1).  Each descriptor is a fully
resolved instance: which element slices it reads, which it writes, on which
clock, with which delay prior, parameters, evidence class, mechanistic status
and ledger.  No dynamics are implemented here - agent E consumes these.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..schema.operators import OperatorSpec
from ..schema.priors import PriorBase
from ..schema.schema import BrainSchema
from .layout import StateLayout

__all__ = ["OperatorDescriptor", "Dispatch", "build_dispatch"]


@dataclass(frozen=True)
class OperatorDescriptor:
    """One instantiated operator, addressable against the state layout."""

    key: str
    src: str
    dst: str
    family: str
    evidence_class: str
    mechanistic_status: str
    clock: str
    #: component name -> element slice on the source region.
    src_slices: dict[str, slice]
    dst_slices: dict[str, slice]
    delay_prior: PriorBase
    params: dict[str, PriorBase]
    is_learned: bool
    differentiable: bool
    gating: str | None
    spec: OperatorSpec

    @property
    def is_self_edge(self) -> bool:
        return self.src == self.dst

    @property
    def parameter_group(self) -> str:
        """Named parameter group used by the gradient masks."""
        return f"operator:{self.key}:params"

    def delay_seconds(self) -> float:
        return float(self.delay_prior.mean())

    def summary(self) -> str:
        return (
            f"{self.key} [{self.family}/{self.evidence_class}/"
            f"{self.mechanistic_status}] on {self.clock}"
        )


@dataclass(frozen=True)
class Dispatch:
    """The full operator dispatch table."""

    descriptors: tuple[OperatorDescriptor, ...]

    def __iter__(self):
        return iter(self.descriptors)

    def __len__(self) -> int:
        return len(self.descriptors)

    def __getitem__(self, key: str) -> OperatorDescriptor:
        for d in self.descriptors:
            if d.key == key:
                return d
        raise KeyError(f"no operator {key!r} in dispatch")

    def keys(self) -> tuple[str, ...]:
        return tuple(d.key for d in self.descriptors)

    def on_clock(self, clock: str) -> tuple[OperatorDescriptor, ...]:
        return tuple(d for d in self.descriptors if d.clock == str(clock))

    def with_family(self, family: str) -> tuple[OperatorDescriptor, ...]:
        return tuple(d for d in self.descriptors if d.family == family)

    def learned(self) -> tuple[OperatorDescriptor, ...]:
        return tuple(d for d in self.descriptors if d.is_learned)

    def into(self, region: str) -> tuple[OperatorDescriptor, ...]:
        return tuple(d for d in self.descriptors if d.dst == region)

    def max_delay(self) -> float:
        """Longest mean conduction delay; sets the history buffer depth."""
        return max((d.delay_seconds() for d in self.descriptors), default=0.0)


def resolve_operator_clock(schema: BrainSchema, op: OperatorSpec) -> str:
    """Clock an operator is dispatched on.

    Explicit ``op.clock`` wins.  Otherwise the destination region's fastest
    component clock is used: an edge must fire at least as often as the state
    it writes, or the write is aliased.
    """
    if op.clock is not None:
        return str(op.clock)
    dst = schema.region(op.dst)
    candidates = [
        (c.temporal.dt, str(c.temporal.clock)) for c in dst.state.components.values()
    ]
    if not candidates:  # pragma: no cover - StateSpec forbids empty components
        raise ValueError(f"region {op.dst!r} has no components to derive a clock from")
    return min(candidates)[1]


def _slices(layout: StateLayout, region: str, via_port: bool) -> dict[str, slice]:
    """Element slices the operator may touch on one side.

    Without a named port the operator addresses the region's whole state.  With
    one, it is restricted to the region's *boundary* components: a port is an
    exchange contract, and an edge routed through a port must not reach into
    private state (thesis sec. 2.1, sec. 2.3).
    """
    entries = layout.of_region(region)
    if via_port:
        entries = tuple(e for e in entries if e.boundary)
    return {e.component: e.elem_slice for e in entries}


def build_dispatch(schema: BrainSchema, layout: StateLayout) -> Dispatch:
    descriptors: list[OperatorDescriptor] = []
    for op in schema.operators:
        # Validate the port names exist before using them.
        if op.src_port is not None:
            schema.region(op.src).port(op.src_port)
        if op.dst_port is not None:
            schema.region(op.dst).port(op.dst_port)
        descriptors.append(
            OperatorDescriptor(
                key=op.key,
                src=op.src,
                dst=op.dst,
                family=op.family,
                evidence_class=op.evidence_class,
                mechanistic_status=op.mechanistic_status,
                clock=resolve_operator_clock(schema, op),
                src_slices=_slices(layout, op.src, op.src_port is not None),
                dst_slices=_slices(layout, op.dst, op.dst_port is not None),
                delay_prior=op.delay_prior,
                params=dict(op.params),
                is_learned=op.is_learned,
                differentiable=op.differentiable,
                gating=op.gating,
                spec=op,
            )
        )
    return Dispatch(descriptors=tuple(descriptors))
