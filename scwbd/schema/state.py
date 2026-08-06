"""Structured regional state and typed ports (thesis sec. 2.1, sec. 2.3).

A region owns a *structured* state: components that need not share shape, dtype,
support, clock or even representation (a field on a mesh, an array indexed by
depth and cell class, a point process, a set of particles, a distribution, or
low-dimensional sufficient statistics).

Two design points here are load-bearing for the compiler:

1. **Stable offsets.**  Every component gets a deterministic packed offset so
   that a compiled model is a memory map with an ABI, not an anonymous tensor.
   The engineering merit is plain: stable offsets make state addressable by
   name across processes and checkpoints, make block-sparse masks expressible
   as index arithmetic instead of gathers, and let two implementations that
   share a schema exchange boundary state without re-encoding it.
2. **Boundary vs private state.**  A module "can expose a coarse boundary state
   while retaining a much larger private state" (sec. 2.1).  ``boundary=True``
   marks the components that are part of the exchange contract; everything else
   is private and may be reorganized by an implementation at will.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, model_validator

from .base import SchemaModel
from .ids import ScaleId
from .ledger import UncertaintyLedger
from .supports import Support, TemporalSupport
from .units import Unit

__all__ = [
    "ComponentSpec",
    "StateSpec",
    "Port",
    "DTYPE_BYTES",
    "ComponentKind",
    "PortDirection",
    "PortRole",
]

#: Byte widths for the dtypes the numerical contract (ARCHITECTURE.md sec. 3)
#: admits.  Default is float32; bfloat16 is permitted only inside learned
#: operators and never in solvers, Fisher information, or covariance.
DTYPE_BYTES: dict[str, int] = {
    "float32": 4,
    "float64": 8,
    "bfloat16": 2,
    "int64": 8,
    "int32": 4,
    "bool": 1,
}

ComponentKind = Literal[
    "sheet",
    "layer",
    "population",
    "frequency",
    "memory",
    "metabolic",
    "uncertainty",
    "field",
    "channels",
    "event",
    "latent",
]

PortDirection = Literal["in", "out", "bidirectional"]
PortRole = Literal["observation", "intervention", "boundary", "internal", "readout"]


class ComponentSpec(SchemaModel):
    """One typed component of a region's state."""

    kind: ComponentKind
    shape: tuple[int, ...]
    units: Unit
    support: Support
    temporal: TemporalSupport
    dtype: Literal["float32", "float64", "bfloat16", "int64", "int32", "bool"] = "float32"
    representation: Literal[
        "dense",
        "sparse",
        "point_process",
        "distribution",
        "particles",
        "sufficient_statistics",
        "graph",
    ] = "dense"
    #: Part of the region's exchange contract (see module docstring).
    boundary: bool = False
    #: Element of the resolution poset this component lives at.
    resolution: ScaleId | None = None
    ledger: UncertaintyLedger | None = None
    description: str = ""

    @model_validator(mode="after")
    def _check(self) -> "ComponentSpec":
        if not self.shape:
            raise ValueError("component shape must have at least one axis")
        if any(n <= 0 for n in self.shape):
            raise ValueError(f"component shape must be positive, got {self.shape}")
        return self

    def numel(self) -> int:
        return int(math.prod(self.shape))

    def nbytes(self) -> int:
        return self.numel() * DTYPE_BYTES[self.dtype]


class StateSpec(SchemaModel):
    """A region's structured state. Components need not share shape."""

    #: "sheet","layer","population","frequency","memory","metabolic","uncertainty"
    components: dict[str, ComponentSpec] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> "StateSpec":
        if not self.components:
            raise ValueError("a StateSpec must declare at least one component")
        return self

    def numel(self) -> int:
        return sum(c.numel() for c in self.components.values())

    def nbytes(self) -> int:
        return sum(c.nbytes() for c in self.components.values())

    @property
    def names(self) -> tuple[str, ...]:
        """Deterministic component order: sorted by name, so layout is stable."""
        return tuple(sorted(self.components))

    def boundary_components(self) -> tuple[str, ...]:
        return tuple(n for n in self.names if self.components[n].boundary)

    def clocks(self) -> tuple[str, ...]:
        return tuple(sorted({c.temporal.clock for c in self.components.values()}))


class Port(SchemaModel):
    """A typed read/write interface of a region."""

    name: str
    state_spec: StateSpec
    support: Support
    temporal: TemporalSupport
    direction: PortDirection

    # -- additions used by the compiler --------------------------------------
    #: What the port is for. ``observation`` ports may carry likelihood
    #: factors; ``intervention`` ports are the only ones that admit causal
    #: losses (Appendix B: "Separates read heads from write operators").
    role: PortRole = "internal"
    modality: str = ""
    #: Source cards permitted to read/write this port.
    source_ids: tuple[str, ...] = ()
    ledger: UncertaintyLedger | None = None

    @model_validator(mode="after")
    def _check(self) -> "Port":
        if not self.name.strip():
            raise ValueError("port name must be non-empty")
        if self.role == "intervention" and self.direction == "out":
            raise ValueError(
                f"intervention port {self.name!r} must accept a write "
                "(direction 'in' or 'bidirectional')"
            )
        if self.role == "observation" and self.direction == "in":
            raise ValueError(
                f"observation port {self.name!r} must emit a read "
                "(direction 'out' or 'bidirectional')"
            )
        return self
