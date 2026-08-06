"""Regions and atlas references (thesis sec. 2.1, sec. 3.1)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .base import SchemaModel
from .ids import FrameId, ScaleId
from .ledger import UncertaintyLedger
from .state import Port, StateSpec

__all__ = ["AtlasRef", "Region", "Authority"]

#: Authority policy for a region's state at a given scale.  "fine" means the
#: region is measured/refined and may claim structure; "consensus" means it is
#: supported by agreement across sources; "coarse_sparse" means it exists as a
#: low-resolution placeholder and must not be advertised as fine detail.
Authority = Literal["fine", "consensus", "coarse_sparse"]


class AtlasRef(SchemaModel):
    """A reference into a named, versioned parcellation.

    Version and hash are required because "an unjustified fine-resolution
    label" and a swapped atlas revision are Appendix B rejection conditions.
    """

    atlas: str
    version: str
    index: int | None = None
    label: str = ""
    frame: FrameId
    #: sha256 of the atlas asset (agent C maintains the hashed manifest).
    asset_hash: str | None = None
    #: Fraction of this region covered by the referenced atlas unit.
    coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    ledger: UncertaintyLedger | None = None

    @model_validator(mode="after")
    def _check(self) -> "AtlasRef":
        if not self.version.strip():
            raise ValueError(f"atlas {self.atlas!r} must declare a version")
        return self


class Region(SchemaModel):
    """A modeled system owning a structured state space and typed ports."""

    id: str
    label: str
    state: StateSpec
    ports: list[Port] = Field(default_factory=list)
    atlas_refs: list[AtlasRef] = Field(default_factory=list)
    authority: Authority

    # -- additions used by the compiler --------------------------------------
    #: Enclosing region for nested/multiresolution tiles.
    parent: str | None = None
    #: Native element of the resolution poset for this region's own state.
    resolution: ScaleId | None = None
    #: Anatomical class; used by agent C to attach connectome priors.
    system: Literal[
        "cortex", "subcortex", "cerebellum", "brainstem", "body", "environment", "device"
    ] = "cortex"
    ledger: UncertaintyLedger | None = None

    @model_validator(mode="after")
    def _check(self) -> "Region":
        names = [p.name for p in self.ports]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"region {self.id!r} has duplicate ports: {sorted(dupes)}")
        return self

    def port(self, name: str) -> Port:
        for p in self.ports:
            if p.name == name:
                return p
        raise KeyError(f"region {self.id!r} has no port {name!r}")

    def ports_with_role(self, role: str) -> tuple[Port, ...]:
        return tuple(p for p in self.ports if p.role == role)

    def numel(self) -> int:
        return self.state.numel()

    def nbytes(self) -> int:
        return self.state.nbytes()
