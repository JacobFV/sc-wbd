"""Packed state layout: the memory map and calling convention (thesis sec. 2.3).

The compiler emits "packed latent-memory regions and stable offsets for every
state field" plus "a serialized state schema through which compatible models
can exchange state without re-tokenizing it".

Two offset spaces are maintained, and they are not the same thing:

* **element offsets** index a single logical state vector.  They are what
  masks, block-sparse adjacency and operator slices use, and they ignore dtype.
* **byte offsets** describe a serialized buffer with per-dtype alignment.  They
  are what a second implementation reads when exchanging boundary state.

``abi_digest()`` hashes the layout so two processes can verify they share a
memory map *before* exchanging anything - a mismatch is a silent corruption
otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from ..schema.base import content_hash_of
from ..schema.schema import BrainSchema
from ..schema.state import DTYPE_BYTES

__all__ = ["LayoutEntry", "StateLayout", "build_state_layout"]


@dataclass(frozen=True, slots=True)
class LayoutEntry:
    """One region/component block in the packed state."""

    region: str
    component: str
    kind: str
    shape: tuple[int, ...]
    dtype: str
    units: str
    clock: str
    resolution: str | None
    boundary: bool
    representation: str
    #: Offset in the logical element vector.
    elem_offset: int
    numel: int
    #: Offset in the serialized byte buffer (dtype-aligned).
    byte_offset: int
    nbytes: int

    @property
    def key(self) -> tuple[str, str]:
        return (self.region, self.component)

    @property
    def qualified(self) -> str:
        return f"{self.region}.{self.component}"

    @property
    def elem_slice(self) -> slice:
        return slice(self.elem_offset, self.elem_offset + self.numel)

    @property
    def byte_slice(self) -> slice:
        return slice(self.byte_offset, self.byte_offset + self.nbytes)

    def as_dict(self) -> dict[str, object]:
        return {
            "region": self.region,
            "component": self.component,
            "kind": self.kind,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "units": self.units,
            "clock": self.clock,
            "resolution": self.resolution,
            "boundary": self.boundary,
            "representation": self.representation,
            "elem_offset": self.elem_offset,
            "numel": self.numel,
            "byte_offset": self.byte_offset,
            "nbytes": self.nbytes,
        }


@dataclass(frozen=True)
class StateLayout:
    """Stable, addressable layout of the whole compiled state."""

    entries: tuple[LayoutEntry, ...]
    total_elements: int
    total_bytes: int
    _by_key: dict[tuple[str, str], LayoutEntry] = field(repr=False, default_factory=dict)
    _by_region: dict[str, tuple[LayoutEntry, ...]] = field(
        repr=False, default_factory=dict
    )

    # -- accessors ---------------------------------------------------------
    def slice_of(self, region: str, component: str) -> slice:
        """Element slice for ``region.component`` in the packed state vector."""
        return self.entry(region, component).elem_slice

    def byte_slice_of(self, region: str, component: str) -> slice:
        return self.entry(region, component).byte_slice

    def offset_of(self, region: str, component: str) -> tuple[int, int]:
        """``(element_offset, byte_offset)``."""
        e = self.entry(region, component)
        return (e.elem_offset, e.byte_offset)

    def entry(self, region: str, component: str) -> LayoutEntry:
        try:
            return self._by_key[(region, component)]
        except KeyError:
            raise KeyError(
                f"no state component {region!r}.{component!r} in layout; "
                f"known: {sorted(self._by_key)[:8]}..."
            ) from None

    def region_slice(self, region: str) -> slice:
        """Element slice spanning all of a region's components.

        Components of one region are packed contiguously, so a region is a
        single slice rather than a gather.
        """
        entries = self.of_region(region)
        if not entries:
            raise KeyError(f"no region {region!r} in layout")
        return slice(entries[0].elem_offset, entries[-1].elem_offset + entries[-1].numel)

    def of_region(self, region: str) -> tuple[LayoutEntry, ...]:
        return self._by_region.get(region, ())

    def regions(self) -> tuple[str, ...]:
        return tuple(self._by_region)

    def components_of(self, region: str) -> tuple[str, ...]:
        return tuple(e.component for e in self.of_region(region))

    def boundary_entries(self) -> tuple[LayoutEntry, ...]:
        """The exchange contract: what another implementation may read/write."""
        return tuple(e for e in self.entries if e.boundary)

    def entries_on_clock(self, clock: str) -> tuple[LayoutEntry, ...]:
        return tuple(e for e in self.entries if e.clock == str(clock))

    def __iter__(self) -> Iterator[LayoutEntry]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    # -- ABI ---------------------------------------------------------------
    def as_dict(self) -> dict[str, object]:
        return {
            "total_elements": self.total_elements,
            "total_bytes": self.total_bytes,
            "entries": [e.as_dict() for e in self.entries],
        }

    def abi_digest(self) -> str:
        """Hash of the full layout; equal digests mean interchangeable state."""
        return content_hash_of(self.as_dict())

    def boundary_abi_digest(self) -> str:
        """Hash of only the boundary contract.

        Two implementations may differ arbitrarily in private state and still
        exchange boundary state if this digest matches.
        """
        return content_hash_of([e.as_dict() for e in self.boundary_entries()])

    def summary(self) -> str:
        return (
            f"StateLayout({len(self.entries)} blocks, "
            f"{self.total_elements} elements, {self.total_bytes} bytes, "
            f"abi={self.abi_digest()[:12]})"
        )


def _align(offset: int, width: int) -> int:
    rem = offset % width
    return offset if rem == 0 else offset + (width - rem)


def build_state_layout(schema: BrainSchema) -> StateLayout:
    """Pack every region/component into a deterministic layout.

    Order is region declaration order, then component name order.  Declaration
    order is used for regions because it is what the schema author controls and
    what the connectome/adjacency indices are written against; name order is
    used within a region so that adding a component cannot silently renumber
    the others' *names* even though it shifts offsets.
    """
    entries: list[LayoutEntry] = []
    elem = 0
    byte = 0
    for region in schema.regions:
        for name in region.state.names:
            comp = region.state.components[name]
            width = DTYPE_BYTES[comp.dtype]
            byte = _align(byte, width)
            numel = comp.numel()
            nbytes = comp.nbytes()
            entries.append(
                LayoutEntry(
                    region=region.id,
                    component=name,
                    kind=comp.kind,
                    shape=tuple(comp.shape),
                    dtype=comp.dtype,
                    units=str(comp.units),
                    clock=str(comp.temporal.clock),
                    resolution=str(comp.resolution) if comp.resolution else None,
                    boundary=comp.boundary,
                    representation=comp.representation,
                    elem_offset=elem,
                    numel=numel,
                    byte_offset=byte,
                    nbytes=nbytes,
                )
            )
            elem += numel
            byte += nbytes
    by_key = {e.key: e for e in entries}
    by_region: dict[str, list[LayoutEntry]] = {}
    for e in entries:
        by_region.setdefault(e.region, []).append(e)
    return StateLayout(
        entries=tuple(entries),
        total_elements=elem,
        total_bytes=byte,
        _by_key=by_key,
        _by_region={k: tuple(v) for k, v in by_region.items()},
    )
