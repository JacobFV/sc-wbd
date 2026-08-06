"""The consumer's view of model state: **declared ports, never raw slices**.

Why this module exists
----------------------
The run-1 artifact stores state as a uniform ``(B, T, N=454, D=28)`` tensor with
a flat component table -- ``{"dim": 28, "components": [{"name": "rate_e",
"offset": 0, "dim": 1}, ...]}``.  Run 2 replaces that with per-family state of
heterogeneous dimension (``ARCHITECTURE.md`` Sec. 5, ``body.tex`` Sec. 2.1's
``X_i in 𝒳_i``).  Any consumer that learned the number ``28``, or the offset of
``rate_e``, or the identity ``D == 28`` breaks silently on that change --
silently because a wrong slice of a real tensor is still a real tensor full of
plausible numbers.

So the contract is inverted.  A consumer never receives state.  It receives a
:class:`PortContract` -- the model's own declaration of what it exports, by
name, with units and clock -- and reads through :class:`PortedState`, which
resolves names to storage itself.  Offsets exist inside this module and are
never handed out.

Three refusals make that structural rather than advisory:

* :class:`RawStateAccessRefused` -- ``PortedState`` has no ``__getitem__``, no
  ``.tensor``, no iteration.  Reaching for one names this module in the error.
* :class:`UndeclaredPort` -- a name the contract does not carry is an error, not
  an empty tensor and not a zero.
* :class:`UnexportedPort` -- a port the model declares but does **not** export
  (``adaptation``, ``hemo``, ``uncertainty`` in run 1) is refused by name.
  Internal state is not a consumer-visible quantity just because it is present
  in the same buffer.

Narrowing N-1 (``ARCHITECTURE.md`` Sec. 5b) permits run 2 to pad family state to
a common width.  That narrowing is conditional on out-of-span reads being
impossible.  :meth:`PortedState.read` enforces the span: a read whose declared
extent exceeds the storage it was given raises
:class:`SpanViolation` rather than returning padding.

Claim limits
------------
Nothing here computes.  It resolves names to storage and refuses the reads it
cannot support.  A port value is a model quantity, not a measurement of anybody.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

import torch
from torch import Tensor

__all__ = [
    "PortError",
    "UndeclaredPort",
    "UnexportedPort",
    "SpanViolation",
    "RawStateAccessRefused",
    "LayoutNotDeclared",
    "DeclaredPort",
    "PortContract",
    "PortValue",
    "PortedState",
]

#: Families a run-1 checkpoint implicitly has exactly one of.  The flat run-1
#: layout carries no family axis at all, so it is read as a single family whose
#: name says so -- a consumer that pins ``family="all_regions"`` is thereby
#: pinning "this is the homogeneous control arm", which is information it
#: should have.
UNIFORM_FAMILY = "all_regions"


class PortError(RuntimeError):
    """Base for every refusal in this module."""


class UndeclaredPort(PortError, KeyError):
    """A port name the contract does not carry.  Never a zero, never empty."""

    def __str__(self) -> str:  # KeyError repr-quotes its arg; undo that
        return self.args[0] if self.args else ""


class UnexportedPort(PortError):
    """A declared port that the model does not export to consumers."""


class SpanViolation(PortError):
    """A declared extent that the storage handed in does not cover (N-1)."""


class RawStateAccessRefused(PortError):
    """Someone reached for the underlying tensor instead of a declared port."""


class LayoutNotDeclared(PortError):
    """A checkpoint that declares no state layout at all."""


@dataclass(frozen=True)
class DeclaredPort:
    """One named, typed, clocked quantity a model family exports.

    ``offset`` is deliberately **not** part of the public identity: it is
    storage detail, it changes between runs, and it is the thing a consumer
    must never learn.  It is excluded from :meth:`identity` and therefore from
    the contract digest, so a pure re-packing of the same ports does not look
    like a contract change -- while a change of name, family, width, units or
    clock does.
    """

    family: str
    name: str
    dim: int
    units: str
    clock: str
    exported: bool = True
    stochastic: bool = False
    description: str = ""
    #: Storage detail. Private by convention; :class:`PortedState` uses it and
    #: nothing outside this module should.
    offset: int | None = None

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError(f"port {self.qualified!r} declares dim={self.dim}")
        if not self.units:
            raise ValueError(
                f"port {self.qualified!r} declares no units; an unlabelled "
                "quantity cannot be consumed safely"
            )
        if not self.clock:
            raise ValueError(f"port {self.qualified!r} declares no clock")

    @property
    def qualified(self) -> str:
        return f"{self.family}.{self.name}"

    @property
    def key(self) -> tuple[str, str]:
        return (self.family, self.name)

    def identity(self) -> dict[str, Any]:
        """What a consumer is entitled to depend on.  Offset is not in it."""
        return {
            "family": self.family,
            "name": self.name,
            "dim": self.dim,
            "units": self.units,
            "clock": self.clock,
            "exported": self.exported,
        }


@dataclass(frozen=True)
class PortContract:
    """Everything a consumer may read from a model, declared by the model.

    Build it from a checkpoint's own ``state_layout`` with
    :meth:`from_state_layout`, which accepts both the run-1 flat form and the
    region/family-indexed form :mod:`scwbd.compiler.layout` emits, and refuses
    anything else rather than guessing.
    """

    ports: tuple[DeclaredPort, ...]
    source: str = "unspecified"
    _by_key: dict[tuple[str, str], DeclaredPort] = field(
        repr=False, compare=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        by_key: dict[tuple[str, str], DeclaredPort] = {}
        for p in self.ports:
            if p.key in by_key:
                raise ValueError(f"duplicate port declaration {p.qualified!r}")
            by_key[p.key] = p
        object.__setattr__(self, "_by_key", by_key)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_state_layout(
        cls, layout: Mapping[str, Any] | None, *, source: str = "state_layout"
    ) -> "PortContract":
        """Read a checkpoint's declared layout into a contract.

        Two shapes are understood, and **nothing else is inferred**:

        ``{"dim": D, "components": [...]}``
            the run-1 uniform layout.  Every component becomes a port of the
            single family :data:`UNIFORM_FAMILY`, which is how the homogeneity
            of the control arm stays visible to whoever pins the contract.

        ``{"entries": [{"region": ..., "component": ..., ...}]}``
            the region/family-indexed layout (:mod:`scwbd.compiler.layout`).
            ``region`` becomes the family.  This is the shape run 2 must emit.

        A checkpoint with no layout raises :class:`LayoutNotDeclared`.  It does
        not get an empty contract: an empty contract reads as "this model
        exports nothing", which is a different and much more permissive claim
        than "this model did not say".
        """
        if not layout:
            raise LayoutNotDeclared(
                "checkpoint declares no state_layout; a consumer cannot be "
                "given ports the model did not declare, and inferring them "
                "from tensor shapes is exactly the coupling this module exists "
                "to prevent"
            )

        if "entries" in layout:
            rows = list(layout["entries"])
            ports = tuple(
                DeclaredPort(
                    family=str(r["region"]),
                    name=str(r["component"]),
                    dim=int(_numel(r)),
                    units=str(r.get("units") or "dimensionless"),
                    clock=str(r.get("clock") or "unspecified"),
                    # the compiler calls the exchange contract "boundary"
                    exported=bool(r.get("boundary", r.get("exported", False))),
                    stochastic=bool(r.get("stochastic", False)),
                    description=str(r.get("description", "")),
                    offset=_opt_int(r.get("elem_offset", r.get("offset"))),
                )
                for r in rows
            )
            return cls(ports=ports, source=f"{source}:region_indexed")

        if "components" in layout:
            rows = list(layout["components"])
            ports = tuple(
                DeclaredPort(
                    family=UNIFORM_FAMILY,
                    name=str(r["name"]),
                    dim=int(r["dim"]),
                    units=str(r.get("units") or "dimensionless"),
                    clock=str(r.get("clock") or "unspecified"),
                    exported=bool(r.get("exported", False)),
                    stochastic=bool(r.get("stochastic", False)),
                    description=str(r.get("description", "")),
                    offset=_opt_int(r.get("offset")),
                )
                for r in rows
            )
            return cls(ports=ports, source=f"{source}:uniform")

        raise LayoutNotDeclared(
            "state_layout has neither 'entries' (region-indexed) nor "
            f"'components' (uniform); got keys {sorted(layout)}. The runtime "
            "refuses to guess a layout it does not recognise"
        )

    # -- queries -----------------------------------------------------------
    @property
    def families(self) -> tuple[str, ...]:
        seen: list[str] = []
        for p in self.ports:
            if p.family not in seen:
                seen.append(p.family)
        return tuple(seen)

    @property
    def is_uniform(self) -> bool:
        """True when every region shares one family -- i.e. the control arm."""
        return self.families == (UNIFORM_FAMILY,)

    def exported_ports(self) -> tuple[DeclaredPort, ...]:
        return tuple(p for p in self.ports if p.exported)

    def names(self) -> tuple[str, ...]:
        return tuple(p.qualified for p in self.ports)

    def port(self, family: str, name: str) -> DeclaredPort:
        try:
            return self._by_key[(family, name)]
        except KeyError:
            raise UndeclaredPort(
                f"no port {family}.{name!r} in this contract "
                f"({self.source}); declared: {sorted(self.names())}"
            ) from None

    def ports_of(self, family: str) -> tuple[DeclaredPort, ...]:
        out = tuple(p for p in self.ports if p.family == family)
        if not out:
            raise UndeclaredPort(
                f"no family {family!r} in this contract ({self.source}); "
                f"declared families: {list(self.families)}"
            )
        return out

    def ports_on_clock(self, clock: str) -> tuple[DeclaredPort, ...]:
        return tuple(p for p in self.ports if p.clock == clock)

    def width_of(self, family: str) -> int:
        """Total declared element count for ``family`` (its span, per N-1)."""
        return sum(p.dim for p in self.ports_of(family))

    # -- identity ----------------------------------------------------------
    def canonical(self) -> dict[str, Any]:
        return {
            "ports": sorted(
                (p.identity() for p in self.ports),
                key=lambda d: (d["family"], d["name"]),
            )
        }

    def digest(self) -> str:
        """Content hash of the *consumable* contract, offsets excluded.

        A consumer pins this.  When run 2 lands, the pin fails loudly at load
        rather than quietly at the first read of a moved component.
        """
        return hashlib.sha256(
            json.dumps(self.canonical(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def exported_digest(self) -> str:
        """Digest over exported ports only -- what a read-only consumer sees."""
        payload = {
            "exported": sorted(
                (p.identity() for p in self.exported_ports()),
                key=lambda d: (d["family"], d["name"]),
            )
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def summary(self) -> str:
        n_exp = len(self.exported_ports())
        return (
            f"PortContract({len(self.ports)} ports, {n_exp} exported, "
            f"{len(self.families)} famil{'y' if len(self.families)==1 else 'ies'}, "
            f"digest {self.digest()[:12]})"
        )


@dataclass(frozen=True)
class PortValue:
    """A typed read: values plus the declaration they were read under."""

    port: DeclaredPort
    values: Tensor

    @property
    def units(self) -> str:
        return self.port.units

    @property
    def clock(self) -> str:
        return self.port.clock

    @property
    def qualified(self) -> str:
        return self.port.qualified

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"PortValue({self.qualified}, {tuple(self.values.shape)}, "
            f"units={self.units!r}, clock={self.clock!r})"
        )


class PortedState:
    """Read-only, name-addressed access to a model's state.

    Deliberately not a tensor, not a mapping, and not iterable.  The only way
    to get numbers out is :meth:`read`, which requires a family and a port
    name, checks the port is exported, and checks the storage actually covers
    the declared span.
    """

    __slots__ = ("_contract", "_storage", "_family_base")

    def __init__(
        self,
        contract: PortContract,
        storage: Mapping[str, Tensor] | Tensor,
    ) -> None:
        """``storage`` is either one tensor (uniform layout) or per-family.

        The last axis is the state axis in both cases; leading axes (batch,
        time, region) are passed through untouched.
        """
        self._contract = contract
        if isinstance(storage, Tensor):
            if len(contract.families) != 1:
                raise SpanViolation(
                    f"a single storage tensor was given for a contract with "
                    f"{len(contract.families)} families "
                    f"({list(contract.families)}); heterogeneous state must be "
                    "passed as a per-family mapping so that no family can read "
                    "into another's span"
                )
            self._storage: dict[str, Tensor] = {contract.families[0]: storage}
        else:
            self._storage = dict(storage)
        # per-family base offset: ports may carry global offsets (run-1 flat
        # layout) or family-local ones. Normalise to family-local so a family's
        # read can never index outside its own tensor.
        base: dict[str, int] = {}
        for fam in contract.families:
            offs = [p.offset for p in contract.ports_of(fam) if p.offset is not None]
            base[fam] = min(offs) if offs else 0
        self._family_base = base

    # -- the only read path ------------------------------------------------
    def read(self, family: str, name: str) -> PortValue:
        """Return the declared port's values, or refuse and say why."""
        port = self._contract.port(family, name)
        if not port.exported:
            raise UnexportedPort(
                f"port {port.qualified!r} is declared but not exported: it is "
                f"model-internal state ({port.description or 'no description'}). "
                "A consumer may not read it. Exported ports for this family: "
                + str([p.name for p in self._contract.ports_of(family) if p.exported])
            )
        try:
            buf = self._storage[family]
        except KeyError:
            raise SpanViolation(
                f"no storage was supplied for family {family!r}; supplied: "
                f"{sorted(self._storage)}"
            ) from None

        if port.offset is None:
            raise SpanViolation(
                f"port {port.qualified!r} declares no offset in this contract "
                f"({self._contract.source}); it cannot be located in storage"
            )
        start = port.offset - self._family_base[family]
        stop = start + port.dim
        width = int(buf.shape[-1])
        if start < 0 or stop > width:
            # N-1's enforced span: padding is never returned as a value.
            raise SpanViolation(
                f"port {port.qualified!r} declares elements [{start}:{stop}) "
                f"but family {family!r} storage is {width} wide. Under "
                "ARCHITECTURE.md Sec. 5b narrowing N-1 padded state is only "
                "equivalent to ragged state if out-of-span reads are "
                "impossible, so this raises instead of returning padding"
            )
        return PortValue(port=port, values=buf[..., start:stop])

    def read_many(self, family: str) -> tuple[PortValue, ...]:
        """Every *exported* port of ``family``, each with its declaration."""
        return tuple(
            self.read(family, p.name)
            for p in self._contract.ports_of(family)
            if p.exported
        )

    @property
    def contract(self) -> PortContract:
        return self._contract

    def families(self) -> tuple[str, ...]:
        return self._contract.families

    # -- the refusals ------------------------------------------------------
    def __getitem__(self, key: Any) -> Any:
        raise RawStateAccessRefused(
            f"PortedState is not indexable (got {key!r}). Model state has no "
            "stable index: run 1 is uniform (B,T,454,28) and run 2 is "
            "per-family with heterogeneous widths, so any index you write "
            "today reads a different quantity tomorrow -- silently, because a "
            "wrong slice is still full of plausible numbers. Use "
            "read(family, name); the declared ports are "
            f"{list(self._contract.names())}"
        )

    def __iter__(self) -> Iterator[Any]:
        raise RawStateAccessRefused(
            "PortedState is not iterable; iterate the declared ports instead "
            "(state.contract.exported_ports()) so that every value you touch "
            "arrives with its units and its clock"
        )

    def __len__(self) -> int:
        raise RawStateAccessRefused(
            "PortedState has no length; a state width is storage detail and "
            "consumers that learned one broke when it changed. Use "
            "contract.width_of(family) if you genuinely need a span"
        )

    def __getattr__(self, item: str) -> Any:
        # __slots__ means anything unknown lands here.
        raise RawStateAccessRefused(
            f"PortedState has no attribute {item!r}. In particular it exposes "
            "no raw tensor: reads go through read(family, name) so that the "
            "declaration travels with the value"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"PortedState({self._contract.summary()})"


def _numel(row: Mapping[str, Any]) -> int:
    if "numel" in row:
        return int(row["numel"])
    shape: Sequence[int] = row.get("shape") or ()
    n = 1
    for s in shape:
        n *= int(s)
    return n if shape else int(row.get("dim", 1))


def _opt_int(v: Any) -> int | None:
    return None if v is None else int(v)
