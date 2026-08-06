"""Segment-layout regional state — the retirement of narrowing N-1.

``body.tex`` §2.1: the components of ``X_i`` "need not have equal shape or even
be ordinary dense tensors".  We stored them padded to ``D = max_f d_f`` anyway,
declared it as narrowing **N-1**, and enforced a span guard to make the padding
observationally equivalent to the ragged layout we had given up.

Measured on the real 414-parcel prior, 9 declared families:

    ragged cells   Σ n_f · d_f    11 662
    padded cells   N · D          24 426      D = 59, set by 2 hippocampal parcels
    waste                         52.3 %

**412 of 414 parcels carried ≥ 27 dead channels so that two could carry 59.**
That is a bad trade in greenfield, and paying it while declaring a narrowing of
the paper's central claim is worse than paying it silently would have been.

Why a **segment layout** and not the alternatives
-------------------------------------------------
The partition is fixed at construction: every family's ``(n_f, d_f)`` is a
compile-time constant and nothing about it varies per batch.  So this is not
actually a ragged-tensor problem — it is a **struct of dense arrays**, and the
right representation is the one that keeps every individual tensor dense.

* ``torch.nested_tensor`` — rejected.  It exists for genuinely data-dependent
  raggedness, which this is not, and it graph-breaks under ``torch.compile`` on
  most ops we need (``index_select``, ``cat`` along the ragged axis, autograd
  through views).  Paying its limitations to express a shape we already know
  statically is the wrong trade twice over.
* **a single flat buffer with segment offsets** — workable, and what you would
  use if the shapes were dynamic.  Every per-family view is a ``narrow`` +
  ``reshape``; those compile, but each one is an aliasing view of a shared
  tensor, so an in-place write anywhere is a correctness hazard everywhere and
  autograd has to thread through the aliasing.  It buys nothing here because we
  do not need a contiguous whole-brain tensor.
* **per-family dense blocks** — chosen.  Each block is an ordinary dense
  ``(B, T, n_f, d_f)`` tensor with a static shape, so **every op is a normal
  dense op**: ``torch.compile`` sees no dynamic shapes, no data-dependent
  control flow, and no nested-tensor fallbacks.  The number of families is small
  (9) and fixed, so the Python-level loop over them is unrolled at trace time.

**The span guard becomes unnecessary rather than merely satisfied.**  There is
no pad, so there is nothing to write into.  ``FamilyStateLayout.assert_clean``
and ``SpanViolation``-on-pad-write have no referent in this layout — the class
of bug they existed to catch is now unrepresentable, which is the only real way
to retire a guard.  What survives is the *type* discipline they enforced: a
family still cannot read a component it does not declare, because there is no
offset arithmetic that could reach one.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, Mapping

import torch
from torch import Tensor

from .families import FamilyStateLayout, PortMismatch, SpanViolation

__all__ = ["FamilyState"]


class FamilyState:
    """Regional state as one dense block per family.  No padding, no spans.

    ``blocks[name]`` is ``(..., n_f, d_f)``.  The leading axes are the caller's
    (``(B,)`` for one step, ``(B, T)`` for a trajectory) and are identical across
    families; only the trailing two differ, which is the whole point.

    Reads are by ``(family, component)`` name.  There is deliberately no way to
    ask for "channel 37 of the state": the concept does not exist here, because
    channel 37 means a different physical quantity in each family.
    """

    __slots__ = ("layout", "blocks")

    def __init__(self, layout: FamilyStateLayout, blocks: Mapping[str, Tensor]) -> None:
        self.layout = layout
        missing = [f.name for f in layout if f.name not in blocks]
        if missing:
            raise SpanViolation(
                f"FamilyState is missing block(s) for {missing}. Every family must contribute, "
                "including families whose operator produced zeros — a missing block is a family "
                "silently dropped from the rollout."
            )
        extra = set(blocks) - {f.name for f in layout}
        if extra:
            raise SpanViolation(f"FamilyState has block(s) for unknown families {sorted(extra)}")
        for f in layout:
            b = blocks[f.name]
            if b.shape[-2:] != (f.n_regions, f.dim):
                raise SpanViolation(
                    f"family {f.name!r} block has trailing shape {tuple(b.shape[-2:])}, expected "
                    f"({f.n_regions}, {f.dim}). In a segment layout the shape IS the declaration; "
                    "there is no pad to absorb a mismatch."
                )
        lead = {tuple(b.shape[:-2]) for b in blocks.values()}
        if len(lead) != 1:
            raise SpanViolation(f"family blocks disagree on their leading (batch/time) shape: {sorted(lead)}")
        self.blocks: dict[str, Tensor] = dict(blocks)

    # -- construction ------------------------------------------------------
    @classmethod
    def zeros(cls, layout: FamilyStateLayout, *lead: int, device=None, dtype=torch.float32) -> "FamilyState":
        return cls(
            layout,
            {f.name: torch.zeros(*lead, f.n_regions, f.dim, device=device, dtype=dtype) for f in layout},
        )

    @classmethod
    def from_padded(cls, layout: FamilyStateLayout, x: Tensor) -> "FamilyState":
        """Adapt a legacy padded ``(..., N, D)`` tensor.  Drops the pad by construction."""
        return cls(layout, {f.name: layout.gather(x, f.name) for f in layout})

    def to_padded(self) -> Tensor:
        """Re-pad to ``(..., N, D)``.  **Interop only** — this re-creates the waste.

        Kept because the observation heads still take a dense tensor; every use
        is a place that has not been converted yet, and there should be none
        left when O-5's head rewrite lands.
        """
        D = self.layout.dim
        chunks = []
        for f in self.layout:
            b = self.blocks[f.name]
            pad = D - f.dim
            chunks.append(torch.cat([b, b.new_zeros(*b.shape[:-1], pad)], dim=-1) if pad else b)
        return self.layout.assemble(chunks)

    # -- geometry ----------------------------------------------------------
    @property
    def lead_shape(self) -> tuple[int, ...]:
        return tuple(next(iter(self.blocks.values())).shape[:-2])

    @property
    def dtype(self) -> torch.dtype:
        return next(iter(self.blocks.values())).dtype

    @property
    def device(self) -> torch.device:
        return next(iter(self.blocks.values())).device

    def n_cells(self) -> int:
        """Total stored scalars per leading element — the number N-1 was wasting."""
        return sum(f.n_regions * f.dim for f in self.layout)

    def __iter__(self) -> Iterator[tuple[str, Tensor]]:
        return iter(self.blocks.items())

    def __getitem__(self, name: str) -> Tensor:
        if name not in self.blocks:
            raise KeyError(f"no family {name!r}; have {sorted(self.blocks)}")
        return self.blocks[name]

    # -- typed access ------------------------------------------------------
    def get(self, family: str, component: str) -> Tensor:
        """``(..., n_f, dim(component))``.  The only sanctioned read."""
        f = self.layout.family(family)
        if component not in f.layout:
            raise SpanViolation(
                f"family {family!r} does not declare component {component!r}; it has "
                f"{[c.name for c in f.layout.components]}. body.tex §2.1 indexes the state SPACE "
                "by region — a component another family owns does not exist here."
            )
        return self.blocks[family][..., f.layout.slice(component)]

    def set(self, family: str, component: str, value: Tensor) -> "FamilyState":
        """Out-of-place write of one component of one family."""
        f = self.layout.family(family)
        if component not in f.layout:
            raise SpanViolation(f"family {family!r} does not declare component {component!r}")
        a, b = f.layout.span(component)
        blk = self.blocks[family]
        if value.shape[-1] != b - a:
            raise SpanViolation(
                f"{family}.{component} is {b - a} channels wide; got {value.shape[-1]}"
            )
        new = torch.cat([blk[..., :a], value.to(blk.dtype), blk[..., b:]], dim=-1)
        return FamilyState(self.layout, {**self.blocks, family: new})

    def port(self, family: str, port: str) -> Tensor:
        """Read a family's declared **out**-port."""
        f = self.layout.family(family)
        p = f.port(port)
        if p.direction != "out":
            raise PortMismatch(f"port {port!r} of family {family!r} is an in-port; it cannot be read from")
        return torch.cat([self.get(family, c) for c in p.components], dim=-1)

    def interface(self, component: str) -> Tensor:
        """``(..., N, dim)`` for a component **every** family declares.

        The shared prefix (``rate_e``, ``rate_i``, ``hemo``, ``uncertainty``) is
        the one place a whole-brain dense tensor is meaningful, because those
        four are the instrument-facing quantities every family exposes and they
        genuinely do have equal shape.  Everything else stays ragged.
        """
        missing = [f.name for f in self.layout if component not in f.layout]
        if missing:
            raise SpanViolation(
                f"component {component!r} is not declared by {missing}, so there is no whole-brain "
                "view of it. Only the shared interface prefix has one; read the rest per family."
            )
        return self.layout.assemble([self.get(f.name, component) for f in self.layout])

    # -- functional updates ------------------------------------------------
    def map(self, fn: Callable[[str, Tensor], Tensor]) -> "FamilyState":
        return FamilyState(self.layout, {n: fn(n, b) for n, b in self.blocks.items()})

    def combine(self, other: "FamilyState", fn: Callable[[Tensor, Tensor], Tensor]) -> "FamilyState":
        if other.layout is not self.layout and len(other.blocks) != len(self.blocks):
            raise SpanViolation("cannot combine FamilyStates built on different partitions")
        return FamilyState(self.layout, {n: fn(b, other.blocks[n]) for n, b in self.blocks.items()})

    def __add__(self, other: "FamilyState") -> "FamilyState":
        return self.combine(other, torch.add)

    def detach(self) -> "FamilyState":
        return self.map(lambda _, b: b.detach())

    def to(self, *a, **kw) -> "FamilyState":
        return self.map(lambda _, b: b.to(*a, **kw))

    def stack(self, others: list["FamilyState"], dim: int = 1) -> "FamilyState":
        """Stack a list of per-step states into a trajectory along ``dim``."""
        seq = [self, *others]
        return FamilyState(
            self.layout, {f.name: torch.stack([s.blocks[f.name] for s in seq], dim=dim) for f in self.layout}
        )

    # -- provenance --------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        padded = self.layout.n_regions * self.layout.dim
        ragged = self.n_cells()
        return {
            "layout": "family_segment",
            "narrowing": "N-1 retired: there is no pad, so there is nothing to enforce",
            "n_families": len(self.layout),
            "n_regions": self.layout.n_regions,
            "cells_ragged": ragged,
            "cells_if_padded": padded,
            "cells_saved": padded - ragged,
            "padding_fraction_avoided": round(1.0 - ragged / padded, 4),
            "blocks": {f.name: [f.n_regions, f.dim] for f in self.layout},
        }
