"""Declared transform chains for the serving path.

``body.tex`` Sec. 2.8, equation (3): the device-to-atlas chain is *stored
separately rather than multiplied and forgotten*, and its uncertainty is
propagated with the relevant adjoint maps while systematic offsets travel as a
separate twist bias.

This module is the runtime's small, opinionated view of that: an ordered set of
:class:`DeclaredEdge` objects that can be resolved between two named frames.
The one rule it exists to enforce:

    **Two differently-named frames are related by a declared, checked edge or
    by nothing at all.  There is no assumed identity.**

That rule is what makes the ``tms-robotics`` bridge safe to write.  The
consumer expresses a coil pose in its own head frame with its own origin
convention; SC-WBD expresses cortical geometry in a subject surface frame.
Wiring the two together with ``Transform.identity()`` would be a silent claim
of tens of millimetres.  Here it is a refusal.

Claim limits: resolving a chain says the *bookkeeping* is consistent.  It says
nothing about whether the declared numbers are correct, and a chain made
entirely of ``assumed`` edges resolves fine and reports itself as assumed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence

import torch
from torch import Tensor

from ._compat import (
    ChainUncertainty,
    Pose,
    PoseUncertainty,
    propagate_chain,
)
from .types import UndeclaredTransform

__all__ = [
    "EdgeProvenance",
    "SessionScope",
    "DeclaredEdge",
    "ResolvedChain",
    "FrameChain",
]

_DT = torch.float64

EdgeProvenance = Literal["definitional", "measured", "assumed"]
#: Which variance bucket this edge's error belongs in (thesis Sec. 2.7).
SessionScope = Literal["measurement", "within_session", "between_session"]


@dataclass(frozen=True)
class DeclaredEdge:
    """One transform that somebody actually declared, with how they got it.

    ``method`` is mandatory and non-empty: an edge whose derivation nobody
    wrote down is not a declared edge.  ``uncertainty`` is mandatory too --
    ``PoseUncertainty`` with a zero covariance is a *claim of exactness*, which
    is legal for a definitional edge and a lie for a measured one.
    """

    pose: Pose
    provenance: EdgeProvenance
    method: str
    uncertainty: PoseUncertainty
    session_scope: SessionScope = "measurement"
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.method.strip():
            raise UndeclaredTransform(
                f"edge {self.pose.label} declares no method; how a transform "
                "was obtained is part of the transform",
                offending_object=self.pose.label,
            )
        if self.provenance == "measured" and float(torch.trace(self.uncertainty.cov)) <= 0.0:
            raise UndeclaredTransform(
                f"measured edge {self.pose.label} declares zero covariance; a "
                "measurement with no error is not a measurement",
                offending_object=self.pose.label,
            )

    @property
    def label(self) -> str:
        return f"{self.pose.label}[{self.provenance}:{self.method}]"

    def inverse(self) -> "DeclaredEdge":
        """The reversed edge, with the twist covariance carried by the adjoint."""
        inv = self.pose.inverse()
        from ._compat import adjoint

        Ad = adjoint(inv.matrix)
        cov = Ad @ self.uncertainty.cov @ Ad.T
        bias = Ad @ self.uncertainty.bias
        sens = None if self.uncertainty.sensitivity is None else Ad @ self.uncertainty.sensitivity
        return DeclaredEdge(
            pose=inv,
            provenance=self.provenance,
            method=self.method + " (inverted)",
            uncertainty=PoseUncertainty(
                cov=cov,
                bias=bias,
                calibration_source=self.uncertainty.calibration_source,
                sensitivity=sens,
            ),
            session_scope=self.session_scope,
            notes=self.notes,
        )


@dataclass(frozen=True)
class ResolvedChain:
    """A composed pose plus the propagated uncertainty of the whole chain."""

    pose: Pose
    edges: tuple[DeclaredEdge, ...]
    uncertainty: ChainUncertainty

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(e.label for e in self.edges)

    @property
    def depends_on_assumed(self) -> bool:
        return any(e.provenance == "assumed" for e in self.edges)

    @property
    def twist_covariance(self) -> Tensor:
        return self.uncertainty.cov

    @property
    def twist_bias(self) -> Tensor:
        return self.uncertainty.bias

    def covariance_by_scope(self) -> dict[str, Tensor]:
        """Split the chain's 6x6 twist covariance by declared session scope.

        Uses the per-edge adjoint Jacobians already computed by
        :func:`~scwbd.transforms.uncertainty.propagate_chain`, so the split
        respects the lever arms rather than adding millimetres naively.  The
        parts sum to ``cov_independent_only``; the shared-calibration cross
        terms are *not* attributable to a single scope and stay in
        :attr:`ChainUncertainty.cov`.
        """
        out = {
            "measurement": torch.zeros(6, 6, dtype=_DT),
            "within_session": torch.zeros(6, 6, dtype=_DT),
            "between_session": torch.zeros(6, 6, dtype=_DT),
        }
        for edge, J in zip(self.edges, self.uncertainty.jacobians):
            out[edge.session_scope] = out[edge.session_scope] + J @ edge.uncertainty.cov @ J.T
        return out

    def variance_by_scope(self) -> dict[str, float]:
        """Trace of :meth:`covariance_by_scope`, per scope."""
        return {k: float(torch.trace(v)) for k, v in self.covariance_by_scope().items()}


class FrameChain:
    """A small graph of :class:`DeclaredEdge` objects.

    Resolution is a breadth-first search over declared edges and their
    inverses.  A missing route raises :class:`~scwbd.runtime.types.
    UndeclaredTransform` (R01); it is never patched with an identity.
    """

    def __init__(self, edges: Sequence[DeclaredEdge] = ()) -> None:
        self._edges: list[DeclaredEdge] = list(edges)
        self._check_units()

    # -- construction -------------------------------------------------------
    def with_edge(self, edge: DeclaredEdge) -> "FrameChain":
        return FrameChain([*self._edges, edge])

    def extended(self, edges: Iterable[DeclaredEdge]) -> "FrameChain":
        return FrameChain([*self._edges, *edges])

    def _check_units(self) -> None:
        units = {e.pose.units for e in self._edges}
        if len(units) > 1:
            raise UndeclaredTransform(
                f"chain mixes length units {sorted(units)}; a unit change is an "
                "explicit affine edge, not an attribute of a rigid pose",
                offending_object=sorted(units),
            )

    # -- queries ------------------------------------------------------------
    @property
    def edges(self) -> tuple[DeclaredEdge, ...]:
        return tuple(self._edges)

    @property
    def frames(self) -> tuple[str, ...]:
        seen: list[str] = []
        for e in self._edges:
            for f in (e.pose.parent, e.pose.child):
                if f not in seen:
                    seen.append(f)
        return tuple(seen)

    def has_frame(self, frame: str) -> bool:
        return frame in self.frames

    def _adjacency(self) -> dict[str, list[DeclaredEdge]]:
        adj: dict[str, list[DeclaredEdge]] = {}
        for e in self._edges:
            adj.setdefault(e.pose.child, []).append(e)  # child -> parent
            adj.setdefault(e.pose.parent, []).append(e.inverse())
        return adj

    def route(self, source: str, target: str) -> tuple[DeclaredEdge, ...]:
        """Declared edges taking coordinates in ``source`` to ``target``."""
        if source == target:
            raise UndeclaredTransform(
                f"source and target are both {source!r}; asking for the "
                "identity usually means a caller lost track of a frame",
                offending_object=source,
            )
        known = self.frames
        for name in (source, target):
            if name not in known:
                raise UndeclaredTransform(
                    f"frame {name!r} appears in no declared edge "
                    f"(declared frames: {list(known)})",
                    offending_object=name,
                )
        adj = self._adjacency()
        prev: dict[str, tuple[str, DeclaredEdge] | None] = {source: None}
        q = deque([source])
        while q:
            cur = q.popleft()
            if cur == target:
                break
            for e in adj.get(cur, ()):
                nxt = e.pose.parent
                if nxt not in prev:
                    prev[nxt] = (cur, e)
                    q.append(nxt)
        if target not in prev:
            raise UndeclaredTransform(
                f"no declared route from {source!r} to {target!r}; the runtime "
                "will not insert an identity between two differently named "
                "frames",
                offending_object=(source, target),
            )
        chain: list[DeclaredEdge] = []
        node = target
        while prev[node] is not None:
            parent, edge = prev[node]  # type: ignore[misc]
            chain.append(edge)
            node = parent
        chain.reverse()
        return tuple(chain)

    def resolve(
        self,
        source: str,
        target: str,
        *,
        shared_covariances: Mapping[str, Any] | None = None,
    ) -> ResolvedChain:
        """Compose the declared route and propagate its uncertainty.

        Cross terms between edges that share a calibration source are retained
        (``ARCHITECTURE.md`` Sec. 3: "dropping ``J_x Sigma_xc J_c^T`` is a bug,
        not an optimization"); :func:`propagate_chain` does that work.
        """
        edges = self.route(source, target)
        poses = [e.pose for e in edges]
        composed = poses[0]
        for p in poses[1:]:
            composed = composed.compose(p)
        unc = propagate_chain(
            poses,
            [e.uncertainty for e in edges],
            shared_covariances=shared_covariances,
        )
        return ResolvedChain(pose=composed, edges=edges, uncertainty=unc)

    def transform_pose(
        self,
        pose: Pose,
        *,
        into: str,
        shared_covariances: Mapping[str, Any] | None = None,
    ) -> tuple[Pose, ResolvedChain]:
        """Re-express ``pose`` (``parent<-coil``) in the frame ``into``.

        Returns the re-expressed pose and the chain that got it there, so the
        caller can report which declared edges it depended on.
        """
        chain = self.resolve(pose.parent, into)
        return chain.pose.compose(pose), chain

