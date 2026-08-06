"""Coordinate frames, calibration manifests, and the frame graph (Appendix C).

"A location is not a three-number property of a brain."  Every sample and
intervention lives in a graph of named frames; each node declares object,
origin, axes, handedness, units, epoch and validity interval; each edge declares
a transform *plus its lineage*, calibration observations, residual and
uncertainty.  Composed paths never replace the original edges.

This module is where refusal R01 gets most of its teeth: unknown units, unknown
handedness, a nominal coordinate with no calibration, an edge with no transform
lineage, or a support whose frame is not in the graph at all.
"""

from __future__ import annotations

from collections import deque
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from .base import SchemaModel
from .ids import FrameId
from .ledger import UncertaintyLedger
from .priors import Prior
from .units import Unit

__all__ = [
    "Handedness",
    "FrameNode",
    "CalibrationManifest",
    "FrameEdge",
    "FrameGraphSpec",
    "TransformKind",
    "ManifestLayer",
]

Handedness = Literal["right", "left", "not_applicable"]

TransformKind = Literal[
    "identity",
    "rigid",
    "affine",
    "deformable",
    "projection",
    "temporal",
    "amplitude",
    "resampling",
]

#: The manifest layers of Appendix C Table tab:calibration-manifest.
ManifestLayer = Literal[
    "physical_reference_frame",
    "deformation_projection",
    "sensor_device_geometry",
    "clock_graph",
    "amplitude_unit",
    "material_field",
    "stimulus_environment",
    "intervention_dose",
    "calibration_validity",
]


class FrameNode(SchemaModel):
    """A physical reference frame."""

    id: FrameId
    #: The physical object the frame is attached to (head, scanner bore, coil).
    object: str
    origin: str
    axes: tuple[str, str, str] = ("x", "y", "z")
    handedness: Handedness
    units: Unit
    epoch: str = "session"
    #: (start, end) seconds relative to ``epoch`` during which the frame is
    #: physically meaningful. None means "not declared" and is an R01 condition
    #: for any frame on a live transform path.
    validity_interval: tuple[float, float] | None = None
    #: Whether coordinates are continuous physical units, array indices, or
    #: normalized image coordinates.  Appendix C requires this distinction be
    #: explicit.
    coordinate_type: Literal["physical", "index", "normalized", "angular"] = "physical"
    description: str = ""

    @model_validator(mode="after")
    def _check(self) -> "FrameNode":
        if len(set(self.axes)) != 3:
            raise ValueError(f"frame {self.id} has duplicate axis labels {self.axes}")
        if self.validity_interval is not None:
            lo, hi = self.validity_interval
            if hi <= lo:
                raise ValueError(f"frame {self.id} validity interval must be ordered")
        if self.coordinate_type == "physical" and self.units.is_dimensionless:
            raise ValueError(
                f"frame {self.id} claims physical coordinates but declares "
                f"dimensionless units {str(self.units)!r}"
            )
        return self


class CalibrationManifest(SchemaModel):
    """Calibration record for a transform, device, clock or amplitude chain.

    Appendix C: "Do not reuse past validity interval silently; inflate
    uncertainty or require recalibration when device, participant geometry or
    environment changes."
    """

    id: str
    layers: tuple[ManifestLayer, ...] = ()
    #: Number of calibration observations backing the fit.
    n_observations: int = Field(default=0, ge=0)
    fitting_method: str = ""
    residual: float | None = Field(default=None, ge=0.0)
    residual_units: Unit = Unit("m")
    #: Query tolerance; a residual beyond it is an Appendix B rejection.
    residual_tolerance: float | None = Field(default=None, gt=0.0)
    #: (start, end) seconds over which the calibration is valid.
    validity_interval: tuple[float, float] | None = None
    validity_domain: dict[str, JsonValue] = Field(default_factory=dict)
    extrapolation_distance: float | None = Field(default=None, ge=0.0)
    recalibration_triggers: tuple[str, ...] = ()
    #: Explicit checks Appendix C requires before training/readout.
    units_checked: bool = False
    handedness_checked: bool = False
    roundtrip_checked: bool = False
    inverse_available: bool = False
    ledger: UncertaintyLedger | None = None

    @model_validator(mode="after")
    def _check(self) -> "CalibrationManifest":
        if self.validity_interval is not None:
            lo, hi = self.validity_interval
            if hi <= lo:
                raise ValueError(
                    f"calibration {self.id!r} validity interval must be ordered"
                )
        return self

    @property
    def residual_ok(self) -> bool:
        if self.residual is None:
            return False
        if self.residual_tolerance is None:
            return False
        return self.residual <= self.residual_tolerance

    @property
    def is_complete(self) -> bool:
        """Everything R01 demands of a calibrated manifest."""
        return (
            self.validity_interval is not None
            and self.units_checked
            and self.handedness_checked
            and self.n_observations > 0
            and self.residual is not None
            and self.ledger is not None
        )

    def missing_fields(self) -> tuple[str, ...]:
        missing: list[str] = []
        if self.validity_interval is None:
            missing.append("validity_interval")
        if not self.units_checked:
            missing.append("units_checked")
        if not self.handedness_checked:
            missing.append("handedness_checked")
        if self.n_observations <= 0:
            missing.append("n_observations")
        if self.residual is None:
            missing.append("residual")
        if self.ledger is None:
            missing.append("ledger")
        return tuple(missing)


class FrameEdge(SchemaModel):
    """A transform between two frames, stored separately from every other edge."""

    src: FrameId
    dst: FrameId
    transform: TransformKind
    #: Free/uncertain transform parameters, each with a prior.
    parameters: dict[str, Prior] = Field(default_factory=dict)
    #: How this transform was obtained: digitization, registration algorithm,
    #: tracker read, manufacturer file.  An edge with no lineage is R01.
    lineage: str = ""
    invertible: bool = True
    calibration: CalibrationManifest | None = None
    ledger: UncertaintyLedger | None = None
    units_in: Unit | None = None
    units_out: Unit | None = None
    #: Round-trip residual for src->dst->src, in ``units_in``.
    roundtrip_residual: float | None = Field(default=None, ge=0.0)
    roundtrip_tolerance: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def _check(self) -> "FrameEdge":
        if self.src == self.dst:
            raise ValueError(f"frame edge is a self-loop at {self.src}")
        return self

    @property
    def key(self) -> tuple[str, str]:
        return (str(self.src), str(self.dst))

    @property
    def roundtrip_ok(self) -> bool:
        if self.roundtrip_residual is None or self.roundtrip_tolerance is None:
            return True  # not asserted; the calibration manifest carries the check
        return self.roundtrip_residual <= self.roundtrip_tolerance


class FrameGraphSpec(SchemaModel):
    """The declared graph of frames and transforms."""

    root: FrameId
    nodes: tuple[FrameNode, ...] = ()
    edges: tuple[FrameEdge, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> "FrameGraphSpec":
        ids = [str(n.id) for n in self.nodes]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate frame nodes: {sorted(dupes)}")
        if str(self.root) not in ids:
            raise ValueError(f"root frame {self.root} is not declared as a node")
        known = set(ids)
        for e in self.edges:
            if str(e.src) not in known or str(e.dst) not in known:
                raise ValueError(
                    f"frame edge {e.key} references an undeclared frame"
                )
        return self

    # -- queries ----------------------------------------------------------
    def has(self, frame: str) -> bool:
        return str(frame) in {str(n.id) for n in self.nodes}

    def node(self, frame: str) -> FrameNode:
        for n in self.nodes:
            if str(n.id) == str(frame):
                return n
        raise KeyError(f"unknown frame {frame!r}")

    def _adjacency(self) -> dict[str, list[tuple[str, FrameEdge, bool]]]:
        """Neighbours as (frame, edge, inverted)."""
        adj: dict[str, list[tuple[str, FrameEdge, bool]]] = {
            str(n.id): [] for n in self.nodes
        }
        for e in self.edges:
            adj[str(e.src)].append((str(e.dst), e, False))
            if e.invertible:
                adj[str(e.dst)].append((str(e.src), e, True))
        return adj

    def path(self, src: str, dst: str) -> tuple[tuple[FrameEdge, bool], ...] | None:
        """Shortest edge path from ``src`` to ``dst``.

        Returns a tuple of (edge, inverted) or None when no defensible path
        exists.  The composed result never replaces the stored edges: callers
        get the edge list and compose it themselves with covariance.
        """
        src, dst = str(src), str(dst)
        if not self.has(src) or not self.has(dst):
            return None
        if src == dst:
            return ()
        adj = self._adjacency()
        prev: dict[str, tuple[str, FrameEdge, bool]] = {}
        seen = {src}
        queue = deque([src])
        while queue:
            cur = queue.popleft()
            for nxt, edge, inverted in adj[cur]:
                if nxt in seen:
                    continue
                seen.add(nxt)
                prev[nxt] = (cur, edge, inverted)
                if nxt == dst:
                    out: list[tuple[FrameEdge, bool]] = []
                    node = dst
                    while node != src:
                        parent, e, inv = prev[node]
                        out.append((e, inv))
                        node = parent
                    return tuple(reversed(out))
                queue.append(nxt)
        return None

    def reachable_from_root(self) -> frozenset[str]:
        return frozenset(
            f for f in (str(n.id) for n in self.nodes) if self.path(str(self.root), f) is not None
        )

    def disconnected_frames(self) -> tuple[str, ...]:
        """Frames with no transform lineage back to the root (R01)."""
        reachable = self.reachable_from_root()
        return tuple(sorted(str(n.id) for n in self.nodes if str(n.id) not in reachable))

    def handedness_conflicts(self) -> tuple[tuple[str, str], ...]:
        """Edges joining frames of opposite handedness without a reflection.

        A rigid transform cannot change handedness; declaring one between a
        right- and a left-handed frame silently mirrors the brain.
        """
        bad: list[tuple[str, str]] = []
        for e in self.edges:
            if e.transform not in ("rigid", "identity", "temporal", "amplitude"):
                continue
            a, b = self.node(str(e.src)), self.node(str(e.dst))
            if "not_applicable" in (a.handedness, b.handedness):
                continue
            if a.handedness != b.handedness:
                bad.append(e.key)
        return tuple(bad)

    def unit_conflicts(self) -> tuple[tuple[str, str], ...]:
        """Edges whose declared output units disagree with the target frame."""
        bad: list[tuple[str, str]] = []
        for e in self.edges:
            if e.units_out is None:
                continue
            target = self.node(str(e.dst))
            if target.coordinate_type != "physical":
                continue
            if not Unit(e.units_out).same_dimension(target.units):
                bad.append(e.key)
        return tuple(bad)
