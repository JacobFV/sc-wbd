"""The frame graph -- Appendix C layers 1--3, thesis §2.8.

    "A manifest node declares a frame, units, handedness, epoch and physical
     object; an edge declares a rigid, affine, deformable, temporal or amplitude
     transform plus parameters, calibration observations and uncertainty. A path
     is valid only if its composed units and semantics agree."

    "The transforms are stored separately rather than multiplied and forgotten."

That last sentence is the design constraint of this module.  :meth:`FrameGraph.path`
returns a :class:`TransformPath` that *references* its edges; it never writes a
composed edge back into the graph, and the composed pose is a derived view that
carries the list of contributing edges with it.  Composition without provenance
is how a 3 mm coregistration error becomes an anatomical claim.

What a path is checked for (all of them, every time):

* **units** -- millimetres never meet metres implicitly;
* **handedness** -- a reflection is a refusal, not a sign;
* **epoch** -- Tuesday's head<-tracker is not Wednesday's;
* **support/kind** -- a deformable warp does not silently become a pose;
* **validity + calibration validity** -- expired means refuse or inflate;
* **round-trip residual** -- measured, never assumed zero;
* **path agreement** -- when several admissible paths exist, all are returned
  and their disagreement is reported as a number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

import torch

from .calibration import CalibrationRecord, ExpiryPolicy, ValidityCheck
from .errors import (
    FrameMismatchError,
    HandednessError,
    NoPathError,
    NonInvertibleTransformError,
    TransformError,
    UnknownFrameError,
)
from .se3 import DTYPE, Pose, ValidityInterval, adjoint, as_tensor, log_so3
from .uncertainty import ChainUncertainty, PoseUncertainty, propagate_chain
from .units import Handedness, require_same_handedness, require_same_unit

EdgeKind = Literal["rigid", "affine", "deformable", "temporal", "amplitude"]


# --------------------------------------------------------------------------
# frames
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Frame:
    """A node of the manifest (Appendix C layer 1).

    Attributes
    ----------
    id:
        e.g. ``"scanner_RAS"``, ``"subject_surface_RAS"``, ``"fiducial_head"``,
        ``"tracker"``, ``"coil"``, ``"mni152"``, ``"voxel_index"``.
    object:
        The *physical* thing the frame is attached to ("participant head",
        "TMS coil", "optical tracker base", "atlas template").  Two frames
        attached to different objects cannot be related by a static edge.
    origin, axes:
        Human-readable definition, e.g. ``"anterior commissure"`` and
        ``"RAS (+x right, +y anterior, +z superior)"``.  Free text, but
        required: an undocumented axis convention is how L/R flips survive.
    handedness, units:
        Chirality and length unit (or ``"voxel"``/``"normalized"`` for index
        frames, which then require an affine edge to reach a metric frame).
    validity:
        When the frame definition holds (a head frame defined by fiducials is
        void once the participant is repositioned).
    """

    id: str
    object: str
    origin: str = "undeclared"
    axes: str = "undeclared"
    handedness: Handedness = Handedness.RIGHT
    units: str = "m"
    validity: ValidityInterval = field(default_factory=ValidityInterval.unbounded)
    epoch: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "handedness", Handedness.coerce(self.handedness))
        if self.object in ("", "undeclared"):
            raise UnknownFrameError(
                f"frame {self.id!r} does not declare the physical object it is attached to",
                remedy=(
                    "Appendix C layer 1 requires object, origin, axes, handedness, "
                    "units and validity interval. 'A location is not a "
                    "three-number property of a brain' (§2.8)."
                ),
                offending_object=self.id,
            )


# --------------------------------------------------------------------------
# edges
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DeformableTransform:
    """A non-rigid warp: point map + local Jacobian + registration residuals.

    §2.8: "Non-rigid image-to-atlas warps retain local Jacobians and
    registration residuals rather than being disguised as rigid pose error."
    Composing one into an SE(3) chain is therefore refused; it can only be
    applied to points, and its uncertainty must be propagated by Monte Carlo or
    intervals (Appendix C).
    """

    forward: Callable[[torch.Tensor], torch.Tensor]
    jacobian: Callable[[torch.Tensor], torch.Tensor] | None = None
    inverse: Callable[[torch.Tensor], torch.Tensor] | None = None
    residual_rms: float | None = None
    residual_map: Any = None
    method: str = "undeclared"

    @property
    def invertible(self) -> bool:
        return self.inverse is not None

    def apply(self, points: Any) -> torch.Tensor:
        p = torch.as_tensor(points, dtype=DTYPE)
        return self.forward(p)

    def local_jacobian(self, points: Any) -> torch.Tensor:
        p = torch.as_tensor(points, dtype=DTYPE)
        if self.jacobian is not None:
            return self.jacobian(p)
        p = p.detach().clone().requires_grad_(True)
        out = self.forward(p)
        rows = []
        for k in range(3):
            g = torch.autograd.grad(out[..., k].sum(), p, retain_graph=True)[0]
            rows.append(g)
        return torch.stack(rows, dim=-2)


@dataclass(frozen=True)
class TransformEdge:
    """One stored transform.  Never replaced by a composition.

    ``kind`` follows Appendix C: ``rigid`` (pose), ``affine`` (scale/shear,
    including voxel->mm and mirrors), ``deformable`` (warp), ``temporal``
    (handled by :mod:`scwbd.transforms.clock_graph`, referenced here only for
    completeness of the manifest), ``amplitude`` (gain/offset, layer 5).
    """

    parent: str
    child: str
    kind: EdgeKind
    pose: Pose | None = None
    matrix: torch.Tensor | None = None  # affine 4x4 (kind="affine")
    warp: DeformableTransform | None = None
    params: dict[str, Any] = field(default_factory=dict)
    calibration: CalibrationRecord = field(default_factory=CalibrationRecord)
    uncertainty: PoseUncertainty | None = None
    invertible: bool = True
    reflection_declared: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if self.kind == "rigid":
            if self.pose is None:
                raise TransformError(
                    f"rigid edge {self.label} carries no pose",
                    remedy="Supply a Pose.",
                    offending_object=self.label,
                )
            if (self.pose.parent, self.pose.child) != (self.parent, self.child):
                raise FrameMismatchError(
                    f"edge {self.label} declares a pose for {self.pose.label}",
                    remedy="The pose's frame pair must match the edge's.",
                    offending_object=(self.label, self.pose.label),
                )
        elif self.kind == "affine":
            if self.matrix is None:
                raise TransformError(
                    f"affine edge {self.label} carries no matrix",
                    remedy="Supply a 4x4 affine matrix.",
                    offending_object=self.label,
                )
            M = as_tensor(self.matrix, shape=(4, 4))
            object.__setattr__(self, "matrix", M)
            det = float(torch.linalg.det(M[:3, :3]))
            if abs(det) < 1e-12 and self.invertible:
                raise NonInvertibleTransformError(
                    f"affine edge {self.label} is singular (det = {det:.3e}) but "
                    "declares itself invertible",
                    remedy=(
                        "A rank-deficient map (e.g. a projection onto a slice) "
                        "has no inverse; declare invertible=False."
                    ),
                    offending_object=self.label,
                )
            if det < 0 and not self.reflection_declared:
                raise HandednessError(
                    f"affine edge {self.label} has det = {det:.6f} < 0: it mirrors "
                    "the coordinate system",
                    remedy=(
                        "If a mirror is genuinely intended (e.g. a left/right "
                        "flipped template), set reflection_declared=True so it "
                        "is auditable. Otherwise fix the axis labels."
                    ),
                    offending_object=self.label,
                )
        elif self.kind == "deformable":
            if self.warp is None:
                raise TransformError(
                    f"deformable edge {self.label} carries no warp",
                    remedy="Supply a DeformableTransform.",
                    offending_object=self.label,
                )
            if self.invertible and not self.warp.invertible:
                raise NonInvertibleTransformError(
                    f"deformable edge {self.label} declares invertible=True but "
                    "supplies no inverse map",
                    remedy=(
                        "Appendix C layer 2: 'No inverse assumed where absent.' "
                        "Supply the inverse warp or declare invertible=False."
                    ),
                    offending_object=self.label,
                )
        elif self.kind in ("temporal", "amplitude"):
            pass
        else:  # pragma: no cover - Literal guards this
            raise TransformError(
                f"unknown edge kind {self.kind!r}",
                remedy="Use rigid / affine / deformable / temporal / amplitude.",
                offending_object=self.kind,
            )

    @property
    def label(self) -> str:
        return f"{self.parent}<-{self.child}"

    @property
    def is_linear(self) -> bool:
        return self.kind in ("rigid", "affine")

    def as_matrix(self) -> torch.Tensor:
        if self.kind == "rigid":
            return self.pose.matrix
        if self.kind == "affine":
            return self.matrix
        raise TransformError(
            f"edge {self.label} of kind {self.kind!r} has no 4x4 matrix",
            remedy="Deformable warps are applied to points, not composed as poses.",
            offending_object=self.label,
        )

    def reversed_edge(self) -> "TransformEdge":
        """The inverse edge -- computed on demand, never stored in the graph."""
        if not self.invertible:
            raise NonInvertibleTransformError(
                f"edge {self.label} declares invertible=False",
                remedy=(
                    "Traverse this edge in its declared direction only, or "
                    "supply the measured reverse transform as its own edge."
                ),
                offending_object=self.label,
            )
        if self.kind == "rigid":
            return TransformEdge(
                self.child,
                self.parent,
                "rigid",
                pose=self.pose.inverse(),
                params=self.params,
                calibration=self.calibration,
                uncertainty=_invert_uncertainty(self.pose, self.uncertainty),
                invertible=True,
                notes=self.notes + " (inverse)",
            )
        if self.kind == "affine":
            M = self.matrix
            det = float(torch.linalg.det(M[:3, :3]))
            if abs(det) < 1e-12:
                raise NonInvertibleTransformError(
                    f"affine edge {self.label} is singular (det = {det:.3e})",
                    remedy="Declare invertible=False and use the forward direction.",
                    offending_object=self.label,
                )
            return TransformEdge(
                self.child,
                self.parent,
                "affine",
                matrix=torch.linalg.inv(M),
                params=self.params,
                calibration=self.calibration,
                uncertainty=self.uncertainty,
                invertible=True,
                reflection_declared=self.reflection_declared,
                notes=self.notes + " (inverse)",
            )
        if self.kind == "deformable":
            w = self.warp
            return TransformEdge(
                self.child,
                self.parent,
                "deformable",
                warp=DeformableTransform(
                    forward=w.inverse,
                    inverse=w.forward,
                    residual_rms=w.residual_rms,
                    method=w.method + "_inverse",
                ),
                params=self.params,
                calibration=self.calibration,
                invertible=True,
                notes=self.notes + " (inverse)",
            )
        raise TransformError(
            f"edge kind {self.kind!r} cannot be inverted here",
            remedy="Temporal edges are inverted by the clock graph.",
            offending_object=self.label,
        )


def _invert_uncertainty(
    pose: Pose, u: PoseUncertainty | None
) -> PoseUncertainty | None:
    """Push a twist covariance through a pose inversion via the adjoint."""
    if u is None:
        return None
    A = adjoint(pose.matrix)
    return PoseUncertainty(
        cov=A @ u.cov @ A.T,
        bias=-(A @ u.bias),
        calibration_source=u.calibration_source,
        sensitivity=None if u.sensitivity is None else A @ u.sensitivity,
    )


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TransformPath:
    """A validated path.  Holds its edges; the composition is a derived view."""

    source: str
    target: str
    edges: tuple[TransformEdge, ...]
    pose: Pose | None
    matrix: torch.Tensor | None
    uncertainty: ChainUncertainty | None
    validity: ValidityInterval
    validity_checks: tuple[ValidityCheck, ...]
    units: str
    handedness: Handedness
    linear: bool
    warnings: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return " . ".join(e.label for e in self.edges)

    @property
    def inflated(self) -> bool:
        return any(not c.inside for c in self.validity_checks)

    def apply(self, points: Any) -> torch.Tensor:
        """Map points along the path, edge by edge (deformable edges included)."""
        p = torch.as_tensor(points, dtype=DTYPE)
        for e in reversed(self.edges):
            if e.kind == "deformable":
                p = e.warp.apply(p)
            else:
                M = e.as_matrix()
                p = p @ M[:3, :3].T + M[:3, 3]
        return p

    def as_record(self) -> dict[str, Any]:
        rec: dict[str, Any] = {
            "source": self.source,
            "target": self.target,
            "edges": [e.label for e in self.edges],
            "kinds": [e.kind for e in self.edges],
            "units": self.units,
            "handedness": self.handedness.value,
            "linear": self.linear,
            "validity": str(self.validity),
            "inflated": self.inflated,
            "validity_checks": [c.as_record() for c in self.validity_checks],
            "warnings": list(self.warnings),
            "calibration_methods": [e.calibration.method for e in self.edges],
        }
        if self.uncertainty is not None:
            rec["translation_sd"] = self.uncertainty.translation_sd.tolist()
            rec["rotation_sd_rad"] = self.uncertainty.rotation_sd_rad.tolist()
            rec["independence_understatement"] = self.uncertainty.understatement()
        return rec


@dataclass(frozen=True)
class PathDisagreement:
    """Numeric disagreement between two admissible paths for the same frame pair."""

    path_a: str
    path_b: str
    translation: float
    rotation_rad: float
    mahalanobis: float | None
    agree: bool
    units: str

    def as_record(self) -> dict[str, Any]:
        return {
            "path_a": self.path_a,
            "path_b": self.path_b,
            "translation": self.translation,
            "units": self.units,
            "rotation_rad": self.rotation_rad,
            "rotation_deg": math.degrees(self.rotation_rad),
            "mahalanobis": self.mahalanobis,
            "agree": self.agree,
        }


@dataclass(frozen=True)
class PathSet:
    """Every admissible path between two frames, plus the rejected ones and why.

    Rejected paths are retained deliberately.  "There is no path" and "there are
    three paths and two of them are expired" are different scientific
    situations, and a caller that cannot see the difference cannot report it.
    """

    source: str
    target: str
    paths: tuple[TransformPath, ...]
    rejected: tuple[tuple[str, str], ...]
    disagreements: tuple[PathDisagreement, ...]
    at: float | None
    errors: tuple[TransformError, ...] = ()

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self):
        return iter(self.paths)

    @property
    def best(self) -> TransformPath:
        """The lowest-uncertainty admissible path (shortest as a tie-break)."""
        if not self.paths:
            if self.errors:
                raise self.errors[0]
            raise NoPathError(
                f"no admissible path from {self.source!r} to {self.target!r}",
                remedy="See PathSet.rejected for why each candidate was refused.",
                offending_object=self.rejected,
            )

        def key(p: TransformPath) -> tuple[float, int]:
            tr = (
                float(torch.trace(p.uncertainty.cov))
                if p.uncertainty is not None
                else math.inf
            )
            return (tr, len(p.edges))

        return min(self.paths, key=key)

    @property
    def agree(self) -> bool:
        return all(d.agree for d in self.disagreements)

    def require_agreement(self) -> "PathSet":
        if not self.agree:
            worst = max(self.disagreements, key=lambda d: d.translation)
            raise TransformError(
                f"admissible paths {self.source} <- {self.target} disagree beyond "
                f"their stated uncertainty: {worst.translation:.6g} {worst.units} / "
                f"{math.degrees(worst.rotation_rad):.4g} deg between "
                f"[{worst.path_a}] and [{worst.path_b}]",
                remedy=(
                    "Two calibration chains cannot both be right. Recalibrate, "
                    "widen the declared uncertainty with evidence, or branch the "
                    "posterior over the two chains -- do not average them."
                ),
                offending_object=[d.as_record() for d in self.disagreements],
            )
        return self

    def as_record(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "at": self.at,
            "n_admissible": len(self.paths),
            "paths": [p.as_record() for p in self.paths],
            "rejected": [{"path": p, "reason": r} for p, r in self.rejected],
            "disagreements": [d.as_record() for d in self.disagreements],
            "agree": self.agree,
        }


# --------------------------------------------------------------------------
# the graph
# --------------------------------------------------------------------------


class FrameGraph:
    """Frames and the calibrated transforms between them.

    Edges are stored in their declared direction and **never** replaced by a
    composition.  Reverse traversal builds an inverse *on demand*
    (:meth:`TransformEdge.reversed_edge`) and only when the edge declares itself
    invertible.
    """

    def __init__(
        self,
        *,
        expiry_policy: ExpiryPolicy | str = ExpiryPolicy.REFUSE,
        max_path_length: int = 8,
        agreement_chi2: float = 12.592,  # chi2_{6, 0.95}
    ) -> None:
        self._frames: dict[str, Frame] = {}
        self._edges: list[TransformEdge] = []
        self._adj: dict[str, list[tuple[str, int, bool]]] = {}
        self.expiry_policy = ExpiryPolicy.coerce(expiry_policy)
        self.max_path_length = max_path_length
        self.agreement_chi2 = agreement_chi2
        self.shared_covariances: dict[str, torch.Tensor] = {}

    # -- declaration -------------------------------------------------------

    def add_frame(self, frame: Frame) -> Frame:
        if frame.id in self._frames:
            raise TransformError(
                f"frame {frame.id!r} is already declared",
                remedy="Frame ids are unique; version the id if the definition changed.",
                offending_object=frame.id,
            )
        self._frames[frame.id] = frame
        self._adj.setdefault(frame.id, [])
        return frame

    def frame(self, fid: str) -> Frame:
        try:
            return self._frames[fid]
        except KeyError:
            raise UnknownFrameError(
                f"frame {fid!r} is not declared (known: {sorted(self._frames)})",
                remedy=(
                    "Declare the frame with its object, origin, axes, handedness, "
                    "units and validity interval before referencing it."
                ),
                offending_object=fid,
            ) from None

    @property
    def frames(self) -> dict[str, Frame]:
        return dict(self._frames)

    @property
    def edges(self) -> tuple[TransformEdge, ...]:
        return tuple(self._edges)

    def declare_shared_calibration(self, name: str, cov: Any) -> None:
        """Register ``Sigma_c`` for a calibration shared by several edges.

        This is what makes T5's cross terms nonzero downstream: the optical
        tracker calibration entering both ``head<-tracker`` and
        ``tracker<-device`` is one random variable, not two.
        """
        C = as_tensor(cov)
        if C.ndim != 2 or C.shape[0] != C.shape[1]:
            raise TransformError(
                f"shared calibration covariance for {name!r} must be square, got {tuple(C.shape)}",
                remedy="Supply Sigma_c for the shared calibration vector.",
                offending_object=name,
            )
        self.shared_covariances[name] = 0.5 * (C + C.T)

    def add_edge(self, edge: TransformEdge) -> TransformEdge:
        """Store an edge, after checking it against its two frame declarations."""
        pf, cf = self.frame(edge.parent), self.frame(edge.child)

        if edge.kind == "rigid":
            require_same_unit(
                pf.units, cf.units, context=f"rigid edge {edge.label}"
            )
            require_same_unit(
                edge.pose.units, pf.units, context=f"rigid edge {edge.label} vs frame units"
            )
            require_same_handedness(
                pf.handedness, cf.handedness, context=f"rigid edge {edge.label}"
            )
            require_same_handedness(
                edge.pose.handedness, pf.handedness, context=f"rigid edge {edge.label}"
            )
        elif edge.kind == "affine":
            # affine edges are exactly where a unit or handedness *change* is
            # legitimate -- that is their job (voxel->mm, mirror templates) --
            # but the change must be declared, not inferred.
            det = float(torch.linalg.det(edge.matrix[:3, :3]))
            if (pf.handedness is not cf.handedness) and not edge.reflection_declared:
                raise HandednessError(
                    f"affine edge {edge.label} joins a {cf.handedness.value}-handed "
                    f"frame to a {pf.handedness.value}-handed frame without "
                    "declaring a reflection",
                    remedy="Set reflection_declared=True and supply det<0, or fix the frames.",
                    offending_object=edge.label,
                )
            if (pf.handedness is cf.handedness) and det < 0:
                raise HandednessError(
                    f"affine edge {edge.label} has det={det:.6f} < 0 but joins two "
                    f"{pf.handedness.value}-handed frames",
                    remedy="A mirror must change the declared handedness of the frame.",
                    offending_object=edge.label,
                )

        idx = len(self._edges)
        self._edges.append(edge)
        self._adj.setdefault(edge.parent, []).append((edge.child, idx, False))
        self._adj.setdefault(edge.child, []).append((edge.parent, idx, True))
        return edge

    def add_rigid(
        self,
        parent: str,
        child: str,
        pose: Pose,
        *,
        uncertainty: PoseUncertainty | None = None,
        calibration: CalibrationRecord | None = None,
        **kw: Any,
    ) -> TransformEdge:
        return self.add_edge(
            TransformEdge(
                parent,
                child,
                "rigid",
                pose=pose,
                uncertainty=uncertainty,
                calibration=calibration or CalibrationRecord(),
                **kw,
            )
        )

    # -- path enumeration --------------------------------------------------

    def _enumerate(self, source: str, target: str) -> list[list[tuple[int, bool]]]:
        """All simple edge sequences from ``target`` frame data into ``source``.

        A path for ``path(a, b)`` is a chain of edges composing to ``T^{a<-b}``,
        i.e. a walk ``a -> ... -> b`` through the (undirected) adjacency where a
        traversal against an edge's declared direction requires inversion.
        """
        self.frame(source)
        self.frame(target)
        results: list[list[tuple[int, bool]]] = []
        stack: list[tuple[str, list[tuple[int, bool]], set[str]]] = [
            (source, [], {source})
        ]
        while stack:
            node, chain, seen = stack.pop()
            if len(chain) >= self.max_path_length:
                continue
            for nxt, idx, reverse in self._adj.get(node, []):
                if nxt in seen:
                    continue
                new_chain = chain + [(idx, reverse)]
                if nxt == target:
                    results.append(new_chain)
                    continue
                stack.append((nxt, new_chain, seen | {nxt}))
        results.sort(key=len)
        return results

    # -- path validation ---------------------------------------------------

    def path(
        self,
        a: str,
        b: str,
        *,
        at: float | None = None,
        expiry_policy: ExpiryPolicy | str | None = None,
        triggers_fired: Sequence[str] = (),
        require_linear: bool = False,
        raise_if_empty: bool = True,
    ) -> PathSet:
        """All admissible paths for ``T^{a<-b}``, validated and cross-checked.

        Every candidate is checked for composed units, handedness, epoch,
        support (edge kind), validity interval and calibration validity.  Failed
        candidates are kept in :attr:`PathSet.rejected` with their reason rather
        than being dropped: "there is no edge" and "there are two routes and
        both expired an hour ago" are different scientific situations.

        When *no* candidate survives, this raises.  If every candidate failed
        the same way the original refusal is re-raised unchanged -- an expired
        calibration must surface as :class:`CalibrationExpiredError`, not as a
        generic "no path".  Pass ``raise_if_empty=False`` to inspect the
        rejections instead.
        """
        policy = ExpiryPolicy.coerce(expiry_policy or self.expiry_policy)
        candidates = self._enumerate(a, b)
        if not candidates:
            raise NoPathError(
                f"no declared transform chain relates frame {b!r} to frame {a!r}",
                remedy=(
                    "Supply the missing calibrated edge, or keep the source "
                    "disconnected from the requested path (Table "
                    "tab:compiler-refusals, row 1). A nominal atlas coordinate "
                    "is not a physical location."
                ),
                offending_object=(a, b),
            )

        admissible: list[TransformPath] = []
        rejected: list[tuple[str, str]] = []
        errors: list[TransformError] = []
        for chain in candidates:
            edges = []
            label_parts = []
            try:
                for idx, reverse in chain:
                    e = self._edges[idx]
                    edges.append(e.reversed_edge() if reverse else e)
                    label_parts.append(e.label + ("^-1" if reverse else ""))
                tp = self._validate(
                    a, b, tuple(edges), at=at, policy=policy, triggers_fired=triggers_fired
                )
                if require_linear and not tp.linear:
                    raise TransformError(
                        f"path {tp.label} contains a deformable edge and cannot be "
                        "expressed as a single pose",
                        remedy=(
                            "§2.8: non-rigid warps retain local Jacobians and "
                            "residuals rather than being disguised as rigid pose "
                            "error. Use TransformPath.apply on points."
                        ),
                        offending_object=tp.label,
                    )
                admissible.append(tp)
            except TransformError as exc:
                rejected.append((" . ".join(label_parts), str(exc)))
                errors.append(exc)

        disagreements = self._compare(admissible)
        ps = PathSet(
            a,
            b,
            tuple(admissible),
            tuple(rejected),
            tuple(disagreements),
            at,
            tuple(errors),
        )
        if not admissible and raise_if_empty:
            kinds = {type(e) for e in errors}
            if len(kinds) == 1:
                raise errors[0]
            raise NoPathError(
                f"no admissible path from {b!r} to {a!r}: every candidate was "
                f"refused ({len(errors)} candidates)",
                remedy=(
                    "Each refusal is listed in PathSet.rejected with its own "
                    "reason and remedy; fix them individually."
                ),
                offending_object=rejected,
            )
        return ps

    def _validate(
        self,
        a: str,
        b: str,
        edges: tuple[TransformEdge, ...],
        *,
        at: float | None,
        policy: ExpiryPolicy,
        triggers_fired: Sequence[str],
    ) -> TransformPath:
        warnings: list[str] = []
        checks: list[ValidityCheck] = []

        # --- frame-definition validity (layer 1) --------------------------
        # This one is not policy-driven: a head frame defined by fiducials
        # simply does not exist after the participant stands up. There is no
        # "inflate the uncertainty" reading of a frame that is not defined.
        frame_validity = self.frame(a).validity
        for fid in [e.child for e in edges] + [a]:
            frame_validity = frame_validity.intersect(self.frame(fid).validity)
        if at is not None and not frame_validity.contains(at):
            from .errors import ValidityIntervalError

            raise ValidityIntervalError(
                f"path {' . '.join(x.label for x in edges)} crosses a frame whose "
                f"definition is not valid at t={at}: joint frame validity "
                f"{frame_validity}",
                remedy=(
                    "A frame definition has no uncertainty-inflation reading: "
                    "re-define the frame (re-digitize fiducials) or query inside "
                    "its validity interval."
                ),
                offending_object=(a, b, at),
            )
        validity = frame_validity
        for e in edges:
            validity = validity.intersect(e.calibration.validity)

        # --- calibration validity (layer 9), policy-driven ----------------
        if at is not None:
            for e in edges:
                chk = e.calibration.check(
                    at, policy=policy, label=e.label, triggers_fired=triggers_fired
                )
                checks.append(chk)
                if not chk.inside:
                    warnings.append(chk.reason)

        # --- units, handedness, epoch, support ----------------------------
        linear = all(e.is_linear for e in edges)
        units = self.frame(b).units
        hand = self.frame(b).handedness
        for e in reversed(edges):
            pf, cf = self.frame(e.parent), self.frame(e.child)
            require_same_unit(
                units, cf.units, context=f"path step {e.label} (incoming units)"
            )
            if e.kind == "rigid":
                require_same_unit(pf.units, cf.units, context=f"rigid edge {e.label}")
                require_same_handedness(
                    hand, cf.handedness, context=f"path step {e.label}"
                )
                require_same_handedness(
                    pf.handedness, cf.handedness, context=f"rigid edge {e.label}"
                )
            units = pf.units
            hand = pf.handedness
        require_same_unit(units, self.frame(a).units, context=f"composed path into {a}")
        require_same_handedness(
            hand, self.frame(a).handedness, context=f"composed path into {a}"
        )

        # --- composition (a derived view; edges stay in the graph) --------
        pose: Pose | None = None
        matrix: torch.Tensor | None = None
        if linear:
            M = torch.eye(4, dtype=DTYPE)
            for e in edges:
                M = M @ e.as_matrix()
            matrix = M
            if all(e.kind == "rigid" for e in edges):
                p = edges[0].pose
                for e in edges[1:]:
                    p = p.compose(e.pose, at=at)
                pose = p

        # --- uncertainty ---------------------------------------------------
        chain_unc: ChainUncertainty | None = None
        if all(e.kind == "rigid" for e in edges) and all(
            e.uncertainty is not None for e in edges
        ):
            poses = [e.pose for e in edges]
            uncs = []
            for e, chk in zip(edges, checks or [None] * len(edges)):
                u = e.uncertainty
                if chk is not None and not chk.inside:
                    u = PoseUncertainty(
                        cov=chk.apply(u.cov),
                        bias=u.bias,
                        calibration_source=u.calibration_source,
                        sensitivity=(
                            None
                            if u.sensitivity is None
                            else u.sensitivity * math.sqrt(chk.inflation_factor)
                        ),
                    )
                uncs.append(u)
            chain_unc = propagate_chain(
                poses, uncs, shared_covariances=self.shared_covariances
            )

        return TransformPath(
            source=a,
            target=b,
            edges=edges,
            pose=pose,
            matrix=matrix,
            uncertainty=chain_unc,
            validity=validity,
            validity_checks=tuple(checks),
            units=self.frame(a).units,
            handedness=self.frame(a).handedness,
            linear=linear,
            warnings=tuple(warnings),
            provenance={
                "edges_stored_separately": True,
                "composition_is_a_view": True,
                "kinds": [e.kind for e in edges],
                "calibration": [
                    {
                        "edge": e.label,
                        "method": e.calibration.method,
                        "n_observations": e.calibration.n_observations,
                        "residual_rms": e.calibration.residual_rms,
                    }
                    for e in edges
                ],
            },
        )

    def _compare(self, paths: Sequence[TransformPath]) -> list[PathDisagreement]:
        out: list[PathDisagreement] = []
        posed = [p for p in paths if p.pose is not None]
        for i in range(len(posed)):
            for j in range(i + 1, len(posed)):
                pi, pj = posed[i], posed[j]
                xi = pi.pose.residual_twist(pj.pose)
                tr = float(torch.linalg.norm(xi[:3]))
                rot = float(torch.linalg.norm(xi[3:]))
                maha = None
                agree = True
                if pi.uncertainty is not None and pj.uncertainty is not None:
                    S = pi.uncertainty.cov + pj.uncertainty.cov
                    S = S + 1e-18 * torch.eye(6, dtype=DTYPE)
                    maha = float(xi @ torch.linalg.solve(S, xi))
                    agree = maha <= self.agreement_chi2
                out.append(
                    PathDisagreement(
                        pi.label, pj.label, tr, rot, maha, agree, pi.units
                    )
                )
        return out

    # -- round trip --------------------------------------------------------

    def round_trip(
        self, a: str, b: str, *, at: float | None = None, points: Any | None = None, **kw: Any
    ) -> dict[str, Any]:
        """Measure ``a -> b -> a``.  Never assumed zero.

        For a linear path the residual is the twist of the composed loop.  For a
        path containing a deformable edge the residual is measured on supplied
        points, which is the only honest thing to do: a warp's round-trip error
        is a field, not a scalar.
        """
        fwd = self.path(a, b, at=at, **kw).best
        bwd = self.path(b, a, at=at, **kw).best
        # "independent" means the two directions did not simply invert the same
        # edges: only then is the residual evidence about calibration rather
        # than about floating point.
        fwd_edges = sorted(tuple(sorted((e.parent, e.child))) for e in fwd.edges)
        bwd_edges = sorted(tuple(sorted((e.parent, e.child))) for e in bwd.edges)
        rec: dict[str, Any] = {
            "forward_path": fwd.label,
            "backward_path": bwd.label,
            "units": fwd.units,
            "independent_paths": fwd_edges != bwd_edges,
        }
        if fwd.matrix is not None and bwd.matrix is not None:
            loop = fwd.matrix @ bwd.matrix
            R = loop[:3, :3]
            rec["translation_residual"] = float(torch.linalg.norm(loop[:3, 3]))
            rec["rotation_residual_rad"] = float(torch.linalg.norm(log_so3(_project_so3(R))))
            rec["scale_residual"] = abs(float(torch.linalg.det(R)) - 1.0)
        if points is not None:
            p = torch.as_tensor(points, dtype=DTYPE)
            back = bwd.apply(fwd.apply(p))
            err = torch.linalg.norm(back - p, dim=-1)
            rec["point_residual_rms"] = float(torch.sqrt((err**2).mean()))
            rec["point_residual_max"] = float(err.max())
            rec["n_points"] = int(p.reshape(-1, 3).shape[0])
        return rec


def _project_so3(R: torch.Tensor) -> torch.Tensor:
    U, _, Vh = torch.linalg.svd(R)
    Rp = U @ Vh
    if float(torch.linalg.det(Rp)) < 0:  # pragma: no cover - loop of proper maps
        U = U.clone()
        U[:, -1] = -U[:, -1]
        Rp = U @ Vh
    return Rp


# --------------------------------------------------------------------------
# the worked chain of equation (3)
# --------------------------------------------------------------------------

#: Frame ids of the device-to-atlas chain in thesis §2.8 equation (3).
DEVICE_TO_ATLAS_FRAMES: tuple[str, ...] = ("atlas", "image", "head", "tracker", "device")


def device_to_atlas_chain(
    graph: FrameGraph, *, at: float | None = None, **kw: Any
) -> TransformPath:
    """Equation (3): ``T^{atlas<-device} = T^{atlas<-image} T^{image<-head}
    T^{head<-tracker} T^{tracker<-device}``.

    Returns the *path*, not a bare matrix, so the four edges stay visible: the
    thesis is explicit that they are "stored separately rather than multiplied
    and forgotten".
    """
    ps = graph.path("atlas", "device", at=at, **kw)
    expected = [
        ("atlas", "image"),
        ("image", "head"),
        ("head", "tracker"),
        ("tracker", "device"),
    ]
    for p in ps.paths:
        if [(e.parent, e.child) for e in p.edges] == expected:
            return p
    raise NoPathError(
        "the canonical atlas<-image<-head<-tracker<-device chain of equation (3) "
        "is not present in this graph",
        remedy=(
            "Declare the four edges: atlas normalization, head->MRI "
            "coregistration, tracker->head fiducial calibration, and the "
            "device (coil/transducer) geometry."
        ),
        offending_object=[p.label for p in ps.paths] or ps.rejected,
    )


__all__ = [
    "Frame",
    "TransformEdge",
    "DeformableTransform",
    "TransformPath",
    "PathSet",
    "PathDisagreement",
    "FrameGraph",
    "DEVICE_TO_ATLAS_FRAMES",
    "device_to_atlas_chain",
]
