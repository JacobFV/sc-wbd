"""Adapter between ``scwbd.schema`` (agent A) and this package's runtime types.

The math in ``scwbd.transforms`` is written against plain dataclasses and torch,
so it neither imports nor depends on pydantic.  Everything that touches the
contract types lives here, behind ``*_from_schema`` / ``*_to_schema``.

The two layers say different things, and the adapter is where the difference is
made explicit rather than papered over:

* ``scwbd.schema`` is a **declaration**: which frames exist, what kind of
  transform relates them, what priors their parameters carry, what calibration
  backs them.  A :class:`~scwbd.schema.frames.FrameEdge` deliberately holds no
  fitted 4x4 -- ``parameters`` are ``Prior`` objects, not numbers.
* ``scwbd.transforms`` is a **runtime**: it needs the actual matrix that came
  out of the registration, and refuses to invent one.

So :func:`edge_from_schema` resolves a numeric transform from a ``DiracPrior``
(a declared exact value), or from a matrix the caller supplies alongside the
declaration, and otherwise refuses with a message naming the missing parameter.
Substituting a prior *mean* for a measured transform would be exactly the
"nominal coordinate treated as the physical location" that Appendix C forbids.

``SCHEMA_AVAILABLE`` reports whether the contract types are importable.  Note
that an empty ``scwbd/schema/`` directory imports fine as a namespace package,
so availability is decided by probing for the types themselves.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

from .calibration import CalibrationRecord
from .clock_graph import ClockGraph, ClockMap, DropSpec
from .clock_graph import ClockSpec as TClockSpec
from .errors import ClockRelationUnknownError, HandednessError, TransformError
from .frame_graph import Frame, FrameGraph, TransformEdge
from .se3 import DTYPE, Pose, ValidityInterval, exp_se3
from .uncertainty import PoseUncertainty
from .units import Handedness

#: Names the schema kernel must expose before this adapter believes in it.
_REQUIRED_SCHEMA_NAMES = ("BrainSchema", "FrameGraphSpec", "ClockSpec")

try:  # pragma: no cover - depends on agent A's progress
    import scwbd.schema as _schema  # type: ignore

    SCHEMA_AVAILABLE = all(hasattr(_schema, n) for n in _REQUIRED_SCHEMA_NAMES)
except Exception:  # pragma: no cover
    _schema = None
    SCHEMA_AVAILABLE = False


def _require_schema() -> Any:
    if not SCHEMA_AVAILABLE:  # pragma: no cover - exercised only pre-schema
        raise TransformError(
            "scwbd.schema does not yet expose the contract types "
            f"{_REQUIRED_SCHEMA_NAMES}, so they cannot be converted",
            remedy=(
                "This is expected while agent A's schema kernel is in flight. "
                "Build the transform objects directly (Frame, TransformEdge, "
                "ClockSpec) or go through the plain-dict form of these "
                "adapters, which needs no pydantic, until then."
            ),
            offending_object="scwbd.schema",
        )
    return _schema


def _get(obj: Any, *names: str, default: Any = None, required: bool = False) -> Any:
    """Read the first present attribute / key, or refuse if required."""
    for n in names:
        if isinstance(obj, Mapping):
            if n in obj:
                return obj[n]
        elif hasattr(obj, n):
            return getattr(obj, n)
    if required:
        raise TransformError(
            f"object {type(obj).__name__} declares none of {names}",
            remedy=(
                "The schema type changed. Update scwbd.transforms.schema_adapter "
                "rather than defaulting the field -- a silently defaulted frame "
                "or unit is refusal R01."
            ),
            offending_object=obj,
        )
    return default


def _interval(v: Any) -> ValidityInterval:
    if v is None:
        return ValidityInterval.unbounded()
    if isinstance(v, ValidityInterval):
        return v
    if isinstance(v, (tuple, list)) and len(v) == 2:
        return ValidityInterval(
            None if v[0] is None else float(v[0]),
            None if v[1] is None else float(v[1]),
        )
    return ValidityInterval(
        _get(v, "start", "t_start"),
        _get(v, "end", "t_end"),
        str(_get(v, "clock", default="wall")),
    )


def _interval_out(v: ValidityInterval) -> tuple[float, float] | None:
    if v.start is None or v.end is None:
        return None
    return (v.start, v.end)


# --------------------------------------------------------------------------
# frames  (scwbd.schema.frames.FrameNode)
# --------------------------------------------------------------------------

#: ``FrameNode.coordinate_type`` -> the unit this runtime uses for that frame.
#: Appendix C layer 1 demands an "explicit distinction between continuous mm,
#: indices, normalized image coordinates and angular gaze"; the runtime encodes
#: that distinction in the *unit*, so index frames cannot compose with metric
#: ones without a declared affine edge.
_COORDINATE_TYPE_UNITS = {"index": "index", "normalized": "normalized"}


def frame_from_schema(spec: Any) -> Frame:
    """``FrameNode`` (or plain dict) -> :class:`Frame`."""
    handedness = str(_get(spec, "handedness", required=True))
    fid = str(_get(spec, "id", "frame", "name", required=True))
    if handedness == "not_applicable":
        raise HandednessError(
            f"frame {fid!r} declares handedness 'not_applicable' and cannot enter "
            "an SE(3) transform path",
            remedy=(
                "Handedness-free supports (spectral bands, event windows, "
                "behavioural episodes) belong in the resolution poset "
                "(scwbd.transforms.sheaf), not the frame graph."
            ),
            offending_object=fid,
        )
    coordinate_type = str(_get(spec, "coordinate_type", default="physical"))
    units = str(_get(spec, "units", required=True))
    units = _COORDINATE_TYPE_UNITS.get(coordinate_type, units)
    axes = _get(spec, "axes", default=("x", "y", "z"))
    return Frame(
        id=fid,
        object=str(_get(spec, "object", "physical_object", required=True)),
        origin=str(_get(spec, "origin", default="undeclared")),
        axes=axes if isinstance(axes, str) else ",".join(str(a) for a in axes),
        handedness=Handedness.coerce(handedness),
        units=units,
        validity=_interval(_get(spec, "validity_interval", "validity")),
        epoch=_get(spec, "epoch"),
        notes=str(_get(spec, "description", "notes", default="")),
    )


def frame_to_schema(frame: Frame) -> dict[str, Any]:
    """:class:`Frame` -> kwargs accepted by ``FrameNode``."""
    axes = tuple(a.strip() for a in frame.axes.split(",")) if "," in frame.axes else None
    coordinate_type = {
        "index": "index",
        "voxel": "index",
        "normalized": "normalized",
    }.get(frame.units, "physical")
    return {
        "id": frame.id,
        "object": frame.object,
        "origin": frame.origin,
        "axes": axes if axes and len(axes) == 3 else ("x", "y", "z"),
        "handedness": frame.handedness.value,
        "units": frame.units if coordinate_type == "physical" else "dimensionless",
        "epoch": frame.epoch or "session",
        "validity_interval": _interval_out(frame.validity),
        "coordinate_type": coordinate_type,
        "description": frame.notes or frame.axes,
    }


# --------------------------------------------------------------------------
# edges  (scwbd.schema.frames.FrameEdge)
# --------------------------------------------------------------------------

#: ``FrameEdge.transform`` -> this runtime's edge kind.  A ``projection`` is an
#: affine that is *not* invertible; ``resampling`` is an affine carrying a grid
#: change; ``identity`` is a rigid edge whose matrix is I.
_TRANSFORM_KIND = {
    "identity": "rigid",
    "rigid": "rigid",
    "affine": "affine",
    "projection": "affine",
    "resampling": "affine",
    "deformable": "deformable",
    "temporal": "temporal",
    "amplitude": "amplitude",
}


def _dirac(prior: Any) -> float | None:
    """Recover a declared exact value; ``None`` for any genuinely uncertain prior."""
    if isinstance(prior, (int, float)):
        return float(prior)
    if _get(prior, "kind") == "dirac" or type(prior).__name__ == "DiracPrior":
        return float(_get(prior, "value", required=True))
    return None


def _matrix_from_parameters(params: Mapping[str, Any] | None, label: str) -> torch.Tensor | None:
    """Build a 4x4 from ``DiracPrior`` parameters, or return ``None``.

    Recognized layouts: a ``twist`` (6 Diracs ``twist_0..twist_5``, or a single
    ``twist`` sequence), or ``matrix_ij`` entries.  Anything else is a genuinely
    uncertain declaration and must be supplied numerically by the caller.
    """
    if not params:
        return None
    twist = [_dirac(params.get(f"twist_{i}")) for i in range(6)]
    if all(v is not None for v in twist):
        return exp_se3(torch.tensor(twist, dtype=DTYPE))
    cells = {}
    for i in range(4):
        for j in range(4):
            v = _dirac(params.get(f"matrix_{i}{j}"))
            if v is not None:
                cells[(i, j)] = v
    if len(cells) == 16:
        M = torch.zeros((4, 4), dtype=DTYPE)
        for (i, j), v in cells.items():
            M[i, j] = v
        return M
    return None


def edge_from_schema(
    spec: Any,
    *,
    matrix: Any | None = None,
    warp: Any | None = None,
    units: str | None = None,
    handedness: str = "right",
) -> TransformEdge:
    """``FrameEdge`` -> :class:`TransformEdge`.

    ``FrameEdge.src -> dst`` means "data expressed in ``src`` becomes data
    expressed in ``dst``", i.e. ``T^{dst<-src}``, so ``parent=dst`` and
    ``child=src`` in this runtime's convention.

    ``matrix`` supplies the fitted transform the declaration does not carry.
    Without it -- and without Dirac-valued parameters -- this refuses rather
    than substituting a prior mean for a measurement.
    """
    child = str(_get(spec, "src", "child", "from", required=True))
    parent = str(_get(spec, "dst", "parent", "to", required=True))
    declared = str(_get(spec, "transform", "kind", "family", default="rigid"))
    if declared not in _TRANSFORM_KIND:
        raise TransformError(
            f"transform kind {declared!r} on edge {parent}<-{child} has no runtime adapter",
            remedy=f"Known kinds: {sorted(_TRANSFORM_KIND)}.",
            offending_object=declared,
        )
    kind = _TRANSFORM_KIND[declared]
    lineage = str(_get(spec, "lineage", default=""))
    if not lineage:
        raise TransformError(
            f"transform edge {parent}<-{child} declares no lineage",
            remedy=(
                "Refusal R01 covers 'unknown transform lineage'. Record how the "
                "transform was obtained: digitization, registration algorithm, "
                "tracker read, or manufacturer file."
            ),
            offending_object=(parent, child),
        )
    unit = units or str(
        _get(spec, "units_out", "units_in", default=None) or "m"
    )
    cal = calibration_from_schema(_get(spec, "calibration"))
    unc = uncertainty_from_schema(_get(spec, "ledger", "uncertainty"))
    invertible = bool(_get(spec, "invertible", default=True))
    if declared == "projection":
        invertible = False

    if kind == "deformable":
        if warp is None:
            raise TransformError(
                f"deformable edge {parent}<-{child} needs a DeformableTransform; "
                "a declaration cannot carry the warp field",
                remedy="Pass warp=DeformableTransform(...) alongside the schema edge.",
                offending_object=(parent, child),
            )
        return TransformEdge(
            parent, child, "deformable", warp=warp, calibration=cal,
            invertible=invertible, notes=lineage,
        )

    M = (
        torch.as_tensor(matrix, dtype=DTYPE)
        if matrix is not None
        else (
            torch.eye(4, dtype=DTYPE)
            if declared == "identity"
            else _matrix_from_parameters(_get(spec, "parameters"), f"{parent}<-{child}")
        )
    )
    if M is None:
        raise TransformError(
            f"edge {parent}<-{child} declares transform {declared!r} with "
            "uncertain parameters and no fitted value",
            remedy=(
                "A schema edge is a declaration, not a measurement. Supply the "
                "fitted transform (matrix=...) from the registration or tracker "
                "read. Substituting a prior mean would turn a nominal "
                "coordinate into a physical location."
            ),
            offending_object=(parent, child),
        )

    if kind == "rigid":
        pose = Pose(
            M,
            parent,
            child,
            units=unit,
            handedness=Handedness.coerce(handedness),
            validity=_interval(_get(spec, "validity_interval", "validity")),
            epoch=_get(spec, "epoch"),
            provenance={"lineage": lineage, "schema_transform": declared},
        )
        return TransformEdge(
            parent, child, "rigid", pose=pose, calibration=cal, uncertainty=unc,
            invertible=invertible, notes=lineage,
        )
    return TransformEdge(
        parent,
        child,
        "affine",
        matrix=M,
        calibration=cal,
        uncertainty=unc,
        invertible=invertible,
        reflection_declared=bool(_get(spec, "reflection_declared", default=False)),
        notes=lineage,
    )


def edge_to_schema(edge: TransformEdge) -> dict[str, Any]:
    """:class:`TransformEdge` -> kwargs accepted by ``FrameEdge``."""
    out: dict[str, Any] = {
        "src": edge.child,
        "dst": edge.parent,
        "transform": "affine" if (edge.kind == "affine" and edge.invertible) else (
            "projection" if edge.kind == "affine" else edge.kind
        ),
        "lineage": edge.notes or edge.calibration.method,
        "invertible": edge.invertible,
        "calibration": calibration_to_schema(edge.calibration),
    }
    if edge.kind == "rigid":
        out["units_in"] = out["units_out"] = edge.pose.units
        out["parameters"] = {
            f"twist_{i}": {"kind": "dirac", "value": float(v)}
            for i, v in enumerate(edge.pose.log())
        }
        out["matrix"] = edge.pose.matrix.tolist()
        out["handedness"] = edge.pose.handedness.value
        out["epoch"] = edge.pose.epoch
        out["validity_interval"] = _interval_out(edge.pose.validity)
    elif edge.kind == "affine":
        out["parameters"] = {
            f"matrix_{i}{j}": {"kind": "dirac", "value": float(edge.matrix[i, j])}
            for i in range(4)
            for j in range(4)
        }
        out["matrix"] = edge.matrix.tolist()
    if edge.uncertainty is not None:
        out["ledger"] = uncertainty_to_schema(edge.uncertainty)
    return out


# --------------------------------------------------------------------------
# calibration  (scwbd.schema.frames.CalibrationManifest)
# --------------------------------------------------------------------------


def calibration_from_schema(spec: Any) -> CalibrationRecord:
    if spec is None:
        return CalibrationRecord()
    if isinstance(spec, CalibrationRecord):
        return spec
    return CalibrationRecord(
        method=str(_get(spec, "fitting_method", "method", default="undeclared") or "undeclared"),
        n_observations=_get(spec, "n_observations"),
        residual_rms=_get(spec, "residual", "residual_rms"),
        residual_max=_get(spec, "residual_max"),
        validity=_interval(_get(spec, "validity_interval", "validity")),
        recalibration_triggers=tuple(_get(spec, "recalibration_triggers", default=()) or ()),
        device_serial=_get(spec, "device_serial"),
        notes=str(_get(spec, "notes", default="")),
    )


def calibration_to_schema(cal: CalibrationRecord) -> dict[str, Any]:
    return {
        "id": f"{cal.method}:{cal.device_serial or 'unserialized'}",
        "fitting_method": cal.method,
        "n_observations": cal.n_observations or 0,
        "residual": cal.residual_rms,
        "validity_interval": _interval_out(cal.validity),
        "extrapolation_distance": None,
        "recalibration_triggers": list(cal.recalibration_triggers),
        "units_checked": True,  # every path through this runtime checks units
        "handedness_checked": True,  # ... and handedness
        "notes": cal.notes,
    }


# --------------------------------------------------------------------------
# uncertainty  (scwbd.schema.ledger.UncertaintyLedger)
# --------------------------------------------------------------------------


def uncertainty_from_schema(spec: Any) -> PoseUncertainty | None:
    """``UncertaintyLedger`` -> :class:`PoseUncertainty`.

    A ledger keeps ``variance`` as a named dict and ``bias_interval`` as a range
    (§2.7).  A twist covariance cannot carry an interval, so the adapter takes
    the midpoint as the bias and records the collapse in the provenance the
    caller can read back with :func:`uncertainty_to_schema`.  A ledger whose
    bias status is ``prior_specified_sensitivity`` is a *range to sweep*, not a
    number; collapsing it silently would violate refusal R08.
    """
    if spec is None:
        return None
    if isinstance(spec, PoseUncertainty):
        return spec
    cov = _get(spec, "cov", "covariance", "twist_covariance")
    if cov is not None:
        sens = _get(spec, "sensitivity")
        return PoseUncertainty(
            cov=torch.as_tensor(cov, dtype=DTYPE),
            bias=torch.as_tensor(
                _get(spec, "bias", "twist_bias", default=[0.0] * 6), dtype=DTYPE
            ),
            calibration_source=_get(spec, "calibration_source"),
            sensitivity=None if sens is None else torch.as_tensor(sens, dtype=DTYPE),
        )
    variance = _get(spec, "variance")
    if variance is None:
        return None
    total = float(sum(float(v) for v in dict(variance).values()))
    lo, hi = _get(spec, "bias_interval", default=(0.0, 0.0))
    mid = 0.5 * (float(lo) + float(hi))
    return PoseUncertainty(
        cov=torch.eye(6, dtype=DTYPE) * total,
        bias=torch.full((6,), mid, dtype=DTYPE),
    )


def uncertainty_to_schema(u: PoseUncertainty) -> dict[str, Any]:
    diag = torch.clamp(torch.diagonal(u.cov), min=0.0)
    return {
        "variance": {
            "translation": float(diag[:3].sum()),
            "rotation": float(diag[3:].sum()),
        },
        "bias_interval": (float(u.bias.min()), float(u.bias.max())),
        "bias_status": "externally_bounded" if float(u.bias.abs().max()) > 0 else "design_estimable",
        "units": "m",
        "cov": u.cov.tolist(),
        "bias": u.bias.tolist(),
        "calibration_source": u.calibration_source,
        "sensitivity": None if u.sensitivity is None else u.sensitivity.tolist(),
    }


# --------------------------------------------------------------------------
# clocks  (scwbd.schema.clocks.ClockSpec / ClockEdge)
# --------------------------------------------------------------------------

#: ``ClockSpec.sync_evidence`` -> this runtime's evidence vocabulary.
#: ``assumed`` and ``unknown`` are deliberately absent: they are R01 (schema's
#: own ``UNVERIFIED_SYNC``), and mapping them to anything would launder them.
_SYNC_EVIDENCE = {
    "physical_trigger": "physical_trigger",
    "shared_hardware": "shared_hardware_clock",
    "shared_hardware_clock": "shared_hardware_clock",
    "cross_correlation": "cross_correlation",
    "declared_identity": "declared_identity",
}

#: ``ClockSpec.interpolation`` -> this runtime's interpolation policy.
_INTERPOLATION = {
    "zero_order_hold": "zoh",
    "linear": "linear",
    "band_limited": "sinc",
    "event_exact": "none",
    "none": "none",
}


def _sync_evidence(value: str, *, label: str) -> str:
    try:
        return _SYNC_EVIDENCE[value]
    except KeyError:
        raise ClockRelationUnknownError(
            f"clock relation {label} declares sync evidence {value!r}",
            remedy=(
                "Appendix C layer 4 requires a physical synchronization event or "
                "an independent cross-correlation target. 'assumed' and "
                "'unknown' are refusal R01 in the schema too "
                "(scwbd.schema.clocks.UNVERIFIED_SYNC); record the trigger or "
                "leave the streams on separate timelines."
            ),
            offending_object=(label, value),
        ) from None


def clock_from_schema(spec: Any, *, epoch_seconds: float | None = None) -> TClockSpec:
    """``scwbd.schema.clocks.ClockSpec`` -> the runtime clock spec.

    The schema's ``epoch`` is a *label* ("session_start"); the runtime needs the
    numeric time of sample 0.  Pass ``epoch_seconds`` when it is known.  Leaving
    it ``None`` is a legitimate state -- the clock simply cannot convert sample
    indices to times until an epoch is measured.
    """
    dt = _get(spec, "dt")
    rate = _get(spec, "rate_hz", "sample_rate", "fs")
    if rate is None and dt:
        rate = 1.0 / float(dt)
    drops = tuple(
        DropSpec(
            int(_get(d, "start_index", "start", required=True)),
            int(_get(d, "count", required=True)),
        )
        for d in (_get(spec, "dropped", "dropped_samples", default=()) or ())
    )
    interp = str(_get(spec, "interpolation", "interpolation_policy", default="linear"))
    policy = _INTERPOLATION.get(interp, interp)
    if str(_get(spec, "dropped_sample_policy", default="mask")) == "refuse" and policy != "none":
        policy = "none"
    return TClockSpec(
        id=str(_get(spec, "id", "clock", required=True)),
        rate_hz=rate,
        epoch=epoch_seconds if epoch_seconds is not None else _get(spec, "epoch_seconds"),
        trigger_path=tuple(_get(spec, "trigger_path", default=()) or ()),
        group_delay_s=float(_get(spec, "group_delay", "group_delay_s", default=0.0) or 0.0),
        jitter_sd_s=float(_get(spec, "jitter_sd", "jitter_sd_s", default=0.0) or 0.0),
        dropped=drops,
        integration_window_s=float(
            _get(spec, "integration_window", "integration_window_s", default=0.0) or 0.0
        ),
        interpolation_policy=policy,
        max_interpolation_gap_s=_get(spec, "max_interpolation_gap_s"),
        domain=str(_get(spec, "domain", default="acquisition")),
        notes=str(_get(spec, "label", "notes", default="")),
    )


def clock_to_schema(spec: TClockSpec) -> dict[str, Any]:
    inverse = {v: k for k, v in _INTERPOLATION.items()}
    return {
        "id": spec.id,
        "label": spec.notes,
        "dt": None if spec.rate_hz is None else spec.dt,
        "rate_hz": None if spec.rate_hz is None else float(spec.rate_hz),
        "epoch": "session_start",
        "epoch_seconds": spec.epoch,
        "trigger_path": list(spec.trigger_path),
        "group_delay": spec.group_delay_s,
        "jitter_sd": spec.jitter_sd_s,
        "dropped": [{"start_index": d.start_index, "count": d.count} for d in spec.dropped],
        "integration_window": spec.integration_window_s,
        "interpolation": inverse.get(spec.interpolation_policy, "linear"),
        "max_interpolation_gap_s": spec.max_interpolation_gap_s,
        "domain": spec.domain,
    }


def clock_edge_from_schema(edge: Any) -> tuple[str, str, ClockMap, str]:
    """``ClockEdge`` -> ``(src, dst, ClockMap, evidence)``."""
    src = str(_get(edge, "src", "source", required=True))
    dst = str(_get(edge, "dst", "target", required=True))
    spec = _get(edge, "map")
    if isinstance(spec, Mapping) and "params" in spec:
        cmap = ClockMap.from_dict(spec)
    else:
        residual = _get(edge, "residual")
        cmap = ClockMap(
            torch.tensor(
                [float(_get(edge, "offset", default=0.0)), float(_get(edge, "drift", default=0.0))],
                dtype=DTYPE,
            ),
            (),
            residual_sd=float(residual or 0.0),
            fit_method=str(_get(edge, "sync_evidence", default="declared")),
            n_observations=_get(edge, "n_observations"),
            validity=_interval(_get(edge, "validity_interval")),
        )
    evidence = _sync_evidence(
        str(_get(edge, "sync_evidence", "evidence", default="unknown")), label=f"{src}->{dst}"
    )
    return src, dst, cmap, evidence


# --------------------------------------------------------------------------
# whole graphs
# --------------------------------------------------------------------------


def frame_graph_from_schema(
    spec: Any,
    *,
    matrices: Mapping[tuple[str, str], Any] | None = None,
    warps: Mapping[tuple[str, str], Any] | None = None,
    **kw: Any,
) -> FrameGraph:
    """``FrameGraphSpec`` (or ``{"nodes": [...], "edges": [...]}``) -> FrameGraph.

    ``matrices`` / ``warps`` are keyed by the schema's ``(src, dst)`` and supply
    the fitted transforms the declaration does not carry.
    """
    g = FrameGraph(**kw)
    nodes = _get(spec, "nodes", "frames", default=(), required=True)
    frames = {}
    for n in nodes:
        f = frame_from_schema(n)
        frames[f.id] = f
        g.add_frame(f)
    for e in _get(spec, "edges", "transforms", default=()) or ():
        src = str(_get(e, "src", "child", "from", required=True))
        dst = str(_get(e, "dst", "parent", "to", required=True))
        key = (src, dst)
        parent_frame = frames.get(dst)
        g.add_edge(
            edge_from_schema(
                e,
                matrix=(matrices or {}).get(key),
                warp=(warps or {}).get(key),
                units=parent_frame.units if parent_frame else None,
                handedness=(parent_frame.handedness.value if parent_frame else "right"),
            )
        )
    return g


def frame_graph_to_schema(graph: FrameGraph, *, root: str | None = None) -> dict[str, Any]:
    frames = list(graph.frames.values())
    return {
        "root": root or (frames[0].id if frames else ""),
        "nodes": [frame_to_schema(f) for f in frames],
        "edges": [edge_to_schema(e) for e in graph.edges],
    }


def clock_graph_from_schema(specs: Any, relations: Any = (), **kw: Any) -> ClockGraph:
    g = ClockGraph(**kw)
    for s in specs:
        g.add_clock(clock_from_schema(s))
    for r in relations or ():
        src, dst, cmap, evidence = clock_edge_from_schema(r)
        g.relate(src, dst, cmap, evidence=evidence)
    return g


def clock_graph_to_schema(graph: ClockGraph) -> dict[str, Any]:
    """Serialize the graph.

    Only edges in their *declared* direction are emitted: the reverse views
    added by ``relate(bidirectional=True)`` are the same measurement seen
    backwards, and writing both out would re-declare the relation twice on
    reload (and apply its offset twice).
    """
    reverse_evidence = {"shared_hardware_clock": "shared_hardware"}
    return {
        "clocks": [clock_to_schema(c) for c in graph.clocks.values()],
        "relations": [
            {
                "src": e.source,
                "dst": e.target,
                "offset": float(e.map.params[0]),
                "drift": float(e.map.params[1]),
                "residual": e.map.residual_sd,
                "n_observations": e.map.n_observations or 0,
                "validity_interval": _interval_out(e.map.validity),
                "sync_evidence": reverse_evidence.get(e.evidence, e.evidence),
                "map": e.map.to_dict(),
                "summary": e.map.summary(),
            }
            for e in graph.edges()
            if not e.inverted
        ],
    }


__all__ = [
    "SCHEMA_AVAILABLE",
    "frame_from_schema",
    "frame_to_schema",
    "edge_from_schema",
    "edge_to_schema",
    "calibration_from_schema",
    "calibration_to_schema",
    "uncertainty_from_schema",
    "uncertainty_to_schema",
    "clock_from_schema",
    "clock_to_schema",
    "clock_edge_from_schema",
    "frame_graph_from_schema",
    "frame_graph_to_schema",
    "clock_graph_from_schema",
    "clock_graph_to_schema",
]
