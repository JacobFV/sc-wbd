"""Transform, frame and clock runtime for SC-WBD-001-beta.

Build-order item 3 (``thesis_contract.tex`` §0.6): "frame-graph composition,
cross-covariance, multirate events, validity intervals, and obstruction
certificates."

This is the layer that makes cross-method fusion legitimate rather than a
numerical coincidence.  Nothing here computes dynamics or physics; it decides
whether two numbers from two instruments are talking about the same place, the
same instant, and the same quantity -- and refuses when they are not.

Modules
-------
``se3``
    SE(3) Lie algebra plus :class:`~scwbd.transforms.se3.Pose`, which carries a
    frame pair, units, handedness, epoch and validity interval.
``frame_graph``
    Frames and calibrated edges (rigid / affine / deformable / temporal /
    amplitude).  Edges are stored separately and never replaced by their
    composition; paths are validated and cross-checked; round trips are measured.
``uncertainty``
    Equation (T5) *with* its cross-covariance terms, adjoint propagation through
    SE(3) chains, Monte-Carlo and interval propagation for nonlinear maps.
``clock_graph``
    Appendix C layer 4: clock identity, rate, epoch, trigger path, offset,
    piecewise-affine drift, jitter, dropped samples, group delay, multirate
    scheduling and interpolation contracts.
``sheaf``
    The resolution poset of §2.6 made operational: restrictions, covers,
    overlap and cocycle residuals, gluing, obstruction certificates, and
    restriction/prolongation pairs.
``calibration``
    Calibration records and the validity-interval expiry policy (layer 9).
``errors``
    Every refusal, each with a reason, a remedy and the offending object.

Refusals implemented here, by ``tab:compiler-refusals`` code:

* **R01** -- unknown/mismatched units, frames, handedness, clock relations,
  epochs, expired calibration, missing inverse.
* **R02** -- prolongation without a restriction partner and tested coverage.
* **R03** -- global cross-scale state when the overlap/cocycle residual exceeds
  tolerance; an obstruction certificate naming the failed path is emitted
  instead.
"""

from __future__ import annotations

from . import calibration, clock_graph, errors, frame_graph, se3, sheaf, uncertainty, units
from .calibration import CalibrationRecord, ExpiryPolicy, ValidityCheck
from .clock_graph import (
    ClockGraph,
    ClockMap,
    ClockSpec,
    DropSpec,
    InterpolationContract,
    MultirateSchedule,
    SyncPoint,
    detect_dropped_samples,
    fit_clock_map,
    schedule_multirate,
)
from .errors import (
    CalibrationExpiredError,
    ClockRelationUnknownError,
    CocycleObstructionError,
    FrameMismatchError,
    HandednessError,
    LinearizationInvalidError,
    NoPathError,
    NonInvertibleTransformError,
    ProlongationWithoutRestrictionError,
    TransformError,
    UnitMismatchError,
)
from .frame_graph import (
    DeformableTransform,
    Frame,
    FrameGraph,
    PathSet,
    TransformEdge,
    TransformPath,
    device_to_atlas_chain,
)
from .se3 import Pose, ValidityInterval, adjoint, exp_se3, log_se3, round_trip_residual
from .sheaf import (
    Cover,
    FineDistribution,
    GlobalSection,
    ObstructionCertificate,
    Prolongation,
    Restriction,
    ScalePair,
    Section,
    Site,
    SupportObject,
    glue,
    glue_or_raise,
    measure_coverage,
)
from .uncertainty import (
    ChainUncertainty,
    IntervalBox,
    PoseUncertainty,
    independence_understatement,
    interval_propagate,
    linearization_error,
    monte_carlo_propagate,
    propagate_chain,
    propagate_first_order,
)
from .units import Handedness

__all__ = [
    # modules
    "se3",
    "frame_graph",
    "uncertainty",
    "clock_graph",
    "sheaf",
    "calibration",
    "units",
    "errors",
    # se3 / frames
    "Pose",
    "ValidityInterval",
    "Handedness",
    "Frame",
    "FrameGraph",
    "TransformEdge",
    "TransformPath",
    "PathSet",
    "DeformableTransform",
    "device_to_atlas_chain",
    "adjoint",
    "exp_se3",
    "log_se3",
    "round_trip_residual",
    # uncertainty
    "propagate_first_order",
    "independence_understatement",
    "monte_carlo_propagate",
    "interval_propagate",
    "linearization_error",
    "IntervalBox",
    "PoseUncertainty",
    "ChainUncertainty",
    "propagate_chain",
    # clocks
    "ClockGraph",
    "ClockSpec",
    "ClockMap",
    "DropSpec",
    "fit_clock_map",
    "detect_dropped_samples",
    "schedule_multirate",
    "MultirateSchedule",
    "SyncPoint",
    "InterpolationContract",
    # sheaf
    "Site",
    "SupportObject",
    "Restriction",
    "Section",
    "Cover",
    "glue",
    "glue_or_raise",
    "GlobalSection",
    "ObstructionCertificate",
    "Prolongation",
    "FineDistribution",
    "ScalePair",
    "measure_coverage",
    # calibration
    "CalibrationRecord",
    "ExpiryPolicy",
    "ValidityCheck",
    # errors
    "TransformError",
    "UnitMismatchError",
    "HandednessError",
    "FrameMismatchError",
    "NoPathError",
    "NonInvertibleTransformError",
    "CalibrationExpiredError",
    "ClockRelationUnknownError",
    "ProlongationWithoutRestrictionError",
    "CocycleObstructionError",
    "LinearizationInvalidError",
]
