"""``scwbd.runtime`` -- the serving surface, and the boundary of what SC-WBD does.

Two runtimes live here and they are not interchangeable:

:class:`~scwbd.runtime.targeting.TargetingService` (``ARCHITECTURE.md`` Sec. 6)
    The *only* thing ``~/Documents/robotics`` (``tms-robotics``) consumes.
    Given a head model and a candidate coil pose it returns the induced
    E-field, a target-engagement **distribution**, a network response with
    model-class disagreement, a full uncertainty ledger, and a decision that is
    one of ``Recommend | Defer | Refuse``.

:class:`~scwbd.runtime.brain_runtime.BrainRuntime` (Sec. 6b)
    A generic research runtime: typed sensory ports in, multirate state
    advance, typed readouts with ledgers.  Not what ``tms-robotics`` consumes.

What this package does **not** contain, by construction
-------------------------------------------------------
No joint commands.  No trajectories.  No actuation.  No stimulation authority.
No human dosing protocol.  No import path from here reaches a device, a
transport, or a controller.  SC-WBD sits strictly upstream of the consumer's
"registered external scalp target" node, and ``tests/runtime/`` contains an
API-surface test that fails if a command-authority symbol ever appears.

Prospective human TMS/tFUS is out of scope for SC-WBD-001-beta
(``thesis_contract.tex`` Sec. 0.6, build-order item 6): no ethics approval, no
consent, no participants, no device.
"""

from __future__ import annotations

from ._compat import SIMULATION_ONLY_NOTICE, Unresolved
from .backends import (
    AnalyticSphericalEField,
    ChargeBEMEField,
    CoilSpec,
    DEFAULT_PROPAGATORS,
    DEFAULT_RESPONSE_OPERATORS,
    FieldResolutionUnresolved,
    FieldSolve,
    GatedAnalyticSphereEField,
    ImpossiblePlacement,
)
from .brain_runtime import BrainRuntime, PortSpec, Readout, RuntimeStep
from .compare import (
    CandidateRanking,
    PreregisteredCandidate,
    PreregisteredPoseSet,
    RankedCandidate,
    UnresolvedCausalAmbiguity,
    compare_poses,
)
from .frames import DeclaredEdge, FrameChain, ResolvedChain
from .head import HeadModel, spherical_phantom
from .provenance import (
    MODEL_DESIGNATION,
    RUNTIME_API_VERSION,
    SCHEMA_VERSION,
    ModelProvenance,
    ProvenanceExpectation,
    ProvenanceMismatch,
)
from .serving import (
    CheckpointNotFound,
    CheckpointRecord,
    ServedModel,
    coil_pose_over_region,
    discover_checkpoint,
)
from .targeting import SessionProtocol, TargetingConfig, TargetingService
from .types import (
    Decision,
    Defer,
    EFieldPrediction,
    EngagementDistribution,
    FieldAccuracy,
    LedgerIncomplete,
    NetworkResponse,
    PoseEvaluation,
    PoseRequest,
    Recommend,
    Refuse,
    UnderspecifiedPose,
    UndeclaredTransform,
    UtilityStatus,
)

__all__ = [
    # notice
    "SIMULATION_ONLY_NOTICE",
    # the tms-robotics path
    "TargetingService",
    "TargetingConfig",
    "SessionProtocol",
    "PoseEvaluation",
    "PoseRequest",
    "EFieldPrediction",
    "FieldAccuracy",
    "EngagementDistribution",
    "NetworkResponse",
    "UtilityStatus",
    "Decision",
    "Recommend",
    "Defer",
    "Refuse",
    "Unresolved",
    "UnderspecifiedPose",
    "UndeclaredTransform",
    "LedgerIncomplete",
    # frames
    "FrameChain",
    "DeclaredEdge",
    "ResolvedChain",
    # head + backends
    "HeadModel",
    "spherical_phantom",
    "CoilSpec",
    "AnalyticSphericalEField",
    "GatedAnalyticSphereEField",
    "ChargeBEMEField",
    "FieldSolve",
    "FieldResolutionUnresolved",
    "ImpossiblePlacement",
    "DEFAULT_RESPONSE_OPERATORS",
    "DEFAULT_PROPAGATORS",
    # offline comparison
    "compare_poses",
    "PreregisteredPoseSet",
    "PreregisteredCandidate",
    "CandidateRanking",
    "RankedCandidate",
    "UnresolvedCausalAmbiguity",
    # serving + provenance
    "ServedModel",
    "CheckpointRecord",
    "CheckpointNotFound",
    "discover_checkpoint",
    "coil_pose_over_region",
    "ModelProvenance",
    "ProvenanceExpectation",
    "ProvenanceMismatch",
    "MODEL_DESIGNATION",
    "SCHEMA_VERSION",
    "RUNTIME_API_VERSION",
    # generic research runtime (Sec. 6b)
    "BrainRuntime",
    "PortSpec",
    "Readout",
    "RuntimeStep",
]
