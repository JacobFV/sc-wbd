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
(``thesis_contract.tex`` Sec. 0.6, build-order item 6) -- as a matter of
*capability*, not of anyone's paperwork.  Governance is gated, not assumed:
:class:`~scwbd.schema.authorization.AuthorizationRecord` records a declared
approval and a validated one changes ``ModelProvenance.claim_scope``.  Even
then a targeting claim is refused, because this release serves
``weights_status="analytic_backend"`` -- there is no trained checkpoint behind
any prediction here.  See ``reports/governance_authorization.md``.
"""

from __future__ import annotations

from ._compat import SIMULATION_ONLY_NOTICE, Unresolved
from .admission import (
    CONSUMER_STANDING_INVARIANTS,
    EARLIEST_CREDIBLE_REVIEW,
    EXPORT_PURPOSES,
    LIVE_PURPOSES,
    AdmissionCondition,
    AdmissionVerdict,
    AuthorizationInvalid,
    CheckpointClaims,
    CheckpointRefused,
    ConsumerInvariants,
    ConsumerInvariantViolation,
    ExportPurpose,
    LiveUseAuthorization,
    admit,
)
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
from .ports import (
    DeclaredPort,
    LayoutNotDeclared,
    PortContract,
    PortedState,
    PortError,
    PortValue,
    RawStateAccessRefused,
    SpanViolation,
    UndeclaredPort,
    UnexportedPort,
)
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
    # the export gate (Sec. 6): what may leave here, for what purpose
    "admit",
    "AdmissionCondition",
    "AdmissionVerdict",
    "CheckpointClaims",
    "CheckpointRefused",
    "ConsumerInvariants",
    "ConsumerInvariantViolation",
    "AuthorizationInvalid",
    "LiveUseAuthorization",
    "ExportPurpose",
    "EXPORT_PURPOSES",
    "LIVE_PURPOSES",
    "EARLIEST_CREDIBLE_REVIEW",
    "CONSUMER_STANDING_INVARIANTS",
    # declared ports: the only way to read model state
    "PortContract",
    "DeclaredPort",
    "PortValue",
    "PortedState",
    "PortError",
    "UndeclaredPort",
    "UnexportedPort",
    "SpanViolation",
    "RawStateAccessRefused",
    "LayoutNotDeclared",
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
