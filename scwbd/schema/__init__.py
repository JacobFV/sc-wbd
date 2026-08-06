"""``scwbd.schema`` - the typed contract layer of SC-WBD-001-beta.

Every type in ARCHITECTURE.md sec. 2 lives here, as a frozen pydantic v2 model
with a stable ``content_hash()``.  Other agents import from this package; the
names and signatures are the binding interface contract.
"""

from __future__ import annotations

from .base import SchemaModel, canonical_json, content_hash_of
from .claims import (
    CLAIM_STRENGTH,
    ClaimClass,
    ClaimManifest,
    ClaimOverride,
    PosteriorClass,
    is_demotion,
)
from .clocks import (
    AllOf,
    AnyOf,
    BoundaryTrigger,
    ClockEdge,
    ClockSpec,
    EventTrigger,
    NotTrigger,
    PeriodicTrigger,
    ScheduleContext,
    SyncEvidence,
    UNVERIFIED_SYNC,
    UpdateTrigger,
)
from .frames import (
    CalibrationManifest,
    FrameEdge,
    FrameGraphSpec,
    FrameNode,
    Handedness,
    ManifestLayer,
    TransformKind,
)
from .ids import ClockId, FrameId, IdError, ScaleId
from .ledger import VARIANCE_COMPONENTS, BiasStatus, UncertaintyLedger
from .lineage import (
    Identity,
    LineageError,
    LineageGraph,
    LineageUnit,
    UnitKind,
)
from .operators import (
    CAUSAL_BASES,
    CAUSAL_STATUSES,
    EvidenceClass,
    Identification,
    IdentificationBasis,
    MechanisticStatus,
    OperatorFamily,
    OperatorSpec,
    ResidualPolicy,
    SemigroupTest,
)
from .poset import (
    CocycleCheck,
    GluingPolicy,
    MapSpec,
    ObstructionCertificate,
    ResolutionPoset,
    ScaleMapPair,
    ScaleNode,
)
from .priors import (
    BetaPrior,
    DiracPrior,
    GammaPrior,
    LogNormalPrior,
    NormalPrior,
    Prior,
    PriorBase,
    UniformPrior,
    as_prior,
)
from .refusals import (
    REFUSAL_CODES,
    REFUSALS,
    CompilerRefusal,
    RefusalCode,
    RefusalRecord,
    RefusalSpec,
    remedy_for,
)
from .regions import AtlasRef, Authority, Region
from .schema import SCHEMA_VERSION, BrainSchema
from .sources import (
    GENERATIVE_ROLES,
    Governance,
    GradientPermission,
    HierarchicalEffect,
    InterventionModel,
    LikelihoodKind,
    Missingness,
    ObservationModel,
    PopulationStructure,
    SafeSet,
    SignalSpec,
    SourceCard,
    SourceRole,
    SplitPolicy,
    TemporalHoldout,
)
from .state import (
    DTYPE_BYTES,
    ComponentKind,
    ComponentSpec,
    Port,
    PortDirection,
    PortRole,
    StateSpec,
)
from .supports import PSF, Support, SupportKind, TemporalSupport
from .units import (
    ARBITRARY_UNITS,
    BASE_DIMENSIONS,
    DIMENSIONLESS,
    Dimension,
    Unit,
    UnitError,
    known_units,
    register_unit,
)

__all__ = [
    # base
    "SchemaModel",
    "canonical_json",
    "content_hash_of",
    # units and ids
    "Unit",
    "UnitError",
    "Dimension",
    "BASE_DIMENSIONS",
    "DIMENSIONLESS",
    "ARBITRARY_UNITS",
    "register_unit",
    "known_units",
    "FrameId",
    "ClockId",
    "ScaleId",
    "IdError",
    # priors
    "Prior",
    "PriorBase",
    "NormalPrior",
    "LogNormalPrior",
    "UniformPrior",
    "BetaPrior",
    "GammaPrior",
    "DiracPrior",
    "as_prior",
    # supports and ledger
    "PSF",
    "Support",
    "SupportKind",
    "TemporalSupport",
    "UncertaintyLedger",
    "BiasStatus",
    "VARIANCE_COMPONENTS",
    # state
    "ComponentSpec",
    "ComponentKind",
    "StateSpec",
    "Port",
    "PortDirection",
    "PortRole",
    "DTYPE_BYTES",
    # regions and operators
    "Region",
    "AtlasRef",
    "Authority",
    "OperatorSpec",
    "OperatorFamily",
    "EvidenceClass",
    "MechanisticStatus",
    "Identification",
    "IdentificationBasis",
    "ResidualPolicy",
    "SemigroupTest",
    "CAUSAL_STATUSES",
    "CAUSAL_BASES",
    # clocks
    "ClockSpec",
    "ClockEdge",
    "ScheduleContext",
    "UpdateTrigger",
    "PeriodicTrigger",
    "EventTrigger",
    "BoundaryTrigger",
    "AllOf",
    "AnyOf",
    "NotTrigger",
    "SyncEvidence",
    "UNVERIFIED_SYNC",
    # poset
    "ResolutionPoset",
    "ScaleNode",
    "ScaleMapPair",
    "MapSpec",
    "CocycleCheck",
    "GluingPolicy",
    "ObstructionCertificate",
    # frames
    "FrameGraphSpec",
    "FrameNode",
    "FrameEdge",
    "CalibrationManifest",
    "Handedness",
    "TransformKind",
    "ManifestLayer",
    # lineage
    "Identity",
    "LineageUnit",
    "LineageGraph",
    "LineageError",
    "UnitKind",
    # sources
    "SourceCard",
    "SourceRole",
    "Governance",
    "PopulationStructure",
    "HierarchicalEffect",
    "SignalSpec",
    "ObservationModel",
    "LikelihoodKind",
    "InterventionModel",
    "SafeSet",
    "Missingness",
    "GradientPermission",
    "SplitPolicy",
    "TemporalHoldout",
    "GENERATIVE_ROLES",
    # schema and claims
    "BrainSchema",
    "SCHEMA_VERSION",
    "ClaimManifest",
    "ClaimOverride",
    "ClaimClass",
    "PosteriorClass",
    "CLAIM_STRENGTH",
    "is_demotion",
    # refusals
    "CompilerRefusal",
    "RefusalRecord",
    "RefusalSpec",
    "RefusalCode",
    "REFUSALS",
    "REFUSAL_CODES",
    "remedy_for",
]
