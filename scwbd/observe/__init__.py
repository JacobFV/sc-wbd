"""Source-native observation operators for SC-WBD-001-beta (agent F).

Every head in this package implements ``Y_m = O_m(X_{0:t}, A_p, B_p; psi_m)
+ eps_m`` (body.tex Sec. 2.4) on its **own** spatial support and its **own**
clock, and every read returns a prediction *plus* the bias--variance ledger of
Sec. 2.7.  No modality is resampled into fictitious equivalence: an EEG head
reads a latent trajectory at 1 ms while a BOLD head reads the same trajectory
at 1 s, and neither is interpolated onto the other's clock (Sec. 2.6, Sec. 7.1).
"""

from __future__ import annotations

from .base import (
    BiasTerm,
    ObservationOperator,
    ObservationRead,
    ObservationRefusal,
    PSF,
    Prior,
    Provenance,
    RefusalR08,
    Support,
    TemporalSupport,
    UNKNOWN,
    UncertaintyLedger,
    Unresolved,
    VarianceDecomposition,
)
from .behavior import (
    BehaviorObservationOperator,
    MotorStage,
    PerceptionStage,
    PolicyStage,
    ReportingBias,
    chronometric,
    ddm_choice_probability,
    ddm_mean_decision_time,
    drift_diffusion_pdf,
    psychometric,
)
from .bold import (
    BOLDObservationOperator,
    BalloonWindkesselParameters,
    BalloonWindkesselReadout,
    CanonicalHRF,
    DriftModel,
    HRFParameters,
    MotionModel,
    PartialVolume,
    PhysiologicalNoise,
    SliceTiming,
    VoxelPSF,
    fraction_to_percent,
    percent_to_fraction,
)
from .eeg import ArtifactModel, EEGNoiseModel, EEGObservationOperator, pink_noise
from .fnirs import (
    EXTINCTION_COEFF,
    ExtracerebralModel,
    FNIRSObservationOperator,
    OpticalProperties,
    PhotonPathModel,
)
from .inverse import (
    InverseSolutionSet,
    ResolutionAnalysis,
    regularization_sweep,
    solve_inverse,
)
from .leadfield import (
    ITIS_CONDUCTIVITY,
    STANDARD_EEG_CONDUCTIVITY,
    BEMLeadField,
    ElectrodeImpedance,
    ElectrodePositionUncertainty,
    LeadField,
    ReferenceOperator,
    SphericalHeadModel,
    TissueConductivityPriors,
    meg_lead_field,
    sarvas_meg,
)

__all__ = [
    # base
    "ObservationOperator",
    "ObservationRead",
    "ObservationRefusal",
    "RefusalR08",
    "Unresolved",
    "UncertaintyLedger",
    "VarianceDecomposition",
    "BiasTerm",
    "Provenance",
    "Support",
    "TemporalSupport",
    "PSF",
    "Prior",
    "UNKNOWN",
    # lead fields
    "SphericalHeadModel",
    "LeadField",
    "BEMLeadField",
    "TissueConductivityPriors",
    "ITIS_CONDUCTIVITY",
    "STANDARD_EEG_CONDUCTIVITY",
    "ReferenceOperator",
    "ElectrodeImpedance",
    "ElectrodePositionUncertainty",
    "sarvas_meg",
    "meg_lead_field",
    # heads
    "EEGObservationOperator",
    "EEGNoiseModel",
    "ArtifactModel",
    "pink_noise",
    "BOLDObservationOperator",
    "CanonicalHRF",
    "HRFParameters",
    "BalloonWindkesselReadout",
    "BalloonWindkesselParameters",
    "SliceTiming",
    "PhysiologicalNoise",
    "MotionModel",
    "DriftModel",
    "PartialVolume",
    "VoxelPSF",
    "fraction_to_percent",
    "percent_to_fraction",
    "FNIRSObservationOperator",
    "PhotonPathModel",
    "ExtracerebralModel",
    "OpticalProperties",
    "EXTINCTION_COEFF",
    "BehaviorObservationOperator",
    "PerceptionStage",
    "PolicyStage",
    "MotorStage",
    "ReportingBias",
    "drift_diffusion_pdf",
    "ddm_choice_probability",
    "ddm_mean_decision_time",
    "psychometric",
    "chronometric",
    # inverse
    "solve_inverse",
    "InverseSolutionSet",
    "ResolutionAnalysis",
    "regularization_sweep",
]
