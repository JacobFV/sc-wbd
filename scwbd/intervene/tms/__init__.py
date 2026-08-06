"""TMS stack: coil -> induced E-field -> candidate response, plus pose and artifact.

**SIMULATION ONLY** -- see :mod:`scwbd.intervene`.

Each stage is a separate module and separately validated (thesis Sec. 7.2:
"pose accuracy, field accuracy, target engagement, network change, symptom
change, and comparative clinical utility are separate validation levels"):

``coil``      geometry, winding discretisation, waveform, dI/dt
``efield``    analytic spherical reference (Sarvas / Heller--van Hulsteyn)
              plus a convergence-tested charge BEM, and a SimNIBS wrapper
``pose``      the full SE(3) device-to-atlas chain and its covariance;
              refuses an underspecified target such as "5 cm anterior"
``response``  plural candidate E-field -> population operators under
              posterior model comparison
``artifact``  TMS--EEG amplifier saturation and peripheral co-stimulation,
              separable from any candidate cortical response
"""

from __future__ import annotations

from .artifact import AmplifierSpec, ArtifactComponents, EvokedTemplate, TMSEEGArtifactModel
from .coil import (
    MU0,
    CircularCoil,
    CoilGeometry,
    FigureEightCoil,
    TMSPulse,
    biphasic,
    halfsine,
    monophasic,
    pulse_waveform_spec,
)
from .efield import (
    ImpossibleGeometry,
    charge_bem_induced_efield,
    assert_sources_exterior,
    ChargeBEM,
    LayeredSphereBEM,
    SphericalHeadModel,
    TriMesh,
    analytic_sphere_efield,
    coil_dipoles_in_head_frame,
    efield_from_coil,
    icosphere,
    primary_efield_dipoles,
    primary_efield_segments,
    simnibs_available,
    simnibs_status,
    triangle_field_integral,
    uniform_dbdt_efield,
)
from .pose import (
    CoilPose,
    FieldUncertainty,
    FrameTransform,
    PoseChain,
    PoseSensitivityReport,
    UnderspecifiedPose,
    coil_pose_on_sphere,
    compose_chain,
    pose_field_sensitivity,
    propagate_pose_uncertainty,
    require_pose,
    se3_adjoint,
    standard_fiducials,
    se3_exp,
    se3_log,
)
from .response import (
    ActivatingFunctionResponse,
    CorticalFrame,
    DirectionalTuningResponse,
    MagnitudeThresholdResponse,
    ModelComparison,
    NormalComponentResponse,
    PopulationState,
    ResponseModelSet,
    TangentialMagnitudeResponse,
    TMSResponseOperator,
    default_candidate_set,
    local_cortical_frame,
)

__all__ = [
    "MU0",
    "CoilGeometry",
    "CircularCoil",
    "FigureEightCoil",
    "TMSPulse",
    "monophasic",
    "biphasic",
    "halfsine",
    "pulse_waveform_spec",
    "ImpossibleGeometry",
    "charge_bem_induced_efield",
    "assert_sources_exterior",
    "analytic_sphere_efield",
    "primary_efield_dipoles",
    "primary_efield_segments",
    "uniform_dbdt_efield",
    "triangle_field_integral",
    "SphericalHeadModel",
    "TriMesh",
    "icosphere",
    "ChargeBEM",
    "LayeredSphereBEM",
    "coil_dipoles_in_head_frame",
    "efield_from_coil",
    "simnibs_available",
    "simnibs_status",
    "UnderspecifiedPose",
    "FrameTransform",
    "PoseChain",
    "CoilPose",
    "require_pose",
    "compose_chain",
    "coil_pose_on_sphere",
    "standard_fiducials",
    "se3_exp",
    "se3_log",
    "se3_adjoint",
    "PoseSensitivityReport",
    "pose_field_sensitivity",
    "FieldUncertainty",
    "propagate_pose_uncertainty",
    "CorticalFrame",
    "local_cortical_frame",
    "PopulationState",
    "TMSResponseOperator",
    "NormalComponentResponse",
    "TangentialMagnitudeResponse",
    "MagnitudeThresholdResponse",
    "DirectionalTuningResponse",
    "ActivatingFunctionResponse",
    "default_candidate_set",
    "ResponseModelSet",
    "ModelComparison",
    "AmplifierSpec",
    "EvokedTemplate",
    "ArtifactComponents",
    "TMSEEGArtifactModel",
]
