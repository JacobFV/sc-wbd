"""tFUS stack: transducer -> skull acoustics -> exposure -> candidate response.

**SIMULATION ONLY** -- see :mod:`scwbd.intervene`.

Separated exactly as thesis Sec. 7.2 requires -- tracked transducer pose,
skull acoustics, steering commands, in-situ pressure and thermal estimates,
uncertain cellular coupling, and downstream network dynamics are different
objects:

``transducer``  element geometry, phasing, steering, pulse sequence
``acoustics``   angular-spectrum / Rayleigh propagation validated against the
                closed-form piston field; split-step skull screens; CT
                Hounsfield conversion; a k-Wave wrapper that refuses rather
                than degrading silently
``exposure``    planned focus vs realized exposure as distinct random
                variables; MI, I_SPPA, I_SPTA, temperature, CEM43
``response``    plural candidate tissue-coupling operators, including an
                auditory-confound null, under posterior model comparison
"""

from __future__ import annotations

from .acoustics import (
    BRAIN,
    CORTICAL_BONE,
    WATER,
    AcousticMedium,
    KWaveSolver,
    SkullLayer,
    SplitStepPropagator,
    angular_spectrum_propagate,
    hounsfield_to_acoustic,
    kwave_available,
    kwave_status,
    on_axis_piston_pressure,
    pressure_dose,
    rayleigh_sommerfeld_pressure,
)
from .exposure import (
    ExposureMetrics,
    FocalDivergence,
    PlannedFocus,
    RealizedExposure,
    accumulate_thermal_dose,
    bioheat_temperature,
    exposure_metrics,
    focal_divergence,
)
from .response import (
    AuditoryConfoundNullResponse,
    IntramembraneCavitationResponse,
    MechanosensitiveChannelResponse,
    RadiationForceResponse,
    TFUSResponseModelSet,
    TFUSResponseOperator,
    ThermalResponse,
    TissueContext,
    default_tfus_candidate_set,
)
from .transducer import (
    AnnularArray,
    PlanarGridArray,
    PulseSequence,
    SingleElementBowl,
    TransducerArray,
)

__all__ = [
    "TransducerArray",
    "SingleElementBowl",
    "AnnularArray",
    "PlanarGridArray",
    "PulseSequence",
    "AcousticMedium",
    "WATER",
    "BRAIN",
    "CORTICAL_BONE",
    "on_axis_piston_pressure",
    "angular_spectrum_propagate",
    "rayleigh_sommerfeld_pressure",
    "SkullLayer",
    "SplitStepPropagator",
    "hounsfield_to_acoustic",
    "pressure_dose",
    "kwave_available",
    "kwave_status",
    "KWaveSolver",
    "PlannedFocus",
    "RealizedExposure",
    "FocalDivergence",
    "focal_divergence",
    "ExposureMetrics",
    "exposure_metrics",
    "bioheat_temperature",
    "accumulate_thermal_dose",
    "TissueContext",
    "TFUSResponseOperator",
    "IntramembraneCavitationResponse",
    "MechanosensitiveChannelResponse",
    "RadiationForceResponse",
    "ThermalResponse",
    "AuditoryConfoundNullResponse",
    "default_tfus_candidate_set",
    "TFUSResponseModelSet",
]
