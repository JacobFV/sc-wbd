"""SC-WBD intervention operators: TMS, tFUS, sensory/cognitive, and ``A_safe``.

**SIMULATION ONLY.**  Per ``paper/thesis_contract.tex`` Sec. 0.6, build-order
item 6 (prospective human TMS/tFUS) is **out of scope** for SC-WBD-001-beta:
there is no ethics approval, no consent, no participants and no device.  Every
public entry point in this package operates on simulated fields and simulated
or previously-recorded open-data responses.  Nothing here drives stimulation
hardware, generates a human dosing protocol, or recommends a stimulation
parameter for a person.  :mod:`scwbd.intervene.safety` implements
:math:`\\mathcal A_{\\rm safe}` as a **refusal mechanism** -- a feasible set
that blocks optimization -- never as permission to stimulate.

The organising claim is the separation of **physical dose** from **neural
effect**.  Four quantities the thesis insists are different are four different
Python types:

=============================  ==========================================
:class:`PhysicalDose`          V/m, Pa, dB SPL, a presented sentence
:class:`TargetEngagement`      drive at a named population, produced only by
                               a *named* candidate response operator
:class:`NetworkEffect`         simulated distributed change after propagation
:class:`ClinicalUtility`       refuses to be constructed in this release
=============================  ==========================================

Layout::

    base.py       Sec. 2.4 controlled SDE; impulse limit behind a tested flag
    safety.py     A_safe, CompilerRefusal(R11), Defer, NoRecommendation
    sensory.py    sensory/cognitive/neurofeedback via declared perceptual ports
    tms/          coil -> E-field -> candidate response; pose chain; EEG artifact
    tfus/         transducer -> acoustics -> exposure -> candidate response
    limits/       declarative, citable A_safe limits (never learned)
"""

from __future__ import annotations

from .base import (
    SIMULATION_ONLY_NOTICE,
    BurstSequence,
    ClinicalUtility,
    DeviceGeometry,
    ExposureWindow,
    ImpulseLimitReport,
    InterventionOperator,
    InterventionRefusal,
    InterventionResult,
    Ledger,
    LinearFieldIntervention,
    MechanisticUncertainty,
    NetworkEffect,
    PhysicalDose,
    TargetEngagement,
    ThermalHistory,
    TissueCoupling,
    WaveformSpec,
    simulation_only_notice,
)
from .safety import (
    CompilerRefusal,
    Defer,
    FeasibleSet,
    NoRecommendation,
    ProposedIntervention,
    RiskSensitiveController,
    SafetyLimits,
    SafetyVerdict,
    SimulatedRanking,
    Violation,
)
from .sensory import (
    ContingencySpec,
    NeurofeedbackLoop,
    PerceptualPort,
    PerceptualResponseOperator,
    PortRegistry,
    SensoryContent,
    SensoryIntervention,
)

__all__ = [
    "SIMULATION_ONLY_NOTICE",
    "simulation_only_notice",
    # base / Sec. 2.4
    "InterventionOperator",
    "InterventionRefusal",
    "InterventionResult",
    "ImpulseLimitReport",
    "ExposureWindow",
    "DeviceGeometry",
    "WaveformSpec",
    "BurstSequence",
    "ThermalHistory",
    "TissueCoupling",
    "MechanisticUncertainty",
    "Ledger",
    "LinearFieldIntervention",
    # the four levels, kept apart
    "PhysicalDose",
    "TargetEngagement",
    "NetworkEffect",
    "ClinicalUtility",
    # safety / A_safe
    "CompilerRefusal",
    "SafetyLimits",
    "FeasibleSet",
    "ProposedIntervention",
    "SafetyVerdict",
    "Violation",
    "RiskSensitiveController",
    "Defer",
    "NoRecommendation",
    "SimulatedRanking",
    # sensory
    "PerceptualPort",
    "PortRegistry",
    "SensoryContent",
    "SensoryIntervention",
    "PerceptualResponseOperator",
    "ContingencySpec",
    "NeurofeedbackLoop",
]
