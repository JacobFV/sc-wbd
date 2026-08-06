"""SC-WBD intervention operators: TMS, tFUS, sensory/cognitive, and ``A_safe``.

**SIMULATION ONLY.**  Every public entry point in this package operates on
simulated fields and simulated or previously-recorded open-data responses.
Nothing here drives stimulation hardware, generates a human dosing protocol,
or recommends a stimulation parameter for a person -- and that is enforced,
not asserted: there is no device command surface to reach.

This docstring used to add "there is no ethics approval, no consent, no
participants and no device".  That was false and the code never checked it;
:mod:`scwbd.schema.authorization` gates prospective human work on a validated
declaration and admits a complete, in-date, in-scope one.  Applying a plan to
a person or to hardware is gated by :mod:`scwbd.intervene.deployment`, which
refuses until a record exists that the preliminary review *occurred* with an
approving outcome -- a scheduled review is not a completed one.
:mod:`scwbd.intervene.safety` implements
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
    numerics.py   independent FD/FDTD solvers for the Sec. 11.1 N3/N4 gates
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
from .deployment import (
    PRELIMINARY_REVIEW_SCHEDULED,
    ApplicationMode,
    LiveApplicationVerdict,
    PreliminaryReviewRecord,
    ReviewFailure,
    authorize_live_application,
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
    # live-application gate
    "PRELIMINARY_REVIEW_SCHEDULED",
    "ApplicationMode",
    "LiveApplicationVerdict",
    "PreliminaryReviewRecord",
    "ReviewFailure",
    "authorize_live_application",
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
