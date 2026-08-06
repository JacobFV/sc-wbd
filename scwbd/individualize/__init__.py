"""``scwbd.individualize`` -- single-modality patient individualization.

*"Sometimes a patient only has MRI data or only has EEG data and we just have to
work with that."*  This package makes that a first-class case rather than a
degraded one, and makes the degradation legible.

The design follows from a measurement, not from a preference.  Agent Fisher's
committed benchmark (``reports/identifiability/results.json``, regenerated and
confirmed here) reports the minimum eigenvalue of the theta-profile
**likelihood-only** expected Fisher information for
``theta = (a21, a32, a13, tau)``:

============================  ===========  ==========================  ============
regime                        EEG only     joint EEG+fMRI              fMRI only
============================  ===========  ==========================  ============
``reference``                 16.008456    16.008480                   2.9294e-06
``weak_coupling_long_delay``  1.839545     1.839545                    4.6778e-07
``low_snr_short_delay``       13.736239    13.737417                   7.6770e-07
============================  ===========  ==========================  ============

EEG alone loses 0.00015% of the joint information; fMRI alone carries between
5.5e+06 and 3.9e+06 times less.  A single "can this patient be individualized?"
flag would have to answer for both.  So individualization is **per parameter
group**:

``availability``  what the patient has, declared; missing is missing (rule 1)
``groups``        the granularity at which individualization is decided
``profile``       which groups are identifiable, computed BEFORE fitting
``hierarchy``     ``theta_{p,s} = mu + alpha + delta + zeta``, R07-enforced
``fit``           fit only the admitted groups; everything else labelled
                  ``population_prior`` and returned bit-identical to the prior
``query``         a query needing an unidentifiable group returns ``Defer``
``report``        the per-patient report a clinician reads
"""

from __future__ import annotations

from .availability import (
    MODALITIES,
    MissingModalityError,
    Modality,
    ModalityAvailability,
    ModalityRecord,
    UndeclaredModalityError,
    ZeroImputationRefused,
    refuse_zero_imputation,
)
from .fit import (
    CONSISTENCY_REL_TOLERANCE,
    INDIVIDUALIZED,
    POPULATION_PRIOR,
    WEAKLY_INDIVIDUALIZED,
    ConsistencyCheck,
    GroupOutcome,
    IndividualizationResult,
    PatientData,
    data_consistency,
    individualize,
    simulate_patient,
)
from .groups import GROUPS, LIKELIHOOD_GROUPS, ParameterGroup, group_by_name
from .hierarchy import (
    Decomposition,
    PopulationModel,
    R07Violation,
    decompose_sessions,
    hierarchical_effect_declarations,
    recover_decomposition,
)
from .profile import (
    GroupIdentifiability,
    IdentifiabilityProfile,
    IdentifiabilityThresholds,
    InadequateDelayLine,
    assert_delay_line_adequate,
    benchmark_config,
    profile_identifiability,
    profiled_information,
)
from .query import Defer, Query, QueryAnswer, Unresolved, answer, coupling_gain_query
from .report import patient_report, profile_report, result_json

__all__ = [
    "CONSISTENCY_REL_TOLERANCE",
    "GROUPS",
    "INDIVIDUALIZED",
    "LIKELIHOOD_GROUPS",
    "MODALITIES",
    "POPULATION_PRIOR",
    "WEAKLY_INDIVIDUALIZED",
    "ConsistencyCheck",
    "Decomposition",
    "Defer",
    "InadequateDelayLine",
    "assert_delay_line_adequate",
    "data_consistency",
    "GroupIdentifiability",
    "GroupOutcome",
    "IdentifiabilityProfile",
    "IdentifiabilityThresholds",
    "IndividualizationResult",
    "MissingModalityError",
    "Modality",
    "ModalityAvailability",
    "ModalityRecord",
    "ParameterGroup",
    "PatientData",
    "PopulationModel",
    "Query",
    "QueryAnswer",
    "R07Violation",
    "UndeclaredModalityError",
    "Unresolved",
    "ZeroImputationRefused",
    "answer",
    "benchmark_config",
    "coupling_gain_query",
    "decompose_sessions",
    "group_by_name",
    "hierarchical_effect_declarations",
    "individualize",
    "patient_report",
    "profile_identifiability",
    "profile_report",
    "profiled_information",
    "recover_decomposition",
    "refuse_zero_imputation",
    "result_json",
    "simulate_patient",
]
