"""Interventional (perturbation) sources and their control graphs.

The corpus gap this package exists to close: 35 of 37 simulated corpus shards
carry ``control_graph: none``, so G4 ("perturbation reduces non-identifiability")
is *unexercisable* rather than merely unrun.  No amount of additional
observational or synthetic data changes that; only genuine interventional
records do.

What is here is one such record set — the ds004024 spTMS runs — together with
the :class:`~scwbd.sources.perturbation.control_graph.ControlGraph` that says
what was manipulated and, just as importantly, what was not recorded.  Read
``reports/perturbation_viability.md`` before consuming any of it: the available
N is 2 participants, and three of the five quantities G4's ``prospective_recovery``
enumerates are not supported by this snapshot.
"""

from .control_graph import (
    G4_RECOVERY_QUANTITIES,
    ControlGraph,
    ControlGraphError,
    ControlVariable,
    ExcludedWindow,
    ExposureInterval,
    Provenance,
)
from .ds004024 import (
    DatasetNotAvailable,
    TMSEpochs,
    available_subjects,
    derive_stimulated_hemisphere,
    load_run,
    measure_saturation,
    records,
)

__all__ = [
    "ControlGraph",
    "ControlGraphError",
    "ControlVariable",
    "DatasetNotAvailable",
    "ExcludedWindow",
    "ExposureInterval",
    "G4_RECOVERY_QUANTITIES",
    "Provenance",
    "TMSEpochs",
    "available_subjects",
    "derive_stimulated_hemisphere",
    "load_run",
    "measure_saturation",
    "records",
]
