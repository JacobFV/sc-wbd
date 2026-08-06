"""``scwbd.dynamics`` — the whole-brain dynamical core (agent E).

A batched, GPU-first, differentiable simulator whose mechanistic backends are
**interchangeable and comparable, not assumed**:

* :mod:`~scwbd.dynamics.backends` — Wilson–Cowan, Jansen–Rit, reduced Wong–Wang
  (with FIC), Stuart–Landau, Kuramoto, the Linear–Gaussian reference, and the
  equal-capacity ``LearnedNeuralOperator`` control;
* :mod:`~scwbd.dynamics.coupling` — delayed block-sparse connectome coupling and
  the ``F_local + F_long + R_theta`` factorization of §4.1;
* :mod:`~scwbd.dynamics.residual` — the learned residual with the R05
  mechanism-dominance guard;
* :mod:`~scwbd.dynamics.integrators` — Euler–Maruyama / Heun / Milstein / SRK,
  adaptive stepping, and the semigroup residual with the R06 guard;
* :mod:`~scwbd.dynamics.scheduler` — multirate co-simulation with forced
  synchronisation and a reported coarsening budget;
* :mod:`~scwbd.dynamics.hemodynamics` — Balloon–Windkessel as a slow field;
* :mod:`~scwbd.dynamics.plasticity` — ``theta = (state, gain, synapse, structure)``;
* :mod:`~scwbd.dynamics.hippocampus` — interchangeable memory backends + the
  discriminating benchmark;
* :mod:`~scwbd.dynamics.subcortical` — thalamus, basal ganglia, cerebellum and
  receptor-typed neuromodulatory fields.

Every stochastic entry point takes an explicit ``seed``; solvers stay fp32.

**Consuming the anatomy prior.**  The adapter onto :class:`scwbd.anatomy.BrainPrior`
is a pair of methods on every backend rather than a free function, so it is easy
to miss when grepping:

* ``Backend.from_prior(brain_prior)`` — bind a backend to the prior;
* ``backend.theta_from_prior(brain_prior, batch, seed=...)`` — draw a batch of
  parameter sets carrying the prior's regional structure.

The result's ``theta.provenance`` records, per parameter, whether the prior was
``applied`` and — when it was not — *why*.  Read it: a backend can legitimately
have no parameter the prior maps onto (Stuart-Landau and Kuramoto have no
relaxation timescale), and in that case its regional values are backend defaults,
not anatomy.  Helper functions: :func:`resolve_prior_field`,
:func:`sample_prior_list`, :func:`map_fragility`.
"""

from .backends import (
    DynamicsBackend,
    JansenRit,
    Kuramoto,
    LearnedNeuralOperator,
    LinearGaussian,
    ReducedWongWang,
    ReducedWongWangSingle,
    StuartLandau,
    WilsonCowan,
    assert_equal_capacity,
    get_backend,
    list_backends,
    map_fragility,
    match_capacity,
    resolve_prior_field,
    sample_prior_list,
    tune_fic,
)
from .coupling import (
    DelayBuffer,
    DelayedConnectome,
    EdgePenalty,
    EdgeSet,
    HybridField,
    LocalField,
    deterministic_scatter,
    distance_matched_control,
    randomized_control,
)
from .hemodynamics import BalloonWindkessel, bold_field_policy, sample_hemodynamic_params
from .hippocampus import (
    HippocampalBackend,
    ModernHopfield,
    SparseDistributedMemory,
    SuccessorRepresentation,
    VectorHaSH,
    compare_backends,
    get_hippocampal_backend,
    signature,
)
from .integrators import (
    AdaptiveStepper,
    BrownianPath,
    SemigroupCertificate,
    SemigroupGuard,
    certify_semigroup,
    euler_maruyama,
    get_integrator,
    heun,
    integrate,
    milstein,
    semigroup_residual,
    stochastic_rk,
)
from .plasticity import (
    GainController,
    StructuralRewriter,
    SynapticPlasticity,
    theta_field_policies,
)
from .residual import DominanceReport, MechanismDominanceGuard, ResidualOperator, ValiditySet
from .scheduler import (
    FieldPolicy,
    MultirateScheduler,
    SyncTrigger,
    prediction_error_trigger,
    sustained_activity_trigger,
)
from .family_backends import (
    BasalGangliaBackend,
    CerebellarForwardBackend,
    HippocampalCodeBackend,
    ThalamicRelayBackend,
)
from .simulator import SimConfig, SimResult, WholeBrainSimulator, fc_correlation, functional_connectivity
from .subcortical import (
    BasalGangliaGate,
    Cerebellum,
    NeuromodulatorBank,
    NeuromodulatoryField,
    ReceptorSpec,
    ThalamicRelay,
)
from .types import (
    MechanismRefusal,
    NumericalBudget,
    ParamPack,
    Prior,
    SemanticCollapseError,
    SemigroupRefusal,
)

__all__ = [
    # backends
    "resolve_prior_field", "sample_prior_list", "map_fragility",
    "DynamicsBackend", "WilsonCowan", "JansenRit", "ReducedWongWang", "ReducedWongWangSingle",
    "StuartLandau", "Kuramoto", "LinearGaussian", "LearnedNeuralOperator",
    "get_backend", "list_backends", "match_capacity", "assert_equal_capacity", "tune_fic",
    # coupling
    "EdgeSet", "DelayBuffer", "DelayedConnectome", "LocalField", "HybridField", "EdgePenalty",
    "randomized_control", "distance_matched_control", "deterministic_scatter",
    # residual / guards
    "ResidualOperator", "MechanismDominanceGuard", "DominanceReport", "ValiditySet",
    # integrators
    "euler_maruyama", "heun", "milstein", "stochastic_rk", "get_integrator", "integrate",
    "BrownianPath", "AdaptiveStepper", "semigroup_residual", "certify_semigroup",
    "SemigroupCertificate", "SemigroupGuard",
    # scheduler
    "FieldPolicy", "MultirateScheduler", "SyncTrigger",
    "sustained_activity_trigger", "prediction_error_trigger",
    # hemodynamics
    "BalloonWindkessel", "sample_hemodynamic_params", "bold_field_policy",
    # plasticity
    "GainController", "SynapticPlasticity", "StructuralRewriter", "theta_field_policies",
    # hippocampus
    "HippocampalBackend", "ModernHopfield", "VectorHaSH", "SparseDistributedMemory",
    "SuccessorRepresentation", "get_hippocampal_backend", "signature", "compare_backends",
    # subcortical
    "ThalamicRelay", "BasalGangliaGate", "Cerebellum", "ReceptorSpec",
    "NeuromodulatoryField", "NeuromodulatorBank",
    # per-family engineered backends (§5) -- these are what makes the subcortical,
    # cerebellar and hippocampal modules reachable from the foundation model
    "ThalamicRelayBackend", "BasalGangliaBackend", "HippocampalCodeBackend",
    "CerebellarForwardBackend",
    # simulator
    "WholeBrainSimulator", "SimConfig", "SimResult", "functional_connectivity", "fc_correlation",
    # types
    "Prior", "ParamPack", "NumericalBudget", "MechanismRefusal", "SemigroupRefusal",
    "SemanticCollapseError",
]
