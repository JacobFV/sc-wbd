"""``scwbd.infer`` -- inference and the linear identifiability laboratory.

This package owns build-order item 2 of ``thesis_contract.tex`` sec. 0.6 and the
first claim gate of sec. 11: the linear--Gaussian native-clock benchmark whose
Fisher-information and recovery results decide which coupling, delay,
observation and calibration parameters the next model is allowed to claim.

Modules
-------
``linear_gaussian``  equations T1--T3 as an exact multirate state-space model
``filters``          Kalman/RTS, EKF, UKF, EnKF, particle/SMC -- native clocks
``fisher``           T4 expected Fisher information, prior reported separately
``identifiability``  the five-design benchmark, sweeps, recovery, profiles
``variational``      body.tex eq. (4) with the R09 pseudo-loss guard in code
``sbi``              amortized neural posterior estimation + SBC / coverage
``model_comparison`` evidence, WAIC, PSIS-LOO, and posterior *branching*
``calibration``      coverage, PIT, reliability, sharpness, scoring, subgroups
``synthetic_slice``  the second sec. 0.3 artifact: end-to-end recovery
``report``           preregistered manifest + machine-readable claim report
``adapters``         stable bindings for ``scwbd.bench`` (G4) and ``scwbd.schema``
"""

from __future__ import annotations

from .types import (
    CalibrationClaimError,
    CoverageResult,
    Interval,
    PosteriorSummary,
    UnresolvedCausalAmbiguity,
    seed_everything,
)

__all__ = [
    "CalibrationClaimError",
    "CoverageResult",
    "Interval",
    "PosteriorSummary",
    "UnresolvedCausalAmbiguity",
    "expected_fisher",
    "kalman_filter",
    "run_benchmark",
    "run_synthetic_slice",
    "seed_everything",
]


def __getattr__(name: str):  # lazy: keeps `import scwbd.infer` cheap
    if name == "expected_fisher":
        from .fisher import expected_fisher

        return expected_fisher
    if name == "kalman_filter":
        from .filters import kalman_filter

        return kalman_filter
    if name == "run_benchmark":
        from .identifiability import run_benchmark

        return run_benchmark
    if name == "run_synthetic_slice":
        from .synthetic_slice import run_synthetic_slice

        return run_synthetic_slice
    raise AttributeError(name)
