"""Shared types, numerical policy, and claim-guard objects for ``scwbd.infer``.

Numerical policy (ARCHITECTURE.md sec. 3)
-----------------------------------------
* Fisher information, covariance propagation and filter recursions run in
  ``float64`` by default.  ``float32`` is the documented minimum; ``bfloat16``
  is never permitted here.
* Every stochastic entry point takes an explicit ``seed``.  Determinism is a
  test (``tests/infer/test_determinism.py``), not an aspiration.

Claim guards
------------
* :class:`PosteriorKind` distinguishes a genuine Bayesian posterior (built only
  from normalized, evaluable probability models) from a *generalized* /
  pseudo-posterior obtained from auxiliary penalties.  Refusal **R09** of
  ``thesis_contract.tex`` Table ``tab:compiler-refusals`` forbids reporting the
  second as if it were the first; :class:`CalibrationClaimError` is the runtime
  error that enforces it.
* :class:`UnresolvedCausalAmbiguity` is the *return value* demanded by
  ``body.tex`` sec. 7.1 when two models remain observationally equivalent but
  imply different interventions.  It is never an average.
"""

from __future__ import annotations

import dataclasses
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
import torch

__all__ = [
    "DTYPE",
    "CalibrationClaimError",
    "CoverageResult",
    "Interval",
    "PosteriorKind",
    "PosteriorSummary",
    "UnresolvedCausalAmbiguity",
    "default_device",
    "as_builtin",
    "cap_gpu_memory",
    "gpu_memory_report",
    "resolve_device",
    "seed_everything",
    "torch_dtype",
]

DTYPE = torch.float64
_ALLOWED_DTYPES = (torch.float32, torch.float64)


def torch_dtype(name: str | torch.dtype | None) -> torch.dtype:
    """Resolve a dtype, refusing anything below the ``float32`` floor."""
    if name is None:
        return DTYPE
    dt = name if isinstance(name, torch.dtype) else getattr(torch, str(name))
    if dt not in _ALLOWED_DTYPES:
        raise ValueError(
            f"dtype {dt} is not permitted in scwbd.infer; solvers, Fisher "
            "information and covariance propagation require float32 minimum "
            "(ARCHITECTURE.md sec. 3)."
        )
    return dt


def default_device() -> torch.device:
    env = os.environ.get("SCWBD_DEVICE")
    if env:
        return torch.device(env)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_device(device: str | torch.device | None) -> torch.device:
    return default_device() if device is None else torch.device(device)


def cap_gpu_memory(gigabytes: float = 20.0, *, device: int = 0) -> dict[str, Any]:
    """Impose a **device-side** cap on this process's CUDA allocations.

    Host cgroup limits (``systemd-run -p MemoryMax=``) do not bound CUDA
    allocations on GB10 unified memory: the GPU allocation is not charged to
    the cgroup, so ``memory.current`` reads green while the caching allocator
    grows without bound.  ``torch.cuda.set_per_process_memory_fraction`` is
    enforced by the allocator itself and therefore actually fires.

    Returns the applied settings so a run can record them in its provenance
    instead of asserting them.
    """
    if not torch.cuda.is_available():
        return {"applied": False, "reason": "no cuda device"}
    total = torch.cuda.get_device_properties(device).total_memory
    frac = min(max(gigabytes * (1024**3) / total, 0.01), 1.0)
    torch.cuda.set_per_process_memory_fraction(frac, device)
    return {
        "applied": True,
        "device": torch.cuda.get_device_name(device),
        "total_gib": total / 1024**3,
        "cap_gib": gigabytes,
        "fraction": frac,
        "mechanism": "torch.cuda.set_per_process_memory_fraction (allocator-enforced)",
        "note": "host cgroup MemoryMax does not bound CUDA allocations on unified "
                "memory; verify with nvidia-smi --query-compute-apps",
    }


def gpu_memory_report(device: int = 0) -> dict[str, Any]:
    """Live allocator state, for provenance.  Reserved != allocated."""
    if not torch.cuda.is_available():
        return {"cuda": False}
    return {
        "cuda": True,
        "allocated_gib": torch.cuda.memory_allocated(device) / 1024**3,
        "reserved_gib": torch.cuda.memory_reserved(device) / 1024**3,
        "max_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
    }


def seed_everything(seed: int) -> torch.Generator:
    """Seed python/numpy/torch and return a CPU generator seeded identically."""
    if not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an int; stochastic entry points are explicit")
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return gen


def as_builtin(obj: Any) -> Any:
    """Recursively convert numpy/torch objects into JSON-serialisable builtins."""
    if isinstance(obj, torch.Tensor):
        return as_builtin(obj.detach().cpu().numpy())
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return as_builtin(obj.item())
        return [as_builtin(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if math.isnan(v):
            return None
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return v
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): as_builtin(v) for k, v in obj.items()}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: as_builtin(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, (list, tuple, set)):
        return [as_builtin(v) for v in obj]
    return obj


# --------------------------------------------------------------------------
# R09: pseudo-posteriors may not inherit calibration claims
# --------------------------------------------------------------------------

PosteriorKind = Literal["bayesian", "generalized_posterior"]


class CalibrationClaimError(RuntimeError):
    """Refusal **R09**.

    Raised when a posterior that was produced (wholly or partly) from
    pseudo-losses -- boundary, scale, distillation or compatibility penalties --
    is asked to behave as a calibrated Bayesian posterior.  Remedy: either
    calibrate the generalized posterior empirically and report it as such, or
    validate and promote the offending factor to a normalized generative
    likelihood.
    """

    code = "R09"

    def __init__(self, message: str, *, offending_object: str = "", remedy: str = ""):
        self.offending_object = offending_object
        self.remedy = remedy or (
            "Report a generalized/pseudo-posterior and calibrate it empirically, "
            "or validate and promote the factor to a generative likelihood."
        )
        super().__init__(f"[R09] {message} Remedy: {self.remedy}")


@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float
    level: float = 0.95

    def contains(self, x: float) -> bool:
        return bool(self.lo <= x <= self.hi)

    @property
    def width(self) -> float:
        return float(self.hi - self.lo)


@dataclass
class PosteriorSummary:
    """A parameter posterior with an explicit calibration provenance.

    ``kind='generalized_posterior'`` marks a posterior whose objective included
    at least one un-normalized pseudo-loss.  ``assert_calibrated_bayesian``
    raises :class:`CalibrationClaimError` for such an object.
    """

    names: list[str]
    mean: np.ndarray
    cov: np.ndarray
    kind: PosteriorKind = "bayesian"
    pseudo_loss_terms: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=float)
        self.cov = np.asarray(self.cov, dtype=float)
        if self.pseudo_loss_terms and self.kind != "generalized_posterior":
            raise CalibrationClaimError(
                "posterior was built with auxiliary pseudo-losses "
                f"{sorted(self.pseudo_loss_terms)} but is tagged 'bayesian'.",
                offending_object="PosteriorSummary",
            )

    @property
    def sd(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.cov), 0.0, None))

    def assert_calibrated_bayesian(self, context: str = "") -> None:
        """Gate every calibration claim through this method."""
        if self.kind != "bayesian":
            raise CalibrationClaimError(
                f"{context or 'this report'} requests calibrated Bayesian "
                "posterior semantics from a generalized posterior derived from "
                f"pseudo-losses {sorted(self.pseudo_loss_terms)}.",
                offending_object="PosteriorSummary",
            )

    def interval(self, name: str, level: float = 0.95) -> Interval:
        i = self.names.index(name)
        z = float(_norm_ppf(0.5 * (1.0 + level)))
        s = float(self.sd[i])
        return Interval(float(self.mean[i] - z * s), float(self.mean[i] + z * s), level)

    def credible_intervals(self, level: float = 0.95) -> dict[str, Interval]:
        return {n: self.interval(n, level) for n in self.names}

    def correlation(self) -> np.ndarray:
        s = self.sd
        s = np.where(s > 0, s, np.inf)
        return self.cov / np.outer(s, s)


def _norm_ppf(p: float) -> float:
    from scipy.stats import norm  # local import: scipy is heavy

    return float(norm.ppf(p))


@dataclass
class CoverageResult:
    """Empirical interval coverage with a binomial (Wilson) error bar."""

    name: str
    nominal: float
    n: int
    n_covered: int

    @property
    def empirical(self) -> float:
        return self.n_covered / self.n if self.n else float("nan")

    @property
    def wilson_interval(self) -> Interval:
        if self.n == 0:
            return Interval(float("nan"), float("nan"), 0.95)
        z = 1.959963984540054
        p = self.empirical
        n = self.n
        denom = 1.0 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return Interval(max(0.0, centre - half), min(1.0, centre + half), 0.95)

    @property
    def standard_error(self) -> float:
        p = self.empirical
        return math.sqrt(max(p * (1 - p), 0.0) / self.n) if self.n else float("nan")

    def is_nominal(self, tol: float = 0.0) -> bool:
        """True when the nominal level lies inside the Wilson interval."""
        ci = self.wilson_interval
        return (ci.lo - tol) <= self.nominal <= (ci.hi + tol)

    def to_dict(self) -> dict[str, Any]:
        ci = self.wilson_interval
        return {
            "name": self.name,
            "nominal": self.nominal,
            "n_replicates": self.n,
            "n_covered": self.n_covered,
            "empirical": self.empirical,
            "standard_error": self.standard_error,
            "wilson95_lo": ci.lo,
            "wilson95_hi": ci.hi,
            "nominal_inside_wilson95": self.is_nominal(),
        }


# --------------------------------------------------------------------------
# Posterior branching (body.tex sec. 7.1)
# --------------------------------------------------------------------------


@dataclass
class UnresolvedCausalAmbiguity:
    """Returned when models are observationally equivalent but differ causally.

    ``body.tex`` sec. 7.1: *"When two models remain observationally equivalent
    but imply different treatments, the output is an unresolved causal
    ambiguity, not an averaged recommendation."*  This object is deliberately
    **not** an exception -- callers must be able to branch on it -- and it
    deliberately exposes no ``mean``/``best`` accessor.
    """

    candidate_models: list[str]
    log_evidence: dict[str, float]
    max_log_evidence_gap: float
    equivalence_threshold: float
    divergent_interventions: dict[str, dict[str, float]]
    intervention_divergence: float
    divergence_threshold: float
    posterior_weights: dict[str, float]
    resolution_experiment: str
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.candidate_models) < 2:
            raise ValueError("an ambiguity needs at least two candidate models")

    @property
    def reason(self) -> str:
        return (
            f"models {self.candidate_models} are observationally equivalent "
            f"(max |Delta log evidence| = {self.max_log_evidence_gap:.3g} <= "
            f"{self.equivalence_threshold:.3g}) but their intervention forecasts "
            f"diverge (standardised divergence {self.intervention_divergence:.3g} > "
            f"{self.divergence_threshold:.3g})."
        )

    def averaged_recommendation(self) -> None:
        raise CalibrationClaimError(
            "an UnresolvedCausalAmbiguity has no averaged recommendation; "
            f"{self.reason}",
            offending_object="UnresolvedCausalAmbiguity",
            remedy=self.resolution_experiment,
        )

    def to_dict(self) -> dict[str, Any]:
        return as_builtin(
            {
                "type": "UnresolvedCausalAmbiguity",
                "candidate_models": self.candidate_models,
                "log_evidence": self.log_evidence,
                "max_log_evidence_gap": self.max_log_evidence_gap,
                "equivalence_threshold": self.equivalence_threshold,
                "divergent_interventions": self.divergent_interventions,
                "intervention_divergence": self.intervention_divergence,
                "divergence_threshold": self.divergence_threshold,
                "posterior_weights": self.posterior_weights,
                "resolution_experiment": self.resolution_experiment,
                "reason": self.reason,
                "detail": self.detail,
            }
        )


def stack_named(values: Sequence[float], names: Sequence[str]) -> dict[str, float]:
    return {str(n): float(v) for n, v in zip(names, values)}
