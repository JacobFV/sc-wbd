"""Core types for the SC-WBD dynamics core (agent E).

Everything here is plain ``torch`` + dataclasses so that the dynamics core can be
developed and tested independently of :mod:`scwbd.schema` (agent A),
:mod:`scwbd.anatomy` (agent C) and :mod:`scwbd.transforms` (agent D).  Adapters
to those modules are exposed as thin ``from_schema`` / ``from_prior`` functions
which duck-type on the foreign objects, so they start working the moment those
modules land without creating an import-time dependency.

Numerical contract (ARCHITECTURE.md §3):

* solver state is ``float32`` (``float64`` permitted for convergence studies);
  ``bfloat16`` is permitted *only* inside learned operators, never in a solver;
* every stochastic entry point takes an explicit ``seed``;
* the parameter dimension is **batched** — every tensor carries a leading
  ``batch`` axis over parameter sets and is evaluated in parallel on device.

Shape conventions used throughout the dynamics core::

    x            (B, N, D)     state: batch x regions x per-region state dim
    coupling     (B, N, K)     coupling input channels consumed by the backend
    theta[name]  broadcastable to (B, N, 1)
    activity     (B, N)        the scalar coupling / observation variable
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor

__all__ = [
    "SOLVER_DTYPES",
    "DTYPE",
    "Prior",
    "ParamPack",
    "broadcast_param",
    "default_device",
    "make_generator",
    "assert_solver_dtype",
    "NumericalBudget",
    "GuardViolation",
    "RefusalError",
    "MechanismRefusal",
    "SemigroupRefusal",
    "SemanticCollapseError",
]

#: dtypes a solver may run in.  bf16/fp16 in a solver is a bug, not an
#: optimisation (ARCHITECTURE.md §3).
SOLVER_DTYPES = (torch.float32, torch.float64)
DTYPE = torch.float32


def default_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve the device for dynamics work: explicit > CUDA > CPU."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_generator(seed: int, device: str | torch.device | None = None) -> torch.Generator:
    """Deterministic per-call RNG.  Determinism is a test, not an aspiration."""
    dev = default_device(device)
    g = torch.Generator(device=dev)
    g.manual_seed(int(seed))
    return g


def assert_solver_dtype(x: Tensor, what: str = "solver state") -> None:
    if x.dtype not in SOLVER_DTYPES:
        raise TypeError(
            f"{what} has dtype {x.dtype}; solvers, Fisher information and covariance "
            f"propagation must stay in {SOLVER_DTYPES} (ARCHITECTURE.md §3). "
            "bfloat16 is permitted only inside learned operators."
        )


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Prior:
    """A minimal, batched prior over a scalar dynamical parameter.

    This is deliberately *not* ``scwbd.schema.Prior`` — agent A owns that type.
    :meth:`from_schema` adapts any object exposing ``dist``/``loc``/``scale`` or
    a ``.sample()`` method, so schema priors drop in unchanged.
    """

    name: str
    loc: float
    scale: float = 0.0
    dist: str = "normal"  # normal | lognormal | uniform | delta
    units: str = "dimensionless"
    low: float | None = None
    high: float | None = None
    provenance: str = "unspecified"

    def sample(
        self,
        shape: Sequence[int],
        *,
        seed: int,
        device: str | torch.device | None = None,
        dtype: torch.dtype = DTYPE,
    ) -> Tensor:
        g = make_generator(seed, device)
        dev = g.device
        shape = tuple(int(s) for s in shape)
        if self.dist == "delta" or self.scale == 0.0:
            out = torch.full(shape, float(self.loc), device=dev, dtype=dtype)
        elif self.dist == "normal":
            out = self.loc + self.scale * torch.randn(shape, generator=g, device=dev, dtype=dtype)
        elif self.dist == "lognormal":
            z = torch.randn(shape, generator=g, device=dev, dtype=dtype)
            out = math.exp(self.loc) * torch.exp(self.scale * z)
        elif self.dist == "uniform":
            lo = self.low if self.low is not None else self.loc - self.scale
            hi = self.high if self.high is not None else self.loc + self.scale
            u = torch.rand(shape, generator=g, device=dev, dtype=dtype)
            out = lo + (hi - lo) * u
        else:  # pragma: no cover - guarded by validation below
            raise ValueError(f"unknown prior family {self.dist!r}")
        if self.low is not None or self.high is not None:
            out = out.clamp(
                min=-math.inf if self.low is None else self.low,
                max=math.inf if self.high is None else self.high,
            )
        return out

    def mean(self) -> float:
        if self.dist == "lognormal":
            return math.exp(self.loc + 0.5 * self.scale**2)
        if self.dist == "uniform":
            lo = self.low if self.low is not None else self.loc - self.scale
            hi = self.high if self.high is not None else self.loc + self.scale
            return 0.5 * (lo + hi)
        return float(self.loc)

    # -- adapters ----------------------------------------------------------
    @classmethod
    def from_schema(cls, obj: Any, name: str = "param") -> "Prior":
        """Adapt an ``scwbd.schema.Prior`` (or anything shaped like one)."""
        if isinstance(obj, Prior):
            return obj
        if isinstance(obj, (int, float)):
            return cls(name=name, loc=float(obj), scale=0.0, dist="delta")
        get = lambda k, d=None: getattr(obj, k, None) if not isinstance(obj, Mapping) else obj.get(k, d)
        dist = get("dist") or get("family") or "normal"
        loc = get("loc")
        if loc is None:
            loc = get("mean")
        scale = get("scale")
        if scale is None:
            scale = get("sd") or 0.0
        if loc is None:
            raise TypeError(f"cannot adapt {obj!r} to a dynamics Prior: no loc/mean")
        return cls(
            name=str(get("name") or name),
            loc=float(loc),
            scale=float(scale),
            dist=str(dist),
            units=str(get("units") or "dimensionless"),
            low=None if get("low") is None else float(get("low")),
            high=None if get("high") is None else float(get("high")),
            provenance=str(get("provenance") or "schema"),
        )


# ---------------------------------------------------------------------------
# Batched parameter packs
# ---------------------------------------------------------------------------


def broadcast_param(
    value: Tensor | float | int,
    batch: int,
    n_regions: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = DTYPE,
) -> Tensor:
    """Broadcast a parameter to ``(B, N, 1)``.

    Accepted shapes: scalar, ``(B,)`` (per parameter set), ``(B, N)``
    (per parameter set and region), ``(B, N, 1)``, or ``(1, N)``.  ``(N,)`` is
    accepted only when ``N != B`` (otherwise it is ambiguous and rejected — an
    ambiguous parameter axis is exactly the kind of silent error this codebase
    is supposed to refuse).
    """
    if not isinstance(value, Tensor):
        return torch.as_tensor(float(value), device=device, dtype=dtype).reshape(1, 1, 1)
    v = value.to(device=device, dtype=dtype) if device is not None else value.to(dtype)
    if v.ndim == 0:
        return v.reshape(1, 1, 1)
    if v.ndim == 1:
        if v.shape[0] == batch and v.shape[0] == n_regions and batch != 1:
            raise ValueError(
                "1-D parameter of length equal to both batch and n_regions is ambiguous; "
                "pass shape (B,1) or (1,N) explicitly."
            )
        if v.shape[0] == batch:
            return v.reshape(batch, 1, 1)
        if v.shape[0] == n_regions:
            return v.reshape(1, n_regions, 1)
        raise ValueError(f"parameter of shape {tuple(v.shape)} matches neither B={batch} nor N={n_regions}")
    if v.ndim == 2:
        return v.reshape(v.shape[0], v.shape[1], 1)
    if v.ndim == 3:
        return v
    raise ValueError(f"parameter has too many dims: {tuple(v.shape)}")


@dataclass
class ParamPack:
    """A batch of parameter sets for one backend.

    ``values`` maps parameter name -> tensor broadcastable to ``(B, N, 1)``.
    Missing names fall back to the backend's ``defaults``.  Everything stays on
    device; nothing is ever moved to CPU during a rollout.
    """

    values: dict[str, Tensor | float] = field(default_factory=dict)
    batch: int = 1
    n_regions: int = 1
    device: torch.device | None = None
    dtype: torch.dtype = DTYPE
    defaults: Mapping[str, float] = field(default_factory=dict)
    #: Where non-default parameter values came from, keyed by parameter name.
    #: Populated by :meth:`DynamicsBackend.theta_from_prior`, which records the
    #: anatomical prior's own citation text and the transform applied to it, so a
    #: parameter drawn from an atlas proxy is never mistaken for a measurement.
    #: Carried through ``with_``/``to``/``detach``: provenance that silently
    #: vanished on a device move would be worse than no provenance at all.
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.device = default_device(self.device)
        # Broadcasting a scalar parameter costs a host->device copy (~0.7 ms on
        # GB10). In a rollout every parameter is fetched twice per step, so the
        # broadcast result is memoised; `set`/`with_` invalidate it.
        self._cache: dict[tuple[str, float | None], Tensor] = {}

    def get(self, name: str, default: float | Tensor | None = None) -> Tensor:
        key = (name, default if isinstance(default, (int, float, type(None))) else id(default))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        if name in self.values:
            v = self.values[name]
        elif name in self.defaults:
            v = self.defaults[name]
        elif default is not None:
            v = default
        else:
            raise KeyError(f"parameter {name!r} not provided and has no default")
        out = broadcast_param(v, self.batch, self.n_regions, device=self.device, dtype=self.dtype)
        self._cache[key] = out
        return out

    def __contains__(self, name: str) -> bool:
        return name in self.values or name in self.defaults

    def set(self, name: str, value: Tensor | float) -> "ParamPack":
        self.values[name] = value
        self._cache.clear()
        return self

    def with_(self, **kw: Tensor | float) -> "ParamPack":
        """Functional update — returns a new pack (shares tensors)."""
        return ParamPack(
            values={**self.values, **kw},
            batch=self.batch,
            n_regions=self.n_regions,
            device=self.device,
            dtype=self.dtype,
            defaults=self.defaults,
            provenance=dict(self.provenance),
        )

    def to(self, device: str | torch.device) -> "ParamPack":
        dev = torch.device(device)
        vals = {k: (v.to(dev) if isinstance(v, Tensor) else v) for k, v in self.values.items()}
        return ParamPack(vals, self.batch, self.n_regions, dev, self.dtype, self.defaults,
                         dict(self.provenance))

    def names(self) -> list[str]:
        return sorted(set(self.values) | set(self.defaults))

    def detach(self) -> "ParamPack":
        vals = {k: (v.detach() if isinstance(v, Tensor) else v) for k, v in self.values.items()}
        return ParamPack(vals, self.batch, self.n_regions, self.device, self.dtype, self.defaults,
                         dict(self.provenance))

    # -- constructors ------------------------------------------------------
    @classmethod
    def sample(
        cls,
        priors: Mapping[str, Prior | float],
        *,
        batch: int,
        n_regions: int,
        seed: int,
        per_region: Iterable[str] = (),
        device: str | torch.device | None = None,
        dtype: torch.dtype = DTYPE,
        defaults: Mapping[str, float] | None = None,
    ) -> "ParamPack":
        """Sample a batch of parameter sets from priors, on device.

        Names listed in ``per_region`` get an independent draw per region
        (shape ``(B, N)``); everything else is drawn per parameter set
        ``(B, 1)``.
        """
        dev = default_device(device)
        per_region = set(per_region)
        vals: dict[str, Tensor | float] = {}
        for i, (name, p) in enumerate(sorted(priors.items())):
            pr = Prior.from_schema(p, name)
            shape = (batch, n_regions) if name in per_region else (batch, 1)
            vals[name] = pr.sample(shape, seed=seed + 1013 * i, device=dev, dtype=dtype)
        return cls(vals, batch, n_regions, dev, dtype, dict(defaults or {}))


# ---------------------------------------------------------------------------
# Budgets and refusals
# ---------------------------------------------------------------------------


@dataclass
class NumericalBudget:
    """The numerical / model-discrepancy budget a run accumulates.

    Per thesis §4.5 temporal coarsening and semigroup residuals are *reported*
    contributions to posterior uncertainty, not invisible implementation
    choices.  Every entry is a variance contribution in state units squared.
    """

    entries: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def add(self, key: str, variance: float, note: str | None = None) -> None:
        self.entries[key] = self.entries.get(key, 0.0) + float(variance)
        if note:
            self.notes.append(note)

    @property
    def total_variance(self) -> float:
        return float(sum(self.entries.values()))

    def as_dict(self) -> dict[str, Any]:
        return {"entries": dict(self.entries), "total_variance": self.total_variance, "notes": list(self.notes)}


@dataclass
class GuardViolation:
    """A machine-readable record that a runtime guard fired."""

    code: str  # R05, R06, ...
    detail: str
    value: float
    tolerance: float
    offending_object: str = ""
    remedy: str = ""

    def __str__(self) -> str:  # pragma: no cover - formatting
        return (
            f"[{self.code}] {self.detail}: {self.value:.4g} > tol {self.tolerance:.4g} "
            f"({self.offending_object}). Remedy: {self.remedy}"
        )


class RefusalError(RuntimeError):
    """Base class for compiler/runtime refusals (thesis_contract Table 1)."""

    code = "R00"

    def __init__(self, violation: GuardViolation):
        self.violation = violation
        super().__init__(str(violation))


class MechanismRefusal(RefusalError):
    """R05 — learned residual silently dominating a mechanistic term."""

    code = "R05"


class SemigroupRefusal(RefusalError):
    """R06 — adaptive stepping for a learned propagator without semigroup testing."""

    code = "R06"


class SemanticCollapseError(ValueError):
    """Raised when a neuromodulator is equated with a psychological construct.

    Thesis §5 is explicit: dopamine is not reward, ACh is not attention, NE is
    not arousal.  Neuromodulators are receptor-, target-, timescale- and
    state-dependent gain/plasticity fields.
    """
