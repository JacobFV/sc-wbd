"""Typed priors.

Every free quantity in a schema (delays, gains, calibration coefficients,
hierarchical effects) is declared as a ``Prior``, never as a bare float.  Each
prior carries units, supports deterministic ``sample(seed)`` (ARCHITECTURE.md
sec. 3: "all stochastic entry points take an explicit seed") and an analytic
``logpdf``.
"""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal, Sequence, Union

import numpy as np
from pydantic import Field, field_validator, model_validator

from .base import SchemaModel
from .units import Unit

__all__ = [
    "PriorBase",
    "NormalPrior",
    "LogNormalPrior",
    "UniformPrior",
    "BetaPrior",
    "GammaPrior",
    "DiracPrior",
    "Prior",
    "as_prior",
]

_ArrayLike = Union[float, Sequence[float], np.ndarray]


class PriorBase(SchemaModel):
    """Common interface for every prior family."""

    units: Unit = Unit("dimensionless")
    #: Free text describing where the prior came from (literature, phantom,
    #: previous fit).  Empty provenance is legal but the compiler records it.
    provenance: str = ""

    # -- interface ----------------------------------------------------------
    def _rng(self, seed: int) -> np.random.Generator:
        if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
            raise TypeError("seed must be an int; determinism is a test, not an aspiration")
        return np.random.default_rng(int(seed))

    def sample(self, seed: int, size: int | tuple[int, ...] | None = None) -> Any:
        raise NotImplementedError

    def logpdf(self, x: _ArrayLike) -> Any:
        raise NotImplementedError

    def mean(self) -> float:
        raise NotImplementedError

    def support(self) -> tuple[float, float]:
        return (-math.inf, math.inf)

    def in_support(self, x: _ArrayLike) -> Any:
        lo, hi = self.support()
        arr = np.asarray(x, dtype=float)
        return np.logical_and(arr >= lo, arr <= hi)


class NormalPrior(PriorBase):
    kind: Literal["normal"] = "normal"
    loc: float
    scale: float = Field(gt=0.0)

    def sample(self, seed: int, size: int | tuple[int, ...] | None = None) -> Any:
        return self._rng(seed).normal(self.loc, self.scale, size)

    def logpdf(self, x: _ArrayLike) -> Any:
        z = (np.asarray(x, dtype=float) - self.loc) / self.scale
        out = -0.5 * z**2 - math.log(self.scale) - 0.5 * math.log(2.0 * math.pi)
        return float(out) if np.ndim(out) == 0 else out

    def mean(self) -> float:
        return self.loc


class LogNormalPrior(PriorBase):
    """Log-normal: ``log(x) ~ Normal(mu, sigma)``.  Support ``x > 0``."""

    kind: Literal["lognormal"] = "lognormal"
    mu: float
    sigma: float = Field(gt=0.0)

    def sample(self, seed: int, size: int | tuple[int, ...] | None = None) -> Any:
        return self._rng(seed).lognormal(self.mu, self.sigma, size)

    def logpdf(self, x: _ArrayLike) -> Any:
        arr = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (np.log(arr) - self.mu) / self.sigma
            out = (
                -0.5 * z**2
                - np.log(arr)
                - math.log(self.sigma)
                - 0.5 * math.log(2.0 * math.pi)
            )
        out = np.where(arr > 0.0, out, -np.inf)
        return float(out) if np.ndim(out) == 0 else out

    def mean(self) -> float:
        return math.exp(self.mu + 0.5 * self.sigma**2)

    def support(self) -> tuple[float, float]:
        return (0.0, math.inf)


class UniformPrior(PriorBase):
    kind: Literal["uniform"] = "uniform"
    low: float
    high: float

    @model_validator(mode="after")
    def _ordered(self) -> "UniformPrior":
        if not self.high > self.low:
            raise ValueError(f"uniform prior needs high > low, got {self.low}, {self.high}")
        return self

    def sample(self, seed: int, size: int | tuple[int, ...] | None = None) -> Any:
        return self._rng(seed).uniform(self.low, self.high, size)

    def logpdf(self, x: _ArrayLike) -> Any:
        arr = np.asarray(x, dtype=float)
        inside = np.logical_and(arr >= self.low, arr <= self.high)
        out = np.where(inside, -math.log(self.high - self.low), -np.inf)
        return float(out) if np.ndim(out) == 0 else out

    def mean(self) -> float:
        return 0.5 * (self.low + self.high)

    def support(self) -> tuple[float, float]:
        return (self.low, self.high)


class BetaPrior(PriorBase):
    kind: Literal["beta"] = "beta"
    alpha: float = Field(gt=0.0)
    beta: float = Field(gt=0.0)

    def sample(self, seed: int, size: int | tuple[int, ...] | None = None) -> Any:
        return self._rng(seed).beta(self.alpha, self.beta, size)

    def logpdf(self, x: _ArrayLike) -> Any:
        arr = np.asarray(x, dtype=float)
        log_b = (
            math.lgamma(self.alpha)
            + math.lgamma(self.beta)
            - math.lgamma(self.alpha + self.beta)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            out = (
                (self.alpha - 1.0) * np.log(arr)
                + (self.beta - 1.0) * np.log1p(-arr)
                - log_b
            )
        out = np.where(np.logical_and(arr > 0.0, arr < 1.0), out, -np.inf)
        return float(out) if np.ndim(out) == 0 else out

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def support(self) -> tuple[float, float]:
        return (0.0, 1.0)


class GammaPrior(PriorBase):
    """Shape/rate parameterization (rate = 1 / scale)."""

    kind: Literal["gamma"] = "gamma"
    shape: float = Field(gt=0.0)
    rate: float = Field(gt=0.0)

    def sample(self, seed: int, size: int | tuple[int, ...] | None = None) -> Any:
        return self._rng(seed).gamma(self.shape, 1.0 / self.rate, size)

    def logpdf(self, x: _ArrayLike) -> Any:
        arr = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = (
                self.shape * math.log(self.rate)
                - math.lgamma(self.shape)
                + (self.shape - 1.0) * np.log(arr)
                - self.rate * arr
            )
        out = np.where(arr > 0.0, out, -np.inf)
        return float(out) if np.ndim(out) == 0 else out

    def mean(self) -> float:
        return self.shape / self.rate

    def support(self) -> tuple[float, float]:
        return (0.0, math.inf)


class DiracPrior(PriorBase):
    """A quantity fixed by declaration.

    A Dirac prior is how a schema says "this is known and not estimated".  It
    is *not* a way to smuggle a point estimate of an uncertain quantity: the
    bias/variance ledger checks (R08) apply independently.
    """

    kind: Literal["dirac"] = "dirac"
    value: float

    def sample(self, seed: int, size: int | tuple[int, ...] | None = None) -> Any:
        self._rng(seed)  # validate seed type even though unused
        if size is None:
            return float(self.value)
        return np.full(size, float(self.value), dtype=float)

    def logpdf(self, x: _ArrayLike) -> Any:
        arr = np.asarray(x, dtype=float)
        out = np.where(arr == self.value, 0.0, -np.inf)
        return float(out) if np.ndim(out) == 0 else out

    def mean(self) -> float:
        return float(self.value)

    def support(self) -> tuple[float, float]:
        return (float(self.value), float(self.value))


Prior = Annotated[
    Union[
        NormalPrior,
        LogNormalPrior,
        UniformPrior,
        BetaPrior,
        GammaPrior,
        DiracPrior,
    ],
    Field(discriminator="kind"),
]

_KINDS: dict[str, type[PriorBase]] = {
    "normal": NormalPrior,
    "lognormal": LogNormalPrior,
    "uniform": UniformPrior,
    "beta": BetaPrior,
    "gamma": GammaPrior,
    "dirac": DiracPrior,
}


def as_prior(obj: Any) -> PriorBase:
    """Coerce a dict (or a prior) into the right concrete prior class."""
    if isinstance(obj, PriorBase):
        return obj
    if isinstance(obj, dict) and "kind" in obj:
        return _KINDS[obj["kind"]](**{k: v for k, v in obj.items() if k != "kind"})
    raise TypeError(f"cannot interpret {obj!r} as a Prior")
