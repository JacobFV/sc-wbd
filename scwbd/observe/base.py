"""Common contract for every SC-WBD observation operator.

Thesis anchors
--------------
* body.tex Sec. 2.4 --- ``Y_m(t) = O_m(X_{0:t}, A_p, B_p; psi_m) + eps_m(t)``.
  The latent process is *not* defined by any one instrument and instruments are
  never coerced into a common resolution.
* body.tex Sec. 2.7 --- ``Y_s = O_s(X; D_s, r_s) + b_s(D_s, r_s, C) + eps_s``.
  "Each read returns a prediction, variance decomposition, estimated bias
  range, model-discrepancy flag, provenance, and validity domain."  Bias and
  variance NEVER collapse into one score.
* thesis_contract.tex Sec. 0.1 refusal **R08** --- "Bias assigned a point
  estimate without an estimator or external bound".  Implemented here as a
  hard, raising validation on :class:`BiasTerm`.

Design notes
------------
The module is deliberately dependency-light: ``torch`` + ``dataclasses``.  The
frozen ``pydantic`` mirrors in ``scwbd.schema`` (agent A) are attached through
the thin adapters :func:`to_schema_support` / :func:`to_schema_ledger`, which
degrade to ``None`` when the schema package has not yet landed.  Nothing in
``scwbd.observe`` imports ``scwbd.schema`` at module scope.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

import torch

__all__ = [
    "Unit",
    "VOLT",
    "TESLA",
    "DIMENSIONLESS",
    "ObservationRefusal",
    "RefusalR08",
    "Unresolved",
    "Prior",
    "PSF",
    "Support",
    "TemporalSupport",
    "VarianceDecomposition",
    "BiasTerm",
    "UncertaintyLedger",
    "Provenance",
    "ObservationRead",
    "ObservationOperator",
    "UNKNOWN",
    "to_schema_support",
    "to_schema_psf",
    "to_schema_temporal",
    "to_schema_ledger",
    "to_schema_ledgers",
    "validate_unit",
    "MICROMOLAR",
]

# --------------------------------------------------------------------------
# units
# --------------------------------------------------------------------------

Unit = str

VOLT: Unit = "V"
TESLA: Unit = "T"
AMPERE_METER: Unit = "A*m"  # current dipole moment
AMPERE_PER_M2: Unit = "A/m^2"
MOLAR: Unit = "mol/m^3"
SECOND: Unit = "s"
DIMENSIONLESS: Unit = "dimensionless"
PERCENT: Unit = "%"

MICROMOLAR: Unit = "mmol/m^3"
"""Micromolar, spelled in the schema's unit algebra (1 uM = 1 mmol/m^3)."""

_FALLBACK_UNITS: frozenset[str] = frozenset(
    {
        "V",
        "T",
        "T/m",
        "A*m",
        "A/m^2",
        # lead-field transfer units: a forward operator is not a measurement,
        # it is the ratio of one to a source amplitude.
        "V/(A*m)",
        "T/(A*m)",
        "V^2/(A*m)^2",
        "T^2/(A*m)^2",
        "S/m",
        "m",
        "m^2",
        "s",
        "s^2",
        "Hz",
        "Pa",
        "mol/m^3",
        "mmol/m^3",
        "dimensionless",
        "%",
        "ohm",
        "K",
        "V^2",
        "T^2",
    }
)


def _schema_unit_validator():  # pragma: no cover - depends on agent A
    try:
        from scwbd.schema.units import Unit as _SchemaUnit

        return _SchemaUnit
    except Exception:
        return None


def validate_unit(unit: Unit) -> Unit:
    """Refusal R01 guard: an unknown unit string is never silently accepted.

    Delegates to agent A's dimensional algebra (``scwbd.schema.units``) so that
    ``scwbd.observe`` and the compiler can never drift apart on what a unit
    string means; falls back to a literal allow-list when the schema package is
    not importable.
    """
    checker = _schema_unit_validator()
    if checker is not None:
        try:
            checker(unit)
            return unit
        except Exception as exc:
            raise ObservationRefusal(
                code="R01",
                message=f"unknown unit {unit!r}: {exc}",
                remedy="use a unit expression in scwbd.schema.units, or register "
                "it there with a documented conversion",
                offending_object=unit,
            ) from None
    if unit not in _FALLBACK_UNITS:
        raise ObservationRefusal(
            code="R01",
            message=f"unknown unit {unit!r}; supply a registered dimensional string",
            remedy="add the unit to the registry with a documented conversion, "
            "or keep the source disconnected from the requested path",
            offending_object=unit,
        )
    return unit


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------


class ObservationRefusal(Exception):
    """A modeling refusal from ``tab:compiler-refusals`` raised at read time.

    These are errors, not warnings: an observation operator that cannot honour
    the contract must not return a number.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        remedy: str,
        offending_object: Any = None,
    ) -> None:
        super().__init__(f"[{code}] {message}\n  remedy: {remedy}")
        self.code = code
        self.message = message
        self.remedy = remedy
        self.offending_object = offending_object


class RefusalR08(ObservationRefusal):
    """Bias point estimate without an estimator or an external bound."""

    def __init__(self, message: str, *, offending_object: Any = None) -> None:
        super().__init__(
            code="R08",
            message=message,
            remedy=(
                "classify the term as design_estimable (name the estimator), "
                "externally_bounded (name the phantom/calibration target/"
                "negative control), or prior_specified_sensitivity (propagate "
                "the declared range, never a point value)"
            ),
            offending_object=offending_object,
        )


@dataclass(frozen=True)
class Unresolved:
    """Returned instead of a number when a read cannot be supported.

    ARCHITECTURE.md Sec. 6: "A runtime read that cannot be supported returns
    ``Unresolved(reason=...)`` rather than a number."
    """

    reason: str
    missing: tuple[str, ...] = ()

    def __bool__(self) -> bool:  # pragma: no cover - guard against `if read:`
        return False


# --------------------------------------------------------------------------
# priors  (conductivity, HRF shape, DPF ... are Priors, never constants)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Prior:
    """A scalar nuisance parameter with a distribution and a citation.

    ``dist`` is one of ``normal``, ``lognormal``, ``uniform``, ``delta``.  A
    ``delta`` prior is permitted but records that the value is *asserted*, which
    is what makes it auditable rather than invisible.
    """

    name: str
    dist: Literal["normal", "lognormal", "uniform", "delta"]
    params: tuple[float, ...]
    units: Unit = DIMENSIONLESS
    source: str = "unspecified"
    validity: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        validate_unit(self.units)
        n = {"normal": 2, "lognormal": 2, "uniform": 2, "delta": 1}[self.dist]
        if len(self.params) != n:
            raise ValueError(
                f"prior {self.name!r}: dist {self.dist} needs {n} params, "
                f"got {len(self.params)}"
            )
        if self.dist in ("normal", "lognormal") and self.params[1] < 0:
            raise ValueError(f"prior {self.name!r}: negative scale")
        if self.dist == "uniform" and self.params[0] > self.params[1]:
            raise ValueError(f"prior {self.name!r}: empty uniform support")

    # -- moments -----------------------------------------------------------
    @property
    def mean(self) -> float:
        if self.dist == "delta":
            return self.params[0]
        if self.dist == "normal":
            return self.params[0]
        if self.dist == "lognormal":
            mu, s = self.params
            return math.exp(mu + 0.5 * s * s)
        lo, hi = self.params
        return 0.5 * (lo + hi)

    @property
    def sd(self) -> float:
        if self.dist == "delta":
            return 0.0
        if self.dist == "normal":
            return self.params[1]
        if self.dist == "lognormal":
            mu, s = self.params
            return math.sqrt((math.exp(s * s) - 1.0)) * math.exp(mu + 0.5 * s * s)
        lo, hi = self.params
        return (hi - lo) / math.sqrt(12.0)

    def sample(
        self,
        shape: Sequence[int] = (),
        *,
        seed: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Draw from the prior.  ``seed`` is mandatory (ARCHITECTURE.md Sec. 3)."""
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        shape = tuple(shape)
        if self.dist == "delta":
            out = torch.full(shape, float(self.params[0]))
        elif self.dist == "normal":
            mu, s = self.params
            out = mu + s * torch.randn(shape, generator=g)
        elif self.dist == "lognormal":
            mu, s = self.params
            out = torch.exp(mu + s * torch.randn(shape, generator=g))
        else:
            lo, hi = self.params
            out = lo + (hi - lo) * torch.rand(shape, generator=g)
        return out.to(device=device, dtype=dtype)

    @classmethod
    def lognormal_from_mean_cv(
        cls,
        name: str,
        mean: float,
        cv: float,
        *,
        units: Unit = DIMENSIONLESS,
        source: str = "unspecified",
        validity: tuple[float, float] | None = None,
    ) -> "Prior":
        """Lognormal parameterised by arithmetic mean and coefficient of variation.

        Convenient for tissue properties, which are reported in the literature
        as ``mean +- spread`` on the natural (positive) scale.
        """
        if mean <= 0:
            raise ValueError("lognormal_from_mean_cv requires a positive mean")
        s2 = math.log1p(cv * cv)
        return cls(
            name=name,
            dist="lognormal",
            params=(math.log(mean) - 0.5 * s2, math.sqrt(s2)),
            units=units,
            source=source,
            validity=validity,
        )


PriorMap = Mapping[str, Prior]


# --------------------------------------------------------------------------
# supports
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PSF:
    """Point-spread / lead-field / integration kernel of a datum.

    thesis Sec. 2.8: an EEG electrode's spatial support **is** its lead field,
    not a scalp label.  ``kind="leadfield"`` therefore carries the actual
    ``matrix`` (n_sensors x n_sources) and the ``source_frame`` it is defined
    in.  ``kind="none"`` is only admissible with an explicit reason.
    """

    kind: Literal[
        "leadfield",
        "gaussian",
        "voxel_box",
        "photon_path",
        "temporal_kernel",
        "aggregation",
        "none",
    ]
    frame: str
    units: Unit = DIMENSIONLESS
    matrix: torch.Tensor | None = None
    fwhm: tuple[float, ...] | None = None
    extent: tuple[float, ...] | None = None
    source_positions: torch.Tensor | None = None
    reason: str | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_unit(self.units)
        if self.kind == "leadfield" and self.matrix is None:
            raise ObservationRefusal(
                code="R01",
                message="PSF(kind='leadfield') without a lead-field matrix",
                remedy="supply the forward operator; a sensor label is not a support",
                offending_object=self,
            )
        if self.kind == "none" and not self.reason:
            raise ObservationRefusal(
                code="R01",
                message="PSF(kind='none') without a documented reason",
                remedy="state why this datum has no spatial extent, or supply a kernel",
                offending_object=self,
            )

    @property
    def shape(self) -> tuple[int, ...] | None:
        return None if self.matrix is None else tuple(self.matrix.shape)


@dataclass(frozen=True)
class Support:
    """Physical support of a datum.  Never a bare coordinate."""

    kind: Literal[
        "voxel",
        "surface_vertex",
        "parcel",
        "sensor",
        "optode_channel",
        "mesh",
        "field",
        "event",
        "band",
        "trial",
    ]
    frame: str
    units: Unit
    psf: PSF | None = None
    extent: tuple[float, ...] | None = None
    n_elements: int | None = None
    labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        validate_unit(self.units)
        if self.labels is not None and self.n_elements is not None:
            if len(self.labels) != self.n_elements:
                raise ValueError("Support: labels/n_elements mismatch")


@dataclass(frozen=True)
class TemporalSupport:
    """Native clock of a datum.  ``dt`` is *never* rewritten by a consumer."""

    clock: str
    dt: float
    integration_window: float = 0.0
    group_delay: float = 0.0
    jitter_sd: float = 0.0
    onset: float = 0.0

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError("TemporalSupport.dt must be positive seconds")
        if self.integration_window < 0:
            raise ValueError("negative integration window")

    @property
    def rate_hz(self) -> float:
        return 1.0 / self.dt

    def sample_times(self, n: int) -> torch.Tensor:
        """Sample instants on *this* clock; the only legitimate time axis."""
        return self.onset + self.dt * torch.arange(n, dtype=torch.float64)


# --------------------------------------------------------------------------
# the ledger
# --------------------------------------------------------------------------

UNKNOWN = "unknown"
"""Sentinel for a variance component that is explicitly not identified.

``None`` is forbidden as a *silent* default: a component is either a number or
the string ``"unknown"``, and :meth:`VarianceDecomposition.is_complete` only
passes when every one of the six thesis Sec. 2.7 components has been addressed.
"""

_VARIANCE_COMPONENTS: tuple[str, ...] = (
    "measurement",
    "within_session",
    "between_session",
    "parameter_posterior",
    "model_class",
    "numerical",
)


@dataclass(frozen=True)
class VarianceDecomposition:
    """The six variance components of thesis Sec. 2.7, kept separate forever.

    "The variance ledger separates, where identifiable, measurement noise,
    within-session stochasticity, between-session biological variation,
    parameter-posterior variance, model-class disagreement, and numerical or
    Monte Carlo error."

    Each field is a non-negative variance in ``units**2`` or the string
    ``"unknown"``.  There is no total-variance shortcut that hides which
    components were never estimated: :attr:`total` refuses if any component is
    unknown, and :attr:`known_total` makes the partial sum explicit.
    """

    measurement: float | str = UNKNOWN
    within_session: float | str = UNKNOWN
    between_session: float | str = UNKNOWN
    parameter_posterior: float | str = UNKNOWN
    model_class: float | str = UNKNOWN
    numerical: float | str = UNKNOWN
    units: Unit = DIMENSIONLESS

    def __post_init__(self) -> None:
        validate_unit(self.units)
        for name in _VARIANCE_COMPONENTS:
            v = getattr(self, name)
            if isinstance(v, str):
                if v != UNKNOWN:
                    raise ValueError(
                        f"variance component {name!r}: string value must be "
                        f"{UNKNOWN!r}, got {v!r}"
                    )
            elif not (float(v) >= 0.0 and math.isfinite(float(v))):
                raise ValueError(
                    f"variance component {name!r} must be a finite non-negative "
                    f"variance, got {v!r}"
                )

    def as_dict(self) -> dict[str, float | str]:
        return {n: getattr(self, n) for n in _VARIANCE_COMPONENTS}

    @property
    def unknown_components(self) -> tuple[str, ...]:
        return tuple(n for n in _VARIANCE_COMPONENTS if getattr(self, n) == UNKNOWN)

    def is_complete(self) -> bool:
        """Every component is either a number or explicitly ``"unknown"``.

        Construction already enforces this, so the method exists so that tests
        and downstream code can assert the *contract* rather than the
        implementation.
        """
        return all(
            isinstance(getattr(self, n), (int, float)) or getattr(self, n) == UNKNOWN
            for n in _VARIANCE_COMPONENTS
        )

    def is_fully_quantified(self) -> bool:
        return len(self.unknown_components) == 0

    @property
    def known_total(self) -> float:
        return float(
            sum(
                float(getattr(self, n))
                for n in _VARIANCE_COMPONENTS
                if getattr(self, n) != UNKNOWN
            )
        )

    @property
    def total(self) -> float:
        if not self.is_fully_quantified():
            raise ObservationRefusal(
                code="R08",
                message=(
                    "total variance requested while components "
                    f"{self.unknown_components} are unknown; a partial sum must "
                    "not be advertised as a total"
                ),
                remedy="use .known_total and report .unknown_components alongside it",
                offending_object=self,
            )
        return self.known_total

    def scaled(self, gain: float) -> "VarianceDecomposition":
        """Propagate through a scalar gain (variance scales as ``gain**2``)."""
        g2 = float(gain) ** 2
        vals = {
            n: (UNKNOWN if getattr(self, n) == UNKNOWN else float(getattr(self, n)) * g2)
            for n in _VARIANCE_COMPONENTS
        }
        return replace(self, **vals)

    def __add__(self, other: "VarianceDecomposition") -> "VarianceDecomposition":
        """Component-wise sum; unknown + anything stays unknown (not silently 0)."""
        if self.units != other.units:
            raise ObservationRefusal(
                code="R01",
                message=f"variance unit mismatch {self.units!r} vs {other.units!r}",
                remedy="convert to a common unit before combining ledgers",
            )
        vals = {}
        for n in _VARIANCE_COMPONENTS:
            a, b = getattr(self, n), getattr(other, n)
            vals[n] = UNKNOWN if (a == UNKNOWN or b == UNKNOWN) else float(a) + float(b)
        return VarianceDecomposition(units=self.units, **vals)


BiasStatus = Literal[
    "design_estimable", "externally_bounded", "prior_specified_sensitivity"
]


@dataclass(frozen=True)
class BiasTerm:
    """One systematic term of ``b_s`` (thesis Sec. 2.7) with a mandatory status.

    Refusal **R08** is enforced in :meth:`__post_init__`: a *point* estimate
    (``interval[0] == interval[1]``) is admissible only when the status is
    ``design_estimable`` **and** an ``estimator`` is named, or
    ``externally_bounded`` **and** an ``external_bound`` is named.  A
    ``prior_specified_sensitivity`` term may never be a point --- it is swept
    over a declared range.
    """

    name: str
    interval: tuple[float, float]
    status: BiasStatus
    units: Unit = DIMENSIONLESS
    estimator: str | None = None
    external_bound: str | None = None
    sensitivity_grid: tuple[float, ...] | None = None
    note: str = ""

    def __post_init__(self) -> None:
        validate_unit(self.units)
        lo, hi = self.interval
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise ValueError(f"bias {self.name!r}: non-finite interval")
        if lo > hi:
            raise ValueError(f"bias {self.name!r}: inverted interval {self.interval}")

        is_point = hi - lo <= 0.0

        if self.status == "design_estimable":
            if not self.estimator:
                raise RefusalR08(
                    f"bias term {self.name!r} declared design_estimable without "
                    "naming the estimator (which replication, randomization, "
                    "intervention, anchor, or hierarchy identifies it?)",
                    offending_object=self,
                )
        elif self.status == "externally_bounded":
            if not self.external_bound:
                raise RefusalR08(
                    f"bias term {self.name!r} declared externally_bounded without "
                    "naming the phantom, calibration target, independent "
                    "instrument, or negative control that supplies the interval",
                    offending_object=self,
                )
        elif self.status == "prior_specified_sensitivity":
            if is_point:
                raise RefusalR08(
                    f"bias term {self.name!r} is a point estimate "
                    f"({lo!r}) with status prior_specified_sensitivity; neither "
                    "an estimator nor an external bound exists, so it must be "
                    "propagated as a range",
                    offending_object=self,
                )
        else:  # pragma: no cover - Literal guards this in typed call sites
            raise ValueError(f"unknown bias status {self.status!r}")

        if is_point and not (self.estimator or self.external_bound):
            raise RefusalR08(
                f"bias term {self.name!r} assigned the point estimate {lo!r} "
                "without an estimator or an external bound",
                offending_object=self,
            )

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.interval[0] + self.interval[1])

    @property
    def half_width(self) -> float:
        return 0.5 * (self.interval[1] - self.interval[0])


@dataclass(frozen=True)
class Provenance:
    """Where a read came from and what it is allowed to claim."""

    operator: str
    version: str = "0.1.0"
    frames: tuple[str, ...] = ()
    clocks: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    seed: int | None = None
    device: str = "cpu"
    dtype: str = "float32"
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UncertaintyLedger:
    """Bias and variance, side by side, never collapsed into one score."""

    variance: VarianceDecomposition
    bias: tuple[BiasTerm, ...] = ()
    model_discrepancy: float | str = UNKNOWN
    model_discrepancy_flag: bool = False
    validity_domain: Mapping[str, Any] = field(default_factory=dict)
    provenance: Provenance | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.model_discrepancy, str) and self.model_discrepancy != UNKNOWN:
            raise ValueError("model_discrepancy string must be 'unknown'")

    @property
    def bias_interval(self) -> tuple[float, float]:
        """Additive envelope of all bias terms (worst-case, not a point)."""
        if not self.bias:
            return (0.0, 0.0)
        lo = sum(b.interval[0] for b in self.bias)
        hi = sum(b.interval[1] for b in self.bias)
        return (float(lo), float(hi))

    @property
    def bias_status(self) -> BiasStatus:
        """Weakest status across terms: the ledger is only as strong as its worst."""
        order: dict[str, int] = {
            "design_estimable": 0,
            "externally_bounded": 1,
            "prior_specified_sensitivity": 2,
        }
        if not self.bias:
            return "design_estimable"
        return max((b.status for b in self.bias), key=lambda s: order[s])  # type: ignore[return-value]

    def bias_by_name(self, name: str) -> BiasTerm | None:
        for b in self.bias:
            if b.name == name:
                return b
        return None

    def is_complete(self) -> bool:
        """Ledger completeness gate used by ``tests/observe/test_ledger.py``."""
        return (
            self.variance.is_complete()
            and self.provenance is not None
            and bool(self.validity_domain)
        )

    def with_bias(self, *terms: BiasTerm) -> "UncertaintyLedger":
        return replace(self, bias=self.bias + tuple(terms))

    # -- agent A's contract type -------------------------------------------
    def to_schema(self, units: Unit | None = None) -> Any:
        """R08-compliant ``scwbd.schema.UncertaintyLedger`` for one unit group."""
        return to_schema_ledger(self, units=units)

    def to_schema_all(self) -> dict[str, Any]:
        """One ``scwbd.schema.UncertaintyLedger`` per unit group."""
        return to_schema_ledgers(self)


# --------------------------------------------------------------------------
# reads and the operator interface
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ObservationRead:
    """A prediction **plus** its ledger.  There is no bare-tensor read path."""

    prediction: torch.Tensor
    units: Unit
    support: Support
    temporal: TemporalSupport
    ledger: UncertaintyLedger
    components: Mapping[str, torch.Tensor] = field(default_factory=dict)
    residual_channels: Mapping[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_unit(self.units)
        if not isinstance(self.prediction, torch.Tensor):
            raise TypeError("ObservationRead.prediction must be a torch.Tensor")
        if self.ledger.provenance is None:
            raise ObservationRefusal(
                code="R01",
                message="ObservationRead without provenance",
                remedy="attach Provenance(operator=..., frames=..., clocks=...)",
                offending_object=self,
            )

    @property
    def n_samples(self) -> int:
        return int(self.prediction.shape[-1])

    def times(self) -> torch.Tensor:
        return self.temporal.sample_times(self.n_samples)

    def sd(self) -> float:
        """Standard deviation from the *known* variance components only."""
        return math.sqrt(self.ledger.variance.known_total)

    # -- agent A's contract types ------------------------------------------
    def to_schema_ledger(self, units: Unit | None = None) -> Any:
        return to_schema_ledger(self.ledger, units=units)

    def to_schema_support(self) -> Any:
        return to_schema_support(self.support)

    def to_schema_temporal(self) -> Any:
        return to_schema_temporal(self.temporal)


class ObservationOperator(ABC):
    """``O_m`` of thesis Sec. 2.4.

    Contract for every subclass:

    1. :attr:`support` and :attr:`temporal` describe the **native** support and
       clock.  An operator never resamples its input onto another modality's
       clock; it evaluates the latent trajectory on the latent's own time axis
       and reports its own sample instants.
    2. :meth:`observe` returns an :class:`ObservationRead` --- prediction plus
       ledger --- or an :class:`Unresolved`.
    3. :attr:`nuisance_priors` exposes ``psi_m`` as priors, never constants.
    """

    name: str = "observation_operator"
    version: str = "0.1.0"

    # -- native descriptors -------------------------------------------------
    @property
    @abstractmethod
    def support(self) -> Support: ...

    @property
    @abstractmethod
    def temporal(self) -> TemporalSupport: ...

    @property
    @abstractmethod
    def units(self) -> Unit: ...

    @property
    def nuisance_priors(self) -> dict[str, Prior]:
        return {}

    # -- the read -----------------------------------------------------------
    @abstractmethod
    def observe(self, *args: Any, **kwargs: Any) -> ObservationRead | Unresolved: ...

    # -- helpers shared by every head --------------------------------------
    def _provenance(self, **extra: Any) -> Provenance:
        return Provenance(
            operator=self.name,
            version=self.version,
            frames=(self.support.frame,),
            clocks=(self.temporal.clock,),
            **extra,
        )

    def sample_latent_on_native_clock(
        self,
        latent: torch.Tensor,
        latent_temporal: TemporalSupport,
        n_out: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select the latent samples that fall on *this* operator's sample grid.

        This is decimation-by-selection, not interpolation.  It exists so that a
        1 ms EEG head and a 1 s BOLD head can consume the *same* latent
        trajectory without either being resampled onto the other's clock
        (thesis Sec. 2.6, Sec. 7.1).  When the native clock is not an integer
        multiple of the latent clock the method refuses rather than
        interpolating: the caller must integrate through an explicit kernel.
        """
        ratio = self.temporal.dt / latent_temporal.dt
        k = round(ratio)
        if k < 1 or abs(ratio - k) > 1e-9 * max(1.0, ratio):
            raise ObservationRefusal(
                code="R01",
                message=(
                    f"{self.name}: native dt {self.temporal.dt}s is not an integer "
                    f"multiple of the latent dt {latent_temporal.dt}s "
                    f"(ratio {ratio:.6f}); implicit interpolation is forbidden"
                ),
                remedy="integrate the latent through a declared kernel, or "
                "declare a clock-graph edge with its own uncertainty",
                offending_object=(self.temporal, latent_temporal),
            )
        n_latent = int(latent.shape[-1])
        offset = int(round((self.temporal.onset - latent_temporal.onset) / latent_temporal.dt))
        max_n = 1 + (n_latent - 1 - offset) // k if n_latent > offset else 0
        n = max_n if n_out is None else min(n_out, max_n)
        idx = offset + k * torch.arange(n, device=latent.device)
        return latent.index_select(-1, idx), idx


# --------------------------------------------------------------------------
# thin adapters to scwbd.schema (agent A) -- optional, never imported eagerly
# --------------------------------------------------------------------------


def _schema_module() -> Any | None:
    try:  # pragma: no cover - depends on a parallel agent's progress
        import scwbd.schema as _s  # type: ignore

        return _s
    except Exception:
        return None


_PSF_KIND_TO_SCHEMA: dict[str, str] = {
    "leadfield": "lead_field",
    "gaussian": "gaussian",
    "voxel_box": "integration_kernel",
    "photon_path": "empirical",
    "temporal_kernel": "integration_kernel",
    "aggregation": "integration_kernel",
    "none": "point",
}

#: Local support kinds that the schema spells differently.  A trial is an event
#: window and an optode channel is a (paired) sensor; neither is a new physical
#: category, so they are renamed rather than pushed into the schema.
_SUPPORT_KIND_TO_SCHEMA: dict[str, str] = {
    "optode_channel": "sensor",
    "trial": "event",
}

#: ``VarianceDecomposition`` field -> ``scwbd.schema.ledger.VARIANCE_COMPONENTS``.
_VARIANCE_KEY_TO_SCHEMA: dict[str, str] = {
    "measurement": "measurement",
    "within_session": "within_session",
    "between_session": "between_session",
    "parameter_posterior": "parameter",
    "model_class": "model_class",
    "numerical": "numerical",
}


def _require_schema(what: str) -> Any:
    mod = _schema_module()
    if mod is None:
        raise ObservationRefusal(
            code="R01",
            message=f"scwbd.schema is not importable, so {what} cannot be emitted",
            remedy="install/repair the schema package; observation heads must not "
            "invent their own contract types on a live path",
        )
    return mod


def to_schema_psf(psf: PSF, *, kernel_ref: str | None = None) -> Any:
    """Convert to ``scwbd.schema.PSF``.

    A lead field is stored by reference: the matrix itself is agent F's asset,
    while the schema records *that* a kernel exists, its output units, and where
    to find it.  ``nominal`` is set only when the kernel really is a placeholder.
    """
    s = _require_schema("a PSF")
    kind = _PSF_KIND_TO_SCHEMA[psf.kind]
    ref = kernel_ref
    if ref is None and psf.matrix is not None:
        ref = f"observe:{psf.kind}:{tuple(psf.matrix.shape)}"
    return s.PSF(
        kind=kind,
        fwhm=psf.fwhm,
        units=psf.units,
        extent_units="m",
        kernel_ref=ref,
        nominal=psf.kind == "none",
    )


def to_schema_support(support: Support) -> Any:
    """Convert to ``scwbd.schema.Support``, carrying the PSF rather than a label."""
    s = _require_schema("a Support")
    return s.Support(
        kind=_SUPPORT_KIND_TO_SCHEMA.get(support.kind, support.kind),
        frame=support.frame,
        units=support.units,
        psf=to_schema_psf(support.psf) if support.psf is not None else None,
        extent=support.extent,
        n_elements=support.n_elements,
        label=support.kind,
    )


def to_schema_temporal(temporal: TemporalSupport) -> Any:
    s = _require_schema("a TemporalSupport")
    return s.TemporalSupport(
        clock=temporal.clock,
        dt=temporal.dt,
        integration_window=temporal.integration_window,
        group_delay=temporal.group_delay,
        jitter_sd=temporal.jitter_sd,
    )


def to_schema_ledger(ledger: "UncertaintyLedger", units: Unit | None = None) -> Any:
    """Emit an R08-compliant ``scwbd.schema.UncertaintyLedger``.

    The schema ledger carries **one** bias interval with **one** status, so this
    adapter selects the bias terms expressed in ``units`` (default: the units of
    the ledger's own variance), takes their additive envelope, and adopts the
    *weakest* status among them --- then supplies the evidence that status
    requires:

    * ``design_estimable``   -> ``bias_estimator`` names every estimator;
    * ``externally_bounded`` -> ``external_bound_source`` names every bound;
    * ``prior_specified_sensitivity`` -> the interval is a non-degenerate range.

    Unknown variance components are **omitted** from the schema dict rather than
    written as zero, and the omission is recorded in ``notes``.  Use
    :func:`to_schema_ledgers` to get one ledger per unit group.
    """
    s = _require_schema("an UncertaintyLedger")
    if units is None:
        units = _sqrt_units(ledger.variance.units)
    terms = [b for b in ledger.bias if b.units == units]
    if not terms:
        terms = [b for b in ledger.bias if b.units == DIMENSIONLESS]
        if terms:
            units = DIMENSIONLESS

    order = {
        "design_estimable": 0,
        "externally_bounded": 1,
        "prior_specified_sensitivity": 2,
    }
    if terms:
        status = max((t.status for t in terms), key=lambda x: order[x])
        lo = float(sum(t.interval[0] for t in terms))
        hi = float(sum(t.interval[1] for t in terms))
    else:
        # no bias term in these units: say so as a swept unknown, never as zero
        status = "prior_specified_sensitivity"
        lo, hi = -0.0, 0.0

    estimators = "; ".join(
        t.estimator for t in terms if t.status == "design_estimable" and t.estimator
    )
    bounds = "; ".join(
        t.external_bound
        for t in terms
        if t.status == "externally_bounded" and t.external_bound
    )

    if status == "prior_specified_sensitivity" and hi <= lo:
        raise RefusalR08(
            "schema ledger would carry a degenerate prior-specified bias "
            f"interval ({lo}, {hi}) in units {units!r}: a term with neither an "
            "estimator nor an external bound must be propagated as a range",
            offending_object=ledger,
        )

    variance: dict[str, float] = {}
    unknown: list[str] = []
    for local, schema_key in _VARIANCE_KEY_TO_SCHEMA.items():
        v = getattr(ledger.variance, local)
        if v == UNKNOWN:
            unknown.append(schema_key)
        else:
            variance[schema_key] = float(v)

    notes = "; ".join(ledger.notes)
    if unknown:
        notes = (
            f"variance components not identified by this acquisition: "
            f"{', '.join(unknown)} (omitted, not zero). " + notes
        )

    domain = {k: _jsonable(v) for k, v in ledger.validity_domain.items()}
    if ledger.provenance is not None:
        domain["provenance_operator"] = ledger.provenance.operator
        domain["provenance_version"] = ledger.provenance.version
        domain["provenance_references"] = list(ledger.provenance.references)
        domain["provenance_seed"] = ledger.provenance.seed
    domain["bias_terms"] = [
        {
            "name": t.name,
            "status": t.status,
            "interval": list(t.interval),
            "units": t.units,
        }
        for t in ledger.bias
    ]

    return s.UncertaintyLedger(
        variance=variance,
        bias_interval=(lo, hi),
        bias_status=status,
        model_discrepancy=(
            None
            if ledger.model_discrepancy == UNKNOWN
            else float(ledger.model_discrepancy)
        ),
        validity_domain=domain,
        bias_estimator=estimators or None,
        external_bound_source=bounds or None,
        units=units,
        notes=notes.strip(),
    )


def to_schema_ledgers(ledger: "UncertaintyLedger") -> dict[str, Any]:
    """One schema ledger per unit group: bias terms in metres, seconds and volts
    are different physical claims and must never be summed into one interval."""
    groups = sorted({b.units for b in ledger.bias})
    if not groups:
        groups = [_sqrt_units(ledger.variance.units)]
    return {u: to_schema_ledger(ledger, units=u) for u in groups}


def _sqrt_units(variance_units: Unit) -> Unit:
    """``V^2 -> V``; used to pick the bias group matching a variance ledger."""
    mapping = {
        "V^2": "V",
        "T^2": "T",
        "m^2": "m",
        "s^2": "s",
        "dimensionless": DIMENSIONLESS,
        "%": PERCENT,
    }
    return mapping.get(variance_units, DIMENSIONLESS)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)
