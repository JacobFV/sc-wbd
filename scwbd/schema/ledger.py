"""The bias-variance ledger (thesis sec. 2.7, ARCHITECTURE.md sec. 2).

"Uncertainty is not represented by one confidence score."  Every source, field,
parameter, edge, solver, cross-scale map, observation operator and intervention
prediction carries this object, and **bias and variance never collapse into one
score**.

The three ``bias_status`` values are not decoration; they are what refusal R08
tests.  A term is:

* ``design_estimable`` only when the acquisition contains the replication,
  randomization, intervention, anchor, or hierarchy that identifies it - so it
  must name the estimator;
* ``externally_bounded`` when a phantom, calibration target, independent
  instrument, or negative control supplies the interval - so it must name that
  external source;
* ``prior_specified_sensitivity`` when neither is available - so it must be
  carried as a *range*, never as a point.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue, model_validator

from .base import SchemaModel

__all__ = ["UncertaintyLedger", "BiasStatus", "VARIANCE_COMPONENTS"]

BiasStatus = Literal[
    "design_estimable",
    "externally_bounded",
    "prior_specified_sensitivity",
]

#: The variance components thesis sec. 2.7 asks to be separated "where
#: identifiable".  Keys outside this set are allowed but flagged in provenance.
VARIANCE_COMPONENTS: tuple[str, ...] = (
    "measurement",
    "within_session",
    "between_session",
    "parameter",
    "model_class",
    "numerical",
)


class UncertaintyLedger(SchemaModel):
    """Per thesis sec. 2.7. bias and variance NEVER collapse into one score."""

    #: measurement / within_session / between_session / parameter /
    #: model_class / numerical -> variance (squared units of the object).
    variance: dict[str, float] = Field(default_factory=dict)
    #: Closed interval for the systematic term. A degenerate interval asserts a
    #: point estimate and is only admissible with an estimator or bound (R08).
    bias_interval: tuple[float, float] = (0.0, 0.0)
    bias_status: BiasStatus = "prior_specified_sensitivity"
    model_discrepancy: float | None = None
    validity_domain: dict[str, JsonValue] = Field(default_factory=dict)

    # -- R08 evidence ------------------------------------------------------
    #: Named estimator for a design-estimable bias (e.g. "phantom_anchor_glm").
    bias_estimator: str | None = None
    #: Named external bound for an externally-bounded bias (phantom, negative
    #: control, independent instrument).
    external_bound_source: str | None = None
    #: Units of the bias interval and of sqrt(variance).
    units: str = "dimensionless"
    notes: str = ""

    @model_validator(mode="after")
    def _check_shape(self) -> "UncertaintyLedger":
        lo, hi = self.bias_interval
        if hi < lo:
            raise ValueError(f"bias_interval must be ordered, got ({lo}, {hi})")
        for name, value in self.variance.items():
            if value < 0.0:
                raise ValueError(f"variance component {name!r} is negative: {value}")
        return self

    # -- derived views -----------------------------------------------------
    @property
    def is_bias_point_estimate(self) -> bool:
        lo, hi = self.bias_interval
        return lo == hi

    @property
    def total_variance(self) -> float:
        return float(sum(self.variance.values()))

    @property
    def bias_width(self) -> float:
        lo, hi = self.bias_interval
        return float(hi - lo)

    def has_estimator(self) -> bool:
        """True when the declared status is backed by the evidence it names."""
        if self.bias_status == "design_estimable":
            return bool(self.bias_estimator)
        if self.bias_status == "externally_bounded":
            return bool(self.external_bound_source)
        # prior_specified_sensitivity is backed by a swept *range*, not a point
        return not self.is_bias_point_estimate

    def unknown_variance_components(self) -> tuple[str, ...]:
        return tuple(k for k in self.variance if k not in VARIANCE_COMPONENTS)

    @classmethod
    def unknown(cls, units: str = "dimensionless") -> "UncertaintyLedger":
        """An explicitly-unknown ledger.

        Rule 1 of ARCHITECTURE.md sec. 7: an unknown field stays unknown. This
        constructor makes "we do not know" representable without pretending the
        bias is zero, and it will be refused by R08 if used on a live path.
        """
        return cls(bias_status="prior_specified_sensitivity", units=units)
