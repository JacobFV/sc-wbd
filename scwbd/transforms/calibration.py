"""Calibration records and the validity-interval policy (Appendix C layer 9).

    "Calibration observations, fitting method, residuals, validity interval,
     extrapolation distance and recalibration triggers ... Do not reuse past
     validity interval silently; inflate uncertainty or require recalibration
     when device, participant geometry or environment changes."

Two admissible behaviours outside the validity interval, and no third:

* :attr:`ExpiryPolicy.REFUSE`  -> :class:`CalibrationExpiredError`.
* :attr:`ExpiryPolicy.INFLATE` -> the path is still returned, but with a
  covariance inflation factor that grows with extrapolation distance, an
  ``inflated=True`` provenance flag, and the reason recorded.

There is deliberately no ``IGNORE``.  Silent reuse is the failure this layer
exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

import torch

from .errors import CalibrationExpiredError, TransformError
from .se3 import DTYPE, ValidityInterval


class ExpiryPolicy(str, Enum):
    REFUSE = "refuse"
    INFLATE = "inflate"

    @classmethod
    def coerce(cls, v: "ExpiryPolicy | str") -> "ExpiryPolicy":
        if isinstance(v, cls):
            return v
        try:
            return cls(str(v).lower())
        except ValueError:
            raise TransformError(
                f"unknown expiry policy {v!r}",
                remedy="Use ExpiryPolicy.REFUSE or ExpiryPolicy.INFLATE. "
                "Silently reusing an expired calibration is not an option.",
                offending_object=v,
            ) from None


@dataclass(frozen=True)
class CalibrationRecord:
    """Provenance of the measurement that produced a transform edge.

    Attributes
    ----------
    method:
        How the transform was fitted ("fiducial_lsq", "icp_surface",
        "phantom_grid", "hydrophone_scan", "cross_correlation", ...).
    n_observations:
        Number of calibration observations (fiducials, landmarks, targets).
    residual_rms / residual_max:
        Fit residuals in the edge's units.  Kept because a 3-fiducial fit with
        4 mm RMS is not the same object as a 40-landmark fit with 0.4 mm RMS,
        even when both produce a 4x4 matrix.
    validity:
        Interval during which the calibration may be used.
    inflation_time_constant:
        Seconds of extrapolation that double the covariance under
        :attr:`ExpiryPolicy.INFLATE`.  Must be positive when a finite validity
        interval is declared.
    recalibration_triggers:
        Declared events that void the calibration ("coil_remount",
        "participant_repositioned", "amplifier_restart", ...).
    """

    method: str = "undeclared"
    n_observations: int | None = None
    residual_rms: float | None = None
    residual_max: float | None = None
    validity: ValidityInterval = field(default_factory=ValidityInterval.unbounded)
    inflation_time_constant: float = 3600.0
    recalibration_triggers: tuple[str, ...] = ()
    fitted_at: float | None = None
    device_serial: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.inflation_time_constant <= 0:
            raise TransformError(
                "inflation_time_constant must be positive",
                remedy="Declare how fast this calibration decays.",
                offending_object=self.inflation_time_constant,
            )

    # ------------------------------------------------------------------
    def check(
        self,
        t: float,
        *,
        policy: "ExpiryPolicy | str" = ExpiryPolicy.REFUSE,
        label: str = "<edge>",
        triggers_fired: Sequence[str] = (),
    ) -> "ValidityCheck":
        """Evaluate the calibration at time ``t``.

        A fired recalibration trigger always refuses, regardless of policy: an
        interval cannot vouch for a device that was remounted inside it.
        """
        policy = ExpiryPolicy.coerce(policy)
        fired = [x for x in triggers_fired if x in self.recalibration_triggers]
        if fired:
            raise CalibrationExpiredError(
                f"calibration for {label} is void: recalibration trigger(s) "
                f"{fired} fired (method={self.method})",
                remedy=f"Recalibrate {label} before using this path.",
                offending_object=label,
            )
        d = self.validity.extrapolation_distance(t)
        if d == 0.0:
            return ValidityCheck(True, 0.0, 1.0, policy, label, "")
        reason = (
            f"calibration for {label} (method={self.method}, validity "
            f"{self.validity}) is being used {d:.6g} s outside its validity "
            f"interval at t={t:.6g}"
        )
        if policy is ExpiryPolicy.REFUSE:
            raise CalibrationExpiredError(
                reason,
                remedy=(
                    "Recalibrate, restrict the query to the validity interval, "
                    "or opt in to ExpiryPolicy.INFLATE and accept the inflated "
                    "ledger (which is recorded in provenance)."
                ),
                offending_object=label,
            )
        factor = 2.0 ** (d / self.inflation_time_constant)
        return ValidityCheck(False, d, factor, policy, label, reason)


@dataclass(frozen=True)
class ValidityCheck:
    """Outcome of a calibration validity check.  Never silently discarded."""

    inside: bool
    extrapolation_distance: float
    inflation_factor: float
    policy: ExpiryPolicy
    label: str
    reason: str

    def apply(self, cov: Any) -> torch.Tensor:
        """Inflate a covariance by ``inflation_factor`` (variance scaling)."""
        C = torch.as_tensor(cov, dtype=DTYPE)
        return C * self.inflation_factor

    def as_record(self) -> dict[str, Any]:
        return {
            "edge": self.label,
            "inside_validity": self.inside,
            "extrapolation_distance_s": self.extrapolation_distance,
            "covariance_inflation_factor": self.inflation_factor,
            "policy": self.policy.value,
            "reason": self.reason,
        }


__all__ = ["ExpiryPolicy", "CalibrationRecord", "ValidityCheck"]
