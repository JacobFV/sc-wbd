"""Schema bindings and ledger constructors for :mod:`scwbd.anatomy`.

Agent A owns ``scwbd.schema``.  This module re-exports the contract types this
package consumes and adds the *anatomy-specific* ledger constructors that make
refusal R08 pass honestly rather than by accident.

R08 as agent A operationalized it
---------------------------------
``UncertaintyLedger.has_estimator()`` is what R08 tests:

``design_estimable``
    must name a ``bias_estimator``;
``externally_bounded``
    must name an ``external_bound_source``;
``prior_specified_sensitivity``
    must carry a *non-degenerate* ``bias_interval`` -- a swept range, never a
    point.

Essentially every object this package produces is a **group average from a
population the model will later claim to individualise**, so almost everything
here is ``prior_specified_sensitivity`` with an interval wide enough to be
honest.  The three constructors below make the intended reading explicit at the
call site; each one asserts ``has_estimator()`` before returning, so a ledger
that would fail R08 cannot leave this module.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scwbd.schema import (  # re-exported for the rest of scwbd.anatomy
    BetaPrior,
    DiracPrior,
    GammaPrior,
    LogNormalPrior,
    NormalPrior,
    Prior,
    PriorBase,
    UncertaintyLedger,
    UniformPrior,
    as_prior,
)

__all__ = [
    "Prior",
    "PriorBase",
    "NormalPrior",
    "LogNormalPrior",
    "UniformPrior",
    "BetaPrior",
    "GammaPrior",
    "DiracPrior",
    "as_prior",
    "UncertaintyLedger",
    "EvidenceClass",
    "MechanisticStatus",
    "EVIDENCE_ORDER",
    "evidence_rank",
    "group_average_ledger",
    "externally_bounded_ledger",
    "design_estimable_ledger",
    "prior_dict",
    "prior_from_dict",
]

EvidenceClass = Literal["hard", "soft", "proposed", "absent"]
MechanisticStatus = Literal["mechanistic", "effective", "functional", "surrogate"]

#: Ordering used for monotone comparisons ("at least soft support").
EVIDENCE_ORDER: tuple[str, ...] = ("absent", "proposed", "soft", "hard")


def evidence_rank(cls: str) -> int:
    """Integer rank of an evidence class; higher means better supported."""
    return EVIDENCE_ORDER.index(cls)


# ---------------------------------------------------------------------------
# ledger constructors
# ---------------------------------------------------------------------------
def group_average_ledger(
    *,
    units: str,
    bias_interval: tuple[float, float],
    variance: dict[str, float],
    forbidden_inference: str,
    n_donors: int | None = None,
    validity_domain: dict[str, Any] | None = None,
    model_discrepancy: float | None = None,
    notes: str = "",
) -> UncertaintyLedger:
    """Ledger for a group-average atlas value.

    Use for every receptor, myelin, thickness, gradient, timescale and
    connectome quantity in this package.  ``bias_status`` is
    ``prior_specified_sensitivity`` because no acquisition here contains the
    replication or the external anchor that would identify the group-to-
    individual offset, and no phantom bounds it.

    Parameters
    ----------
    bias_interval
        The swept range over which downstream sensitivity analysis must run.
        Must be non-degenerate: a point here would assert that a group average
        equals an individual value, which is precisely what Appendix A forbids.
    forbidden_inference
        One sentence naming the inference this object does **not** license.
        Written into ``validity_domain["forbidden_inference"]`` so it travels
        with the number.
    n_donors
        Donor / subject count behind the average, when known.  Recorded so a
        consumer can see that "group average" sometimes means ``n = 3``.

    Raises
    ------
    ValueError
        If the interval is degenerate (R08 would refuse the object).
    """
    lo, hi = float(bias_interval[0]), float(bias_interval[1])
    if not hi > lo:
        raise ValueError(
            "a group-average ledger needs a non-degenerate bias_interval: a point "
            "estimate would assert that the atlas value equals a subject value "
            "(thesis Appendix A forbids this; refusal R08 would reject it)"
        )
    vd: dict[str, Any] = {
        "population": "group average",
        "forbidden_inference": forbidden_inference,
    }
    if n_donors is not None:
        vd["n_donors"] = int(n_donors)
    if validity_domain:
        vd.update(validity_domain)
    led = UncertaintyLedger(
        variance={k: float(v) for k, v in variance.items()},
        bias_interval=(lo, hi),
        bias_status="prior_specified_sensitivity",
        model_discrepancy=model_discrepancy,
        validity_domain=vd,
        units=units,
        notes=notes,
    )
    assert led.has_estimator(), "constructed ledger would fail R08"
    return led


def externally_bounded_ledger(
    *,
    units: str,
    bias_interval: tuple[float, float],
    external_bound_source: str,
    variance: dict[str, float],
    validity_domain: dict[str, Any] | None = None,
    notes: str = "",
) -> UncertaintyLedger:
    """Ledger whose bias is bounded by a named external instrument or phantom.

    In this package only geometric quantities qualify: mesh areas and Euclidean
    distances are bounded by the template's own voxel/vertex resolution, which
    is an external, checkable quantity rather than a modelling assumption.
    """
    if not external_bound_source:
        raise ValueError("externally_bounded requires a named external_bound_source (R08)")
    led = UncertaintyLedger(
        variance={k: float(v) for k, v in variance.items()},
        bias_interval=(float(bias_interval[0]), float(bias_interval[1])),
        bias_status="externally_bounded",
        external_bound_source=external_bound_source,
        validity_domain=dict(validity_domain or {}),
        units=units,
        notes=notes,
    )
    assert led.has_estimator(), "constructed ledger would fail R08"
    return led


def design_estimable_ledger(
    *,
    units: str,
    bias_interval: tuple[float, float],
    bias_estimator: str,
    variance: dict[str, float],
    validity_domain: dict[str, Any] | None = None,
    notes: str = "",
) -> UncertaintyLedger:
    """Ledger whose bias is identified by the acquisition design itself.

    Nothing in ``scwbd.anatomy`` currently qualifies -- there is no anatomical
    acquisition here with the replication or randomization that would identify
    a group-to-individual offset.  The constructor exists so that a future
    source which *does* qualify has a typed way to say so instead of quietly
    reusing the group-average constructor.
    """
    if not bias_estimator:
        raise ValueError("design_estimable requires a named bias_estimator (R08)")
    led = UncertaintyLedger(
        variance={k: float(v) for k, v in variance.items()},
        bias_interval=(float(bias_interval[0]), float(bias_interval[1])),
        bias_status="design_estimable",
        bias_estimator=bias_estimator,
        validity_domain=dict(validity_domain or {}),
        units=units,
        notes=notes,
    )
    assert led.has_estimator(), "constructed ledger would fail R08"
    return led


# ---------------------------------------------------------------------------
# prior (de)serialisation for npz caches
# ---------------------------------------------------------------------------
def prior_dict(p: PriorBase) -> dict[str, Any]:
    """JSON-round-trippable form of a schema prior."""
    return p.model_dump(mode="json")


def prior_from_dict(d: dict[str, Any]) -> PriorBase:
    return as_prior(d)


def prior_quantile(p: PriorBase, q: float, *, seed: int = 0, n: int = 200_000) -> float:
    """Quantile of a schema prior.

    Closed form where the family provides one, Monte Carlo otherwise.  The
    Monte Carlo path takes an explicit seed so the result is reproducible
    (ARCHITECTURE.md §3).
    """
    import math

    from scipy import stats

    if isinstance(p, DiracPrior):
        return float(p.value)
    if isinstance(p, NormalPrior):
        return float(stats.norm.ppf(q, p.loc, p.scale))
    if isinstance(p, LogNormalPrior):
        return float(math.exp(stats.norm.ppf(q, p.mu, p.sigma)))
    if isinstance(p, UniformPrior):
        return float(p.low + q * (p.high - p.low))
    if isinstance(p, BetaPrior):
        return float(stats.beta.ppf(q, p.alpha, p.beta))
    if isinstance(p, GammaPrior):
        return float(stats.gamma.ppf(q, p.shape, scale=1.0 / p.rate))
    return float(np.quantile(np.asarray(p.sample(seed, n)), q))


__all__.append("prior_quantile")
