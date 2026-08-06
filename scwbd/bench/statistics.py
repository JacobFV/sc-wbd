"""Reporting discipline of ``body.tex`` §11.2 (agent J).

Everything a gate or ablation is allowed to report goes through here:

* sampling variance and bootstrap / posterior intervals
  (:func:`bootstrap_ci`, :func:`paired_bootstrap`, block variants for
  autocorrelated time series);
* calibration in the *intended deployment population*, never aggregate
  accuracy alone (:func:`calibration`, :class:`CalibrationReport`);
* **estimated optimism from model selection** (:func:`selection_optimism`) —
  the winner's curse incurred by picking the best of several candidates on the
  same data;
* bias analyses across session, device, site, anatomy, demographic strata and
  task context (:func:`stratified_bias`), and the *plausible systematic error*
  that every ablation must report next to its variance
  (:func:`systematic_error`);
* the "smoothed away the effect of interest" check (:func:`smoothing_check`),
  which is the executable form of §11.4's warning that a lower-variance model
  is not preferred when it achieves stability by attenuating the effect;
* plotting helpers so that every reported number ships with an interval.

No function here fits a brain model.  This module is measurement only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .report import Interval, Metric

__all__ = [
    "gaussian_log_score",
    "crps_gaussian",
    "bootstrap_ci",
    "paired_bootstrap",
    "PairedDiff",
    "pit_values",
    "CalibrationReport",
    "calibration",
    "selection_optimism",
    "SelectionOptimism",
    "StratumBias",
    "BiasAnalysis",
    "stratified_bias",
    "systematic_error",
    "SmoothingVerdict",
    "smoothing_check",
    "data_efficiency_curve",
    "decision_regret",
    "plot_metric_intervals",
    "plot_calibration",
    "plot_data_efficiency",
]

_EPS = 1e-12


# --------------------------------------------------------------------------
# scoring rules
# --------------------------------------------------------------------------
def gaussian_log_score(y: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Per-observation Gaussian log predictive density (higher is better).

    This is *the* headline metric for held-out likelihood: it is proper, so a
    model cannot buy accuracy with overconfidence.
    """
    y = np.asarray(y, dtype=float)
    mean = np.asarray(mean, dtype=float)
    sd = np.maximum(np.asarray(sd, dtype=float), _EPS)
    z = (y - mean) / sd
    return -0.5 * z**2 - np.log(sd) - 0.5 * math.log(2.0 * math.pi)


def crps_gaussian(y: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Continuous ranked probability score (lower is better)."""
    from scipy.stats import norm  # local import: scipy is optional at import time

    y = np.asarray(y, dtype=float)
    mean = np.asarray(mean, dtype=float)
    sd = np.maximum(np.asarray(sd, dtype=float), _EPS)
    z = (y - mean) / sd
    return sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / math.sqrt(math.pi))


# --------------------------------------------------------------------------
# resampling
# --------------------------------------------------------------------------
def _resample_index(n: int, rng: np.random.Generator, block: int | None) -> np.ndarray:
    if block is None or block <= 1:
        return rng.integers(0, n, size=n)
    n_blocks = int(math.ceil(n / block))
    starts = rng.integers(0, max(n - block + 1, 1), size=n_blocks)
    idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])
    return idx[:n]


def bootstrap_ci(
    x: Sequence[float] | np.ndarray,
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
    block: int | None = None,
) -> tuple[float, Interval]:
    """Point estimate + percentile bootstrap interval.

    ``block`` enables a moving-block bootstrap for autocorrelated series; use
    it for anything sampled on a neural or hemodynamic clock, where an iid
    bootstrap understates the interval.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("bootstrap_ci on empty sample")
    point = float(statistic(x))
    if x.size == 1:
        return point, Interval(point, point, level, "degenerate-n=1")
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        boots[b] = statistic(x[_resample_index(x.size, rng, block)])
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(boots, [a, 1.0 - a])
    method = "moving-block-bootstrap" if block else "bootstrap-percentile"
    return point, Interval(float(lo), float(hi), level, method)


@dataclass(frozen=True)
class PairedDiff:
    """Paired difference (candidate - baseline) with its interval."""

    name: str
    mean: float
    interval: Interval
    p_two_sided: float
    n: int

    @property
    def significant_win(self) -> bool:
        return self.interval.lo > 0.0

    @property
    def significant_loss(self) -> bool:
        return self.interval.hi < 0.0

    @property
    def indistinguishable(self) -> bool:
        return not (self.significant_win or self.significant_loss)

    def metric(self, *, kind: str = "accuracy", threshold: float = 0.0,
               units: str = "nats/obs") -> Metric:
        return Metric(
            name=self.name,
            value=self.mean,
            units=units,
            kind=kind,  # type: ignore[arg-type]
            interval=self.interval,
            threshold=threshold,
            direction="greater_is_better",
            require_interval_beats_threshold=True,
            note=f"paired bootstrap over n={self.n}, two-sided p~{self.p_two_sided:.3g}",
        )


def paired_bootstrap(
    candidate: Sequence[float] | np.ndarray,
    baseline: Sequence[float] | np.ndarray,
    *,
    name: str = "delta",
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
    block: int | None = None,
) -> PairedDiff:
    """Paired bootstrap of ``mean(candidate - baseline)``.

    Pairing is mandatory: unpaired comparisons of two models scored on the
    same held-out points throw away the correlation that makes the comparison
    informative, and usually flatter the noisier model.
    """
    a = np.asarray(candidate, dtype=float).ravel()
    b = np.asarray(baseline, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"paired_bootstrap needs matched shapes, got {a.shape} vs {b.shape}")
    d = a - b
    point, iv = bootstrap_ci(d, n_boot=n_boot, level=level, seed=seed, block=block)
    # two-sided bootstrap p-value: how often the resampled mean crosses zero
    rng = np.random.default_rng(seed + 1)
    boots = np.array([np.mean(d[_resample_index(d.size, rng, block)]) for _ in range(n_boot)])
    frac_le0 = float(np.mean(boots <= 0.0))
    p = 2.0 * min(frac_le0, 1.0 - frac_le0)
    return PairedDiff(name=name, mean=point, interval=iv, p_two_sided=min(p, 1.0), n=int(d.size))


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------
def pit_values(y: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Probability integral transform of Gaussian predictions."""
    from scipy.stats import norm

    sd = np.maximum(np.asarray(sd, dtype=float), _EPS)
    return norm.cdf((np.asarray(y, dtype=float) - np.asarray(mean, dtype=float)) / sd)


@dataclass
class CalibrationReport:
    nominal: np.ndarray
    empirical: np.ndarray
    #: mean |empirical - nominal| coverage error
    coverage_error: float
    coverage_error_interval: Interval
    #: positive part of (nominal - empirical): intervals too narrow
    overconfidence: float
    overconfidence_interval: Interval
    #: mean predictive sd (sharpness); reported so that "well calibrated
    #: because uninformative" is visible rather than hidden
    sharpness: float
    n: int

    def metrics(self, prefix: str = "") -> list[Metric]:
        p = f"{prefix}." if prefix else ""
        return [
            Metric(
                name=f"{p}coverage_error",
                value=self.coverage_error,
                kind="calibration",
                interval=self.coverage_error_interval,
                direction="less_is_better",
            ),
            Metric(
                name=f"{p}overconfidence",
                value=self.overconfidence,
                kind="calibration",
                interval=self.overconfidence_interval,
                direction="less_is_better",
                note="positive = predictive intervals too narrow",
            ),
            Metric(
                name=f"{p}sharpness_mean_sd",
                value=self.sharpness,
                kind="diagnostic",
                exact=True,
                note="reported so calibration-by-vagueness is visible",
            ),
        ]


_DEFAULT_LEVELS = np.array([0.5, 0.68, 0.8, 0.9, 0.95, 0.99])


def calibration(
    y: np.ndarray,
    mean: np.ndarray,
    sd: np.ndarray,
    *,
    levels: Sequence[float] | np.ndarray = _DEFAULT_LEVELS,
    seed: int = 0,
    n_boot: int = 1000,
) -> CalibrationReport:
    """Central-interval coverage calibration for Gaussian predictions."""
    from scipy.stats import norm

    y = np.asarray(y, dtype=float).ravel()
    mean = np.asarray(mean, dtype=float).ravel()
    sd = np.maximum(np.asarray(sd, dtype=float).ravel(), _EPS)
    levels = np.asarray(levels, dtype=float)
    z = np.abs((y - mean) / sd)
    emp = np.array([float(np.mean(z <= norm.ppf(0.5 + lv / 2.0))) for lv in levels])
    err = np.abs(emp - levels)
    over = np.maximum(levels - emp, 0.0)

    # bootstrap over observations, recomputing the whole coverage curve
    rng = np.random.default_rng(seed)
    e_boot = np.empty(n_boot)
    o_boot = np.empty(n_boot)
    crit = norm.ppf(0.5 + levels / 2.0)
    for b in range(n_boot):
        idx = rng.integers(0, z.size, size=z.size)
        zb = z[idx]
        empb = np.array([float(np.mean(zb <= c)) for c in crit])
        e_boot[b] = float(np.mean(np.abs(empb - levels)))
        o_boot[b] = float(np.mean(np.maximum(levels - empb, 0.0)))
    q = [0.025, 0.975]
    e_lo, e_hi = np.quantile(e_boot, q)
    o_lo, o_hi = np.quantile(o_boot, q)
    return CalibrationReport(
        nominal=levels,
        empirical=emp,
        coverage_error=float(np.mean(err)),
        coverage_error_interval=Interval(float(e_lo), float(e_hi)),
        overconfidence=float(np.mean(over)),
        overconfidence_interval=Interval(float(o_lo), float(o_hi)),
        sharpness=float(np.mean(sd)),
        n=int(y.size),
    )


# --------------------------------------------------------------------------
# optimism from model selection
# --------------------------------------------------------------------------
@dataclass
class SelectionOptimism:
    """Winner's curse incurred by selecting the best of ``n_models``.

    Estimated by nested resampling over evaluation folds: on each bootstrap
    resample of the folds we select the arg-max model *in sample* and score it
    on the out-of-bag folds.  The mean gap is the optimism that must be
    subtracted before a selected model's score is reported as a result.
    """

    optimism: float
    interval: Interval
    n_models: int
    n_folds: int
    selected_model: int
    naive_score: float

    @property
    def corrected_score(self) -> float:
        return self.naive_score - self.optimism

    def metrics(self, prefix: str = "selection") -> list[Metric]:
        return [
            Metric(
                name=f"{prefix}.optimism",
                value=self.optimism,
                kind="systematic",
                interval=self.interval,
                direction="less_is_better",
                note=(
                    f"selection among {self.n_models} candidates on {self.n_folds} folds; "
                    f"naive={self.naive_score:.6g} corrected={self.corrected_score:.6g}"
                ),
            ),
            Metric(
                name=f"{prefix}.corrected_score",
                value=self.corrected_score,
                kind="diagnostic",
                exact=True,
            ),
        ]


def selection_optimism(
    fold_scores: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    higher_is_better: bool = True,
) -> SelectionOptimism:
    """``fold_scores`` has shape ``(n_models, n_folds)`` (higher = better)."""
    s = np.asarray(fold_scores, dtype=float)
    if s.ndim != 2:
        raise ValueError("fold_scores must be (n_models, n_folds)")
    if not higher_is_better:
        s = -s
    n_models, n_folds = s.shape
    if n_folds < 3:
        raise ValueError("selection optimism needs >=3 folds to resample")
    means = s.mean(axis=1)
    sel = int(np.argmax(means))
    naive = float(means[sel])
    rng = np.random.default_rng(seed)
    gaps: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_folds, size=n_folds)
        oob = np.setdiff1d(np.arange(n_folds), np.unique(idx))
        if oob.size == 0:
            continue
        in_means = s[:, idx].mean(axis=1)
        k = int(np.argmax(in_means))
        gaps.append(float(in_means[k] - s[k, oob].mean()))
    if not gaps:  # pragma: no cover - astronomically unlikely
        gaps = [0.0]
    g = np.asarray(gaps)
    lo, hi = np.quantile(g, [0.025, 0.975])
    return SelectionOptimism(
        optimism=float(np.mean(g)),
        interval=Interval(float(lo), float(hi), method="nested-bootstrap"),
        n_models=n_models,
        n_folds=n_folds,
        selected_model=sel,
        naive_score=naive if higher_is_better else -naive,
    )


# --------------------------------------------------------------------------
# bias / systematic error
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class StratumBias:
    factor: str
    level: str
    n: int
    bias: float
    interval: Interval
    rmse: float


@dataclass
class BiasAnalysis:
    """Per-stratum bias across session/device/site/anatomy/demographic/task."""

    strata: list[StratumBias] = field(default_factory=list)

    @property
    def worst(self) -> StratumBias | None:
        if not self.strata:
            return None
        return max(self.strata, key=lambda s: abs(s.bias))

    @property
    def worst_abs_bias(self) -> float:
        w = self.worst
        return abs(w.bias) if w else float("nan")

    @property
    def spread(self) -> float:
        """Max-min bias across strata: how much the error depends on context."""
        if not self.strata:
            return float("nan")
        b = [s.bias for s in self.strata]
        return float(max(b) - min(b))

    def table(self) -> list[dict[str, Any]]:
        return [
            {
                "factor": s.factor,
                "level": s.level,
                "n": s.n,
                "bias": s.bias,
                "ci": [s.interval.lo, s.interval.hi],
                "rmse": s.rmse,
            }
            for s in self.strata
        ]

    def metrics(self, prefix: str = "bias") -> list[Metric]:
        w = self.worst
        note = f"worst stratum: {w.factor}={w.level} (n={w.n})" if w else "no strata supplied"
        return [
            Metric(
                name=f"{prefix}.worst_stratum_abs_bias",
                value=self.worst_abs_bias,
                kind="systematic",
                interval=(w.interval if w else Interval(float("nan"), float("nan"))),
                direction="less_is_better",
                note=note,
            ),
            Metric(
                name=f"{prefix}.across_stratum_spread",
                value=self.spread,
                kind="systematic",
                exact=True,
                direction="less_is_better",
            ),
        ]


def stratified_bias(
    y: np.ndarray,
    pred: np.ndarray,
    strata: Mapping[str, Sequence[Any]],
    *,
    seed: int = 0,
    n_boot: int = 500,
    min_n: int = 5,
) -> BiasAnalysis:
    """Signed bias ``mean(pred - y)`` within each level of each factor.

    ``strata`` maps a factor name ("site", "device", "session", "anatomy",
    "demographic", "task") to a per-observation label array.  §11.2 requires
    all of them; supply what exists and the report will say which were absent.
    """
    y = np.asarray(y, dtype=float).ravel()
    pred = np.asarray(pred, dtype=float).ravel()
    resid = pred - y
    out: list[StratumBias] = []
    for factor, labels in strata.items():
        lab = np.asarray(list(labels))
        if lab.shape[0] != y.shape[0]:
            raise ValueError(f"stratum {factor!r} has {lab.shape[0]} labels for {y.shape[0]} obs")
        for level in sorted(set(lab.tolist()), key=str):
            m = lab == level
            n = int(m.sum())
            if n < min_n:
                continue
            _, iv = bootstrap_ci(resid[m], n_boot=n_boot, seed=seed)
            out.append(
                StratumBias(
                    factor=factor,
                    level=str(level),
                    n=n,
                    bias=float(np.mean(resid[m])),
                    interval=iv,
                    rmse=float(np.sqrt(np.mean(resid[m] ** 2))),
                )
            )
    return BiasAnalysis(strata=out)


def systematic_error(
    bias: BiasAnalysis,
    *,
    external_bounds: Mapping[str, tuple[float, float]] | None = None,
    model_discrepancy: float | None = None,
) -> tuple[float, str, dict[str, Any]]:
    """Plausible systematic error to report *next to* the variance.

    Returns ``(magnitude, status, detail)`` where ``status`` is one of the
    thesis §2.7 ledger statuses: ``design_estimable`` (strata supplied and the
    contrast identifies the bias), ``externally_bounded`` (a phantom /
    calibration target / independent instrument bounds it), or
    ``prior_specified_sensitivity`` (neither: swept, never advertised as
    estimated).
    """
    parts: dict[str, Any] = {}
    mag = 0.0
    status = "prior_specified_sensitivity"
    if bias.strata:
        mag = max(mag, bias.worst_abs_bias)
        status = "design_estimable"
        parts["worst_stratum_abs_bias"] = bias.worst_abs_bias
        parts["across_stratum_spread"] = bias.spread
    if external_bounds:
        b = max(max(abs(lo), abs(hi)) for lo, hi in external_bounds.values())
        mag = max(mag, float(b))
        status = "externally_bounded" if not bias.strata else status
        parts["external_bounds"] = dict(external_bounds)
    if model_discrepancy is not None:
        mag = max(mag, abs(float(model_discrepancy)))
        parts["model_discrepancy"] = float(model_discrepancy)
    if not bias.strata and not external_bounds and model_discrepancy is None:
        parts["warning"] = (
            "no strata, no external bound, no discrepancy estimate: systematic error is "
            "prior-specified sensitivity only and must not be advertised as estimated"
        )
    return float(mag), status, parts


# --------------------------------------------------------------------------
# the "smoothed away the effect of interest" check
# --------------------------------------------------------------------------
@dataclass
class SmoothingVerdict:
    """§11.4: a lower-variance model is not preferred when it achieves
    stability by smoothing away the effect of interest."""

    arm: str
    reference: str
    variance_ratio: float          # var(arm residual) / var(ref residual)
    effect_true: float
    effect_arm: float
    effect_reference: float
    effect_retention: float        # |effect_arm| / |effect_true|
    lower_variance: bool
    attenuated: bool
    interval: Interval
    retention_floor: float

    @property
    def smoothed_away(self) -> bool:
        """True when the arm buys stability by destroying the effect."""
        return bool(self.lower_variance and self.attenuated)

    @property
    def verdict(self) -> str:
        if self.smoothed_away:
            return (
                "REJECT-PREFERENCE: lower variance obtained by attenuating the effect of "
                f"interest to {self.effect_retention:.2f} of truth "
                f"(floor {self.retention_floor:.2f})"
            )
        if self.lower_variance:
            return "lower variance with the effect preserved"
        return "no variance advantage"

    def metrics(self, prefix: str = "smoothing") -> list[Metric]:
        return [
            Metric(
                name=f"{prefix}.effect_retention",
                value=self.effect_retention,
                kind="systematic",
                interval=self.interval,
                threshold=self.retention_floor,
                direction="greater_is_better",
                note=self.verdict,
            ),
            Metric(
                name=f"{prefix}.variance_ratio",
                value=self.variance_ratio,
                kind="diagnostic",
                exact=True,
                note="residual variance of arm / reference; <1 means 'more stable'",
            ),
            Metric(
                name=f"{prefix}.smoothed_away_effect",
                value=float(self.smoothed_away),
                kind="systematic",
                exact=True,
                threshold=0.5,
                direction="less_is_better",
                note="1.0 = the arm's stability came from destroying the effect",
            ),
        ]


def smoothing_check(
    *,
    arm_name: str,
    reference_name: str,
    y_true: np.ndarray,
    pred_arm: np.ndarray,
    pred_reference: np.ndarray,
    effect: Callable[[np.ndarray], float],
    retention_floor: float = 0.5,
    seed: int = 0,
    n_boot: int = 1000,
) -> SmoothingVerdict:
    """Fire when an arm's lower variance comes from attenuating the effect.

    ``effect`` extracts the *effect of interest* from a signal: a condition
    contrast, an intervention response amplitude, a high-frequency energy, a
    dose slope.  It is applied identically to the truth and to both
    predictions, so "the model is smoother" and "the model lost the effect"
    are separated rather than conflated.
    """
    y_true = np.asarray(y_true, dtype=float)
    pred_arm = np.asarray(pred_arm, dtype=float)
    pred_reference = np.asarray(pred_reference, dtype=float)
    v_arm = float(np.var(pred_arm - y_true))
    v_ref = float(np.var(pred_reference - y_true))
    e_true = float(effect(y_true))
    e_arm = float(effect(pred_arm))
    e_ref = float(effect(pred_reference))
    denom = abs(e_true) if abs(e_true) > _EPS else _EPS
    retention = abs(e_arm) / denom

    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        et = effect(y_true[idx])
        ea = effect(pred_arm[idx])
        boots[b] = abs(ea) / (abs(et) if abs(et) > _EPS else _EPS)
    lo, hi = np.quantile(boots, [0.025, 0.975])

    # "smoother predictions" is measured on the predictions themselves as well
    # as on the residuals: a model can be stable because it is right, or
    # because it is flat.  Flatness relative to the reference is the tell.
    lower_variance = bool(v_arm < v_ref or np.var(pred_arm) < np.var(pred_reference))
    return SmoothingVerdict(
        arm=arm_name,
        reference=reference_name,
        variance_ratio=(v_arm / v_ref) if v_ref > _EPS else float("inf"),
        effect_true=e_true,
        effect_arm=e_arm,
        effect_reference=e_ref,
        effect_retention=float(retention),
        lower_variance=lower_variance,
        attenuated=bool(retention < retention_floor),
        interval=Interval(float(lo), float(hi)),
        retention_floor=float(retention_floor),
    )


# --------------------------------------------------------------------------
# data efficiency and decisions
# --------------------------------------------------------------------------
def data_efficiency_curve(
    sizes: Sequence[int],
    scores: Sequence[Sequence[float]],
    *,
    target: float | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Score-vs-training-size curve with intervals, plus n-to-reach-target.

    ``scores[i]`` are the per-seed (or per-fold) held-out scores at
    ``sizes[i]``.  Data efficiency is the *area under this curve*, and the
    smallest training size that reaches ``target``; a model that only wins at
    the largest size has not demonstrated data efficiency.
    """
    sizes = list(sizes)
    means, ivs = [], []
    for i, s in enumerate(scores):
        pt, iv = bootstrap_ci(np.asarray(s, dtype=float), seed=seed + i)
        means.append(pt)
        ivs.append(iv)
    auc = float(np.trapezoid(means, np.log(np.asarray(sizes, dtype=float))))
    n_to_target: int | None = None
    if target is not None:
        for n, m in zip(sizes, means):
            if m >= target:
                n_to_target = int(n)
                break
    return {
        "sizes": sizes,
        "mean": means,
        "ci_lo": [iv.lo for iv in ivs],
        "ci_hi": [iv.hi for iv in ivs],
        "auc_log_size": auc,
        "n_to_target": n_to_target,
        "target": target,
    }


def decision_regret(
    utility: np.ndarray,
    chosen: np.ndarray,
    *,
    seed: int = 0,
) -> tuple[float, Interval]:
    """Regret of a model-guided choice against the oracle choice.

    ``utility`` is ``(n_decisions, n_options)`` realised utility; ``chosen``
    the option index the model selected.  Regret is ``max_j u[i,j] -
    u[i,chosen[i]]``; lower is better, zero is the oracle.
    """
    u = np.asarray(utility, dtype=float)
    c = np.asarray(chosen, dtype=int).ravel()
    if u.ndim != 2 or c.shape[0] != u.shape[0]:
        raise ValueError("utility must be (n_decisions, n_options) matching chosen")
    r = u.max(axis=1) - u[np.arange(u.shape[0]), c]
    return bootstrap_ci(r, seed=seed)


# --------------------------------------------------------------------------
# plotting: every reported number ships with an interval
# --------------------------------------------------------------------------
def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_metric_intervals(
    labels: Sequence[str],
    values: Sequence[float],
    intervals: Sequence[Interval],
    *,
    path: str,
    title: str = "",
    xlabel: str = "",
    zero_line: bool = True,
) -> str:
    """Horizontal caterpillar plot.  A point without an interval is refused."""
    if not (len(labels) == len(values) == len(intervals)):
        raise ValueError("labels/values/intervals length mismatch")
    if any(iv is None for iv in intervals):
        raise ValueError("plot_metric_intervals refuses to plot a point estimate with no interval")
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(labels) + 1.6))
    ys = np.arange(len(labels))
    lo = np.array([iv.lo for iv in intervals])
    hi = np.array([iv.hi for iv in intervals])
    v = np.asarray(values, dtype=float)
    ax.errorbar(v, ys, xerr=[v - lo, hi - v], fmt="o", capsize=3, color="#22506e")
    if zero_line:
        ax.axvline(0.0, color="#b03030", lw=1, ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_calibration(reports: Mapping[str, CalibrationReport], *, path: str,
                     title: str = "calibration in the deployment population") -> str:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], color="black", lw=1, ls="--", label="ideal")
    for name, r in reports.items():
        ax.plot(r.nominal, r.empirical, marker="o", label=f"{name} (n={r.n})")
    ax.set_xlabel("nominal central coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_data_efficiency(curves: Mapping[str, Mapping[str, Any]], *, path: str,
                         title: str = "data efficiency") -> str:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for name, c in curves.items():
        ax.plot(c["sizes"], c["mean"], marker="o", label=name)
        ax.fill_between(c["sizes"], c["ci_lo"], c["ci_hi"], alpha=0.18)
    ax.set_xscale("log")
    ax.set_xlabel("training set size")
    ax.set_ylabel("held-out log score (nats/obs)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
