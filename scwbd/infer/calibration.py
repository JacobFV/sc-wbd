"""Calibration diagnostics and proper scoring rules.

``body.tex`` sec. 11.2: *"Metrics are reported with sampling variance, bootstrap
or posterior intervals ... and bias analyses across session, device, acquisition
site, anatomy, demographic strata, and task context.  Aggregate accuracy cannot
substitute for calibration within the intended deployment population."*

Everything here therefore returns an error bar, and every aggregate metric has a
subgroup counterpart with a worst-group summary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .types import CoverageResult, Interval, as_builtin

__all__ = [
    "ReliabilityDiagram",
    "bootstrap_ci",
    "crps_ensemble",
    "crps_gaussian",
    "expected_coverage_curve",
    "interval_coverage",
    "log_score_gaussian",
    "pit_histogram",
    "pit_values",
    "reliability_diagram",
    "sharpness",
    "subgroup_calibration",
]

_SQRT2 = math.sqrt(2.0)


def _phi(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)


def _Phi(x: np.ndarray) -> np.ndarray:
    from scipy.special import ndtr

    return ndtr(x)


def interval_coverage(
    truth: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    nominal: float = 0.95,
    name: str = "",
) -> CoverageResult:
    truth, lower, upper = map(np.asarray, (truth, lower, upper))
    inside = (lower <= truth) & (truth <= upper)
    return CoverageResult(name, nominal, int(truth.size), int(inside.sum()))


def pit_values(
    truth: np.ndarray, mean: np.ndarray, sd: np.ndarray
) -> np.ndarray:
    """Probability integral transform under a Gaussian predictive law."""
    sd = np.where(np.asarray(sd) > 0, sd, np.nan)
    return _Phi((np.asarray(truth) - np.asarray(mean)) / sd)


def pit_histogram(pit: np.ndarray, n_bins: int = 20) -> dict[str, Any]:
    """PIT histogram plus a chi-square test of uniformity."""
    from scipy.stats import chisquare, kstest

    pit = np.asarray(pit)
    pit = pit[np.isfinite(pit)]
    counts, edges = np.histogram(pit, bins=n_bins, range=(0.0, 1.0))
    expected = pit.size / n_bins
    chi = chisquare(counts, f_exp=np.full(n_bins, expected))
    ks = kstest(pit, "uniform")
    return {
        "n": int(pit.size),
        "n_bins": n_bins,
        "counts": counts.tolist(),
        "bin_edges": edges.tolist(),
        "expected_per_bin": float(expected),
        "chi2_statistic": float(chi.statistic),
        "chi2_pvalue": float(chi.pvalue),
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "mean": float(pit.mean()),
        "uniform_at_005": bool(chi.pvalue > 0.05),
    }


@dataclass
class ReliabilityDiagram:
    bin_centres: list[float]
    predicted: list[float]
    observed: list[float]
    counts: list[int]
    observed_se: list[float]
    expected_calibration_error: float

    def to_dict(self) -> dict[str, Any]:
        return as_builtin(self.__dict__ | {
            "expected_calibration_error": self.expected_calibration_error
        })


def reliability_diagram(
    probabilities: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
) -> ReliabilityDiagram:
    p = np.asarray(probabilities, float)
    y = np.asarray(outcomes, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    centres, pred, obs, cnt, se = [], [], [], [], []
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        centres.append(float(0.5 * (edges[b] + edges[b + 1])))
        if n == 0:
            pred.append(float("nan")); obs.append(float("nan"))
            cnt.append(0); se.append(float("nan"))
            continue
        pb, ob = float(p[m].mean()), float(y[m].mean())
        pred.append(pb); obs.append(ob); cnt.append(n)
        se.append(float(math.sqrt(max(ob * (1 - ob), 0.0) / n)))
        ece += n / p.size * abs(pb - ob)
    return ReliabilityDiagram(centres, pred, obs, cnt, se, float(ece))


def sharpness(sd: np.ndarray, level: float = 0.95) -> dict[str, float]:
    """Sharpness = expected predictive interval width.  Only meaningful jointly
    with calibration: a sharp but miscalibrated forecast is worse than a wide
    honest one."""
    from scipy.stats import norm

    sd = np.asarray(sd, float)
    z = float(norm.ppf(0.5 * (1 + level)))
    w = 2 * z * sd
    return {
        "level": level,
        "mean_interval_width": float(w.mean()),
        "median_interval_width": float(np.median(w)),
        "se": float(w.std(ddof=1) / math.sqrt(w.size)) if w.size > 1 else float("nan"),
        "mean_sd": float(sd.mean()),
    }


def log_score_gaussian(
    truth: np.ndarray, mean: np.ndarray, sd: np.ndarray
) -> dict[str, float]:
    """Negatively oriented log score (smaller is better), with its standard error."""
    truth, mean, sd = map(lambda a: np.asarray(a, float), (truth, mean, sd))
    s = 0.5 * np.log(2 * math.pi * sd**2) + 0.5 * ((truth - mean) / sd) ** 2
    return {
        "log_score": float(s.mean()),
        "se": float(s.std(ddof=1) / math.sqrt(s.size)) if s.size > 1 else float("nan"),
        "n": int(s.size),
    }


def crps_gaussian(
    truth: np.ndarray, mean: np.ndarray, sd: np.ndarray
) -> dict[str, float]:
    """Closed-form CRPS for a Gaussian predictive distribution."""
    truth, mean, sd = map(lambda a: np.asarray(a, float), (truth, mean, sd))
    z = (truth - mean) / sd
    c = sd * (z * (2 * _Phi(z) - 1) + 2 * _phi(z) - 1 / math.sqrt(math.pi))
    return {
        "crps": float(c.mean()),
        "se": float(c.std(ddof=1) / math.sqrt(c.size)) if c.size > 1 else float("nan"),
        "n": int(c.size),
    }


def crps_ensemble(truth: np.ndarray, ensemble: np.ndarray) -> dict[str, float]:
    """Fair (unbiased) ensemble CRPS.  ``ensemble`` is ``[n, m]``."""
    truth = np.asarray(truth, float)
    ens = np.asarray(ensemble, float)
    n, m = ens.shape
    term1 = np.abs(ens - truth[:, None]).mean(axis=1)
    srt = np.sort(ens, axis=1)
    i = np.arange(1, m + 1)
    # E|X - X'| via the order-statistic identity, with the m/(m-1) fair factor
    term2 = (2.0 / (m * (m - 1))) * ((2 * i - m - 1) * srt).sum(axis=1)
    c = term1 - 0.5 * term2
    return {
        "crps": float(c.mean()),
        "se": float(c.std(ddof=1) / math.sqrt(c.size)) if c.size > 1 else float("nan"),
        "n": int(c.size),
        "ensemble_size": m,
    }


def expected_coverage_curve(
    truth: np.ndarray,
    mean: np.ndarray,
    sd: np.ndarray,
    levels: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),
) -> dict[str, Any]:
    """Empirical vs nominal central-interval coverage across levels."""
    from scipy.stats import norm

    rows = []
    for lv in levels:
        z = float(norm.ppf(0.5 * (1 + lv)))
        cr = interval_coverage(truth, mean - z * sd, mean + z * sd,
                               nominal=lv, name=f"level_{lv}")
        rows.append(cr.to_dict())
    return {"levels": list(levels), "coverage": rows}


def bootstrap_ci(
    values: np.ndarray,
    statistic=np.mean,
    *,
    n_boot: int = 4000,
    level: float = 0.95,
    seed: int = 0,
) -> Interval:
    rng = np.random.default_rng(seed)
    v = np.asarray(values, float)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    stats = statistic(v[idx], axis=1)
    lo, hi = np.quantile(stats, [(1 - level) / 2, 1 - (1 - level) / 2])
    return Interval(float(lo), float(hi), level)


def paired_bootstrap_difference(
    a: np.ndarray, b: np.ndarray, *, n_boot: int = 4000, level: float = 0.95, seed: int = 0
) -> dict[str, Any]:
    """Paired bootstrap for ``mean(a) - mean(b)`` on matched replicates."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("paired bootstrap needs matched samples")
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    d = (a[idx] - b[idx]).mean(axis=1)
    lo, hi = np.quantile(d, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {
        "difference": float((a - b).mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "level": level,
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def subgroup_calibration(
    truth: np.ndarray,
    mean: np.ndarray,
    sd: np.ndarray,
    groups: Sequence[Any],
    *,
    nominal: float = 0.95,
) -> dict[str, Any]:
    """Per-subgroup coverage / PIT / log score, plus the **worst** subgroup.

    Aggregate calibration can hide a systematically overconfident stratum; the
    thesis requires the stratified view, so this returns the worst group as a
    first-class number rather than an appendix.
    """
    from scipy.stats import norm

    truth, mean, sd = map(lambda a: np.asarray(a, float), (truth, mean, sd))
    groups = np.asarray(groups)
    z = float(norm.ppf(0.5 * (1 + nominal)))
    per: dict[str, Any] = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        cr = interval_coverage(truth[m], mean[m] - z * sd[m], mean[m] + z * sd[m],
                               nominal=nominal, name=str(g))
        per[str(g)] = {
            "coverage": cr.to_dict(),
            "pit": pit_histogram(pit_values(truth[m], mean[m], sd[m]), n_bins=10),
            "log_score": log_score_gaussian(truth[m], mean[m], sd[m]),
            "crps": crps_gaussian(truth[m], mean[m], sd[m]),
            "n": int(m.sum()),
        }
    worst = min(per, key=lambda k: per[k]["coverage"]["empirical"]) if per else None
    worst_ls = max(per, key=lambda k: per[k]["log_score"]["log_score"]) if per else None
    overall = interval_coverage(truth, mean - z * sd, mean + z * sd, nominal=nominal,
                                name="overall")
    return {
        "nominal": nominal,
        "overall": overall.to_dict(),
        "per_group": per,
        "worst_coverage_group": worst,
        "worst_coverage": per[worst]["coverage"]["empirical"] if worst else None,
        "worst_log_score_group": worst_ls,
        "coverage_spread": (
            float(max(per[k]["coverage"]["empirical"] for k in per)
                  - min(per[k]["coverage"]["empirical"] for k in per)) if per else None
        ),
    }
