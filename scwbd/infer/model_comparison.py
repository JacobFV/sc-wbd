"""Model comparison over alternative dynamical backends ``M``, with **posterior
branching** instead of averaging.

``body.tex`` sec. 7.1 states the rule this module exists to enforce:

    "When two models remain observationally equivalent but imply different
    treatments, the output is an unresolved causal ambiguity, not an averaged
    recommendation."

So :func:`compare_models` returns either a ranked comparison **or** an
:class:`~scwbd.infer.types.UnresolvedCausalAmbiguity` -- an object with no
``mean``, whose ``averaged_recommendation()`` raises.  This is also the runtime
face of refusal **R04** (an effective/causal operator estimated from passive
correlation alone): the constructed example in ``tests/infer`` is a pair of
models that differ only in the *direction* of a coupling and are exactly
indistinguishable under a symmetric passive read, yet make opposite predictions
under a write to one node.

Also here: exact marginal likelihood for linear--Gaussian backends (the Kalman
prediction-error decomposition integrates the states exactly), Laplace evidence
for the general case, WAIC, PSIS-LOO with the Pareto ``k`` diagnostic, and
Bayes factors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .types import UnresolvedCausalAmbiguity, as_builtin

__all__ = [
    "ModelEvidence",
    "bayes_factor",
    "compare_models",
    "laplace_log_evidence",
    "psis_loo",
    "stacking_weights",
    "waic",
]


@dataclass
class ModelEvidence:
    name: str
    log_evidence: float
    method: str
    n_parameters: int | None = None
    detail: dict[str, Any] = field(default_factory=dict)


def laplace_log_evidence(
    log_posterior_at_mode: float, hessian: np.ndarray, log_prior_normaliser: float = 0.0
) -> float:
    """``log Z ~= log p(y, eta*) + d/2 log 2pi - 1/2 log|H|``."""
    H = np.asarray(hessian, float)
    d = H.shape[0]
    sign, logdet = np.linalg.slogdet(0.5 * (H + H.T))
    if sign <= 0:
        return float("-inf")
    return float(
        log_posterior_at_mode + 0.5 * d * math.log(2 * math.pi) - 0.5 * logdet
        + log_prior_normaliser
    )


def waic(log_lik: np.ndarray) -> dict[str, float]:
    """WAIC from a ``[n_draws, n_observations]`` pointwise log-likelihood."""
    ll = np.asarray(log_lik, float)
    lppd = np.log(np.exp(ll - ll.max(0)).mean(0)) + ll.max(0)
    p_waic = ll.var(0, ddof=1)
    elpd = lppd - p_waic
    n = ll.shape[1]
    return {
        "elpd_waic": float(elpd.sum()),
        "se": float(np.std(elpd, ddof=1) * math.sqrt(n)),
        "p_waic": float(p_waic.sum()),
        "waic": float(-2 * elpd.sum()),
        "n_observations": int(n),
        "p_waic_max": float(p_waic.max()),
        "warning": (
            "p_waic > 0.4 for some observation: WAIC is unreliable, prefer PSIS-LOO"
            if float(p_waic.max()) > 0.4 else ""
        ),
    }


def _gpdfit(x: np.ndarray) -> tuple[float, float]:
    """Empirical-Bayes generalised-Pareto fit (Zhang & Stephens 2009).

    ``x`` must be sorted ascending and strictly positive.  Returns ``(k, sigma)``
    with the usual weakly informative prior pulling ``k`` toward 0.5.
    """
    n = x.size
    m = 30 + int(math.sqrt(n))
    prior_b, prior_k = 3.0, 10.0
    b = 1.0 - np.sqrt(m / (np.arange(1, m + 1, dtype=float) - 0.5))
    b /= prior_b * x[max(int(n / 4 + 0.5) - 1, 0)]
    b += 1.0 / x[-1]
    k = np.log1p(-b[:, None] * x[None, :]).mean(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        L = n * (np.log(-(b / k)) - k - 1.0)
    ok = np.isfinite(L)
    if not ok.any():
        return float("nan"), float("nan")
    b, L = b[ok], L[ok]
    w = 1.0 / np.exp(L - L[:, None]).sum(axis=1)
    w /= w.sum()
    b_post = float((b * w).sum())
    k_post = float(np.log1p(-b_post * x).mean())
    sigma = -k_post / b_post if b_post != 0 else float("nan")
    k_post = (n * k_post + prior_k * 0.5) / (n + prior_k)
    return k_post, sigma


def _psis_smooth(lw: np.ndarray) -> tuple[np.ndarray, float]:
    """Pareto-smooth the largest importance weights; returns (log w, k-hat)."""
    S = lw.size
    M = int(min(0.2 * S, 3 * math.sqrt(S)))
    if M < 5:
        return lw, float("nan")
    order = np.argsort(lw)
    tail = order[-M:]
    cutoff = lw[order[-M - 1]]
    x = np.exp(lw[tail] - cutoff) - 1.0
    srt = np.argsort(x)
    xs = x[srt]
    if xs[0] <= 0:
        return lw, float("nan")
    k, sigma = _gpdfit(xs)
    if not np.isfinite(k) or not np.isfinite(sigma) or k >= 1.0:
        return lw, float(k)
    p = (np.arange(1, M + 1) - 0.5) / M
    if abs(k) < 1e-8:
        q = -sigma * np.log1p(-p)
    else:
        q = sigma / k * (np.power(1.0 - p, -k) - 1.0)
    out = lw.copy()
    out[tail[srt]] = np.log1p(q) + cutoff
    return np.minimum(out, lw.max()), float(k)


def psis_loo(log_lik: np.ndarray) -> dict[str, Any]:
    """Pareto-smoothed importance-sampling LOO with the ``k-hat`` diagnostic."""
    ll = np.asarray(log_lik, float)
    S, N = ll.shape
    elpd = np.zeros(N)
    khat = np.zeros(N)
    for i in range(N):
        lw = -ll[:, i]
        lw = lw - lw.max()
        sm, k = _psis_smooth(lw)
        khat[i] = k
        w = np.exp(sm - sm.max())
        w /= w.sum()
        elpd[i] = math.log(max(float((w * np.exp(ll[:, i] - ll[:, i].max())).sum()), 1e-300)) + ll[:, i].max()
    bad = int((khat > 0.7).sum())
    return {
        "elpd_loo": float(elpd.sum()),
        "se": float(np.std(elpd, ddof=1) * math.sqrt(N)),
        "looic": float(-2 * elpd.sum()),
        "pareto_k_max": float(np.nanmax(khat)),
        "n_khat_above_0.7": bad,
        "n_observations": int(N),
        "reliable": bool(bad == 0),
        "warning": "" if bad == 0 else f"{bad} observations with k-hat > 0.7: LOO unreliable",
    }


def bayes_factor(a: ModelEvidence, b: ModelEvidence) -> dict[str, Any]:
    d = a.log_evidence - b.log_evidence
    if d > 4.6:
        strength = "decisive for " + a.name
    elif d > 2.3:
        strength = "strong for " + a.name
    elif d > 1.0:
        strength = "substantial for " + a.name
    elif d > -1.0:
        strength = "not worth more than a bare mention"
    elif d > -2.3:
        strength = "substantial for " + b.name
    elif d > -4.6:
        strength = "strong for " + b.name
    else:
        strength = "decisive for " + b.name
    return {
        "numerator": a.name, "denominator": b.name,
        "log_bayes_factor": float(d), "bayes_factor": float(math.exp(min(d, 700))),
        "interpretation": strength,
    }


def stacking_weights(elpd_pointwise: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Log-score stacking weights.  Provided for *predictive* pooling only --
    never for causal recommendations (see :func:`compare_models`)."""
    from scipy.optimize import minimize

    names = list(elpd_pointwise)
    L = np.stack([np.exp(np.asarray(elpd_pointwise[n], float)) for n in names], axis=1)

    def neg(w_raw):
        w = np.exp(w_raw - w_raw.max())
        w = w / w.sum()
        return -np.log(np.maximum(L @ w, 1e-300)).sum()

    res = minimize(neg, np.zeros(len(names)), method="Nelder-Mead",
                   options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-8})
    w = np.exp(res.x - res.x.max())
    w = w / w.sum()
    return {n: float(v) for n, v in zip(names, w)}


def compare_models(
    evidences: Sequence[ModelEvidence],
    *,
    intervention_predictions: Mapping[str, Mapping[str, float]] | None = None,
    intervention_uncertainty: Mapping[str, float] | None = None,
    equivalence_threshold: float = 2.3,
    divergence_threshold: float = 2.0,
    resolution_experiment: str = "",
) -> dict[str, Any] | UnresolvedCausalAmbiguity:
    """Rank models -- unless they are observationally equivalent *and* imply
    different interventions, in which case the output is the ambiguity itself.

    Parameters
    ----------
    equivalence_threshold:
        Two models are treated as observationally equivalent when their log
        evidences differ by less than this (default 2.3 nats ~ a Bayes factor
        of 10, i.e. below "strong" evidence).
    intervention_predictions:
        ``{model: {intervention: predicted_effect}}``.
    intervention_uncertainty:
        ``{intervention: sd}`` used to standardise the divergence, so that a
        disagreement is only counted when it exceeds the *predictive* noise.
        Without it the divergence is reported in raw units.
    """
    evs = list(evidences)
    if len(evs) < 2:
        raise ValueError("model comparison needs at least two models")
    logz = {e.name: float(e.log_evidence) for e in evs}
    best = max(logz, key=logz.get)
    ranked = sorted(logz, key=logz.get, reverse=True)
    lz = np.array([logz[n] for n in ranked])
    w = np.exp(lz - lz.max())
    w = w / w.sum()
    weights = {n: float(v) for n, v in zip(ranked, w)}

    tied = [n for n in ranked if logz[best] - logz[n] <= equivalence_threshold]
    if intervention_predictions and len(tied) >= 2:
        keys = sorted(
            set().union(*[set(intervention_predictions[n]) for n in tied
                          if n in intervention_predictions])
        )
        worst = 0.0
        divergent: dict[str, dict[str, float]] = {}
        for k in keys:
            vals = {n: float(intervention_predictions[n][k]) for n in tied
                    if n in intervention_predictions and k in intervention_predictions[n]}
            if len(vals) < 2:
                continue
            spread = max(vals.values()) - min(vals.values())
            sd = float((intervention_uncertainty or {}).get(k, 1.0))
            z = spread / sd if sd > 0 else float("inf")
            divergent[k] = dict(vals, spread=spread, standardised_divergence=z)
            worst = max(worst, z)
        if worst > divergence_threshold:
            return UnresolvedCausalAmbiguity(
                candidate_models=tied,
                log_evidence={n: logz[n] for n in tied},
                max_log_evidence_gap=float(max(logz[n] for n in tied)
                                           - min(logz[n] for n in tied)),
                equivalence_threshold=equivalence_threshold,
                divergent_interventions=divergent,
                intervention_divergence=float(worst),
                divergence_threshold=divergence_threshold,
                posterior_weights={n: weights[n] for n in tied},
                resolution_experiment=resolution_experiment or (
                    "acquire the intervention with the largest standardised "
                    "divergence under an identified design; passive data cannot "
                    "separate these models (refusal R04)."
                ),
                detail={"all_log_evidence": logz},
            )
    return {
        "ranking": ranked,
        "log_evidence": logz,
        "posterior_weights": weights,
        "best": best,
        "bayes_factor_best_vs_second": (
            bayes_factor(
                next(e for e in evs if e.name == ranked[0]),
                next(e for e in evs if e.name == ranked[1]),
            )
        ),
        "observationally_tied": tied,
        "resolved": True,
    }
