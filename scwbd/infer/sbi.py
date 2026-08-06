"""Simulation-based inference: amortized neural posterior estimation with
calibration diagnostics.

This is the object agent I inherits as the amortized posterior
(``ARCHITECTURE.md`` sec. 5: *"observations -> p(theta | Y)"*).  It is built
here, on the linear--Gaussian benchmark, precisely because the benchmark has an
exact posterior to check it against.

Contents
--------
* :func:`multirate_summary_statistics` -- summaries that respect native clocks
  (EEG autocovariance at millisecond lags, cross-sensor lagged covariances that
  carry the conduction delay, stimulus-locked evoked responses, BOLD evoked
  time courses and cross-parcel covariance).  Nothing is resampled.
* :class:`ConditionalMAF` -- a conditional masked autoregressive flow written
  directly in torch (no external flow dependency).
* :func:`train_npe` / :func:`sequential_npe` -- amortized and round-based
  (SNPE-B, importance-weighted) posterior estimation.
* :func:`simulation_based_calibration` -- SBC rank histograms with a
  chi-square uniformity test, and :func:`expected_coverage` -- empirical vs
  nominal credible-region coverage.

SBC is the honest check: a correctly specified inference pipeline must produce
**uniform** ranks.  A non-uniform histogram is a defect in the posterior, not a
property of the data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .calibration import pit_histogram
from .types import as_builtin, resolve_device, seed_everything

__all__ = [
    "ConditionalMAF",
    "NPEResult",
    "expected_coverage",
    "multirate_summary_statistics",
    "sequential_npe",
    "simulation_based_calibration",
    "train_npe",
]


# --------------------------------------------------------------------------
# Summary statistics -- computed on each modality's own clock
# --------------------------------------------------------------------------


def multirate_summary_statistics(
    data: dict[str, Tensor],
    *,
    dt_eeg: float = 1e-3,
    eeg_lags_ms: Sequence[int] = (1, 2, 3, 5, 8, 12, 16, 20, 25, 30),
    n_bold_lags: int = 3,
) -> Tensor:
    """Fixed-length summaries of a multirate record, ``[B, d]``.

    EEG lagged cross-covariances at millisecond lags are the statistics that
    can carry a conduction delay; block-averaging them onto the BOLD clock --
    which is what naive resampling does -- destroys them.  BOLD contributes
    its own slow covariance structure and evoked amplitude.
    """
    feats = []
    if "eeg" in data:
        y = data["eeg"]                       # [B, E, T, p]
        B, E, T, p = y.shape
        yc = y - y.mean(dim=2, keepdim=True)
        var = yc.var(dim=2).mean(dim=1)                       # [B, p]
        feats.append(torch.log(var + 1e-30))
        for L in eeg_lags_ms:
            L = int(round(L * 1e-3 / dt_eeg))
            if L >= T:
                continue
            a = yc[:, :, : T - L, :]
            b = yc[:, :, L:, :]
            cc = torch.einsum("betp,betq->bpq", a, b) / (E * (T - L))
            feats.append(cc.reshape(B, p * p))
        feats.append(yc.mean(dim=(1, 2)))
    if "bold" in data:
        z = data["bold"]                      # [B, E, N, q]
        B, E, N, q = z.shape
        zc = z - z.mean(dim=2, keepdim=True)
        feats.append(torch.log(zc.var(dim=2).mean(dim=1) + 1e-30))
        for L in range(1, min(n_bold_lags, N)):
            a, b = zc[:, :, : N - L, :], zc[:, :, L:, :]
            cc = torch.einsum("betp,betq->bpq", a, b) / (E * (N - L))
            feats.append(cc.reshape(B, q * q))
        feats.append(z.mean(dim=(1, 2)))
        feats.append(z.mean(dim=1).reshape(B, -1)[:, : min(N * q, 24)])
    return torch.cat([f.reshape(f.shape[0], -1) for f in feats], dim=1)


# --------------------------------------------------------------------------
# Conditional masked autoregressive flow
# --------------------------------------------------------------------------


class _MaskedLinear(nn.Linear):
    def __init__(self, i: int, o: int, mask: Tensor):
        super().__init__(i, o)
        self.register_buffer("mask", mask)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return nn.functional.linear(x, self.weight * self.mask, self.bias)


class _MADE(nn.Module):
    """One autoregressive conditioner producing (shift, log-scale).

    The context reaches the outputs through **two** paths: the masked hidden
    stack, and an unmasked direct map.  The direct map matters: for ``dim == 1``
    (and for the first variable in any ordering) the autoregressive output mask
    is empty by construction, so without it the conditioner would be a constant
    and the "conditional" flow would silently return the prior -- a
    normalised, plausible-looking density that ignores the data.
    """

    def __init__(self, dim: int, ctx: int, hidden: int, reverse: bool):
        super().__init__()
        order = np.arange(dim)[::-1].copy() if reverse else np.arange(dim)
        self.register_buffer("order", torch.tensor(order.copy(), dtype=torch.long))
        deg_in = torch.tensor(order.copy(), dtype=torch.long)
        deg_h = torch.arange(hidden) % max(dim - 1, 1)
        m1 = (deg_h.unsqueeze(1) >= deg_in.unsqueeze(0)).float()
        m2 = (deg_h.unsqueeze(1) >= deg_h.unsqueeze(0)).float()
        m3 = (deg_in.unsqueeze(1) > deg_h.unsqueeze(0)).float()
        self.l1 = _MaskedLinear(dim, hidden, m1)
        self.c1 = nn.Linear(ctx, hidden)
        self.l2 = _MaskedLinear(hidden, hidden, m2)
        self.shift = _MaskedLinear(hidden, dim, m3)
        self.logscale = _MaskedLinear(hidden, dim, m3)
        self.ctx_shift = nn.Linear(ctx, dim)
        self.ctx_logscale = nn.Linear(ctx, dim)
        for lin in (self.shift, self.logscale, self.ctx_shift, self.ctx_logscale):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, x: Tensor, ctx: Tensor) -> tuple[Tensor, Tensor]:
        h = torch.tanh(self.l1(x) + self.c1(ctx))
        h = torch.tanh(self.l2(h))
        shift = self.shift(h) + self.ctx_shift(ctx)
        logscale = torch.tanh(self.logscale(h) + self.ctx_logscale(ctx)) * 3.0
        return shift, logscale


class ConditionalMAF(nn.Module):
    """Conditional MAF over ``theta`` given summary statistics.

    Density evaluation is a single pass; sampling is autoregressive over
    ``dim`` steps per flow layer, which is fine for ``dim = 9``.
    """

    def __init__(self, dim: int, ctx: int, n_flows: int = 5, hidden: int = 128):
        super().__init__()
        self.dim = dim
        self.embed = nn.Sequential(
            nn.Linear(ctx, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU()
        )
        self.ctx_out = hidden
        self.flows = nn.ModuleList(
            [_MADE(dim, hidden, hidden, reverse=bool(i % 2)) for i in range(n_flows)]
        )

    def log_prob(self, theta: Tensor, context: Tensor) -> Tensor:
        c = self.embed(context)
        z = theta
        logdet = torch.zeros(theta.shape[0], dtype=theta.dtype, device=theta.device)
        for f in self.flows:
            s, ls = f(z, c)
            z = (z - s) * torch.exp(-ls)
            logdet = logdet - ls.sum(-1)
        base = -0.5 * (z**2).sum(-1) - 0.5 * self.dim * math.log(2 * math.pi)
        return base + logdet

    @torch.no_grad()
    def sample(self, n: int, context: Tensor, generator: torch.Generator | None = None) -> Tensor:
        c = self.embed(context)
        if c.shape[0] == 1:
            c = c.expand(n, -1)
        z = torch.randn(n, self.dim, dtype=c.dtype, device=c.device, generator=generator)
        for f in reversed(self.flows):
            x = torch.zeros_like(z)
            for _ in range(self.dim):
                s, ls = f(x, c)
                x = z * torch.exp(ls) + s
            z = x
        return z


@dataclass
class NPEResult:
    flow: ConditionalMAF
    theta_mean: np.ndarray
    theta_scale: np.ndarray
    x_mean: np.ndarray
    x_scale: np.ndarray
    history: list[float]
    validation: list[float]
    rounds: int = 1
    detail: dict[str, Any] = field(default_factory=dict)

    def posterior_samples(self, x: Tensor, n: int = 2000, seed: int = 0) -> np.ndarray:
        p = next(self.flow.parameters())
        x = torch.as_tensor(x, dtype=p.dtype, device=p.device).reshape(1, -1)
        xs = (x - torch.tensor(self.x_mean, dtype=p.dtype, device=p.device)) / torch.tensor(
            self.x_scale, dtype=p.dtype, device=p.device
        )
        g = torch.Generator(device=p.device)
        g.manual_seed(int(seed))
        z = self.flow.sample(n, xs, generator=g)
        return z.detach().cpu().numpy() * self.theta_scale + self.theta_mean


def _standardise(a: Tensor) -> tuple[Tensor, np.ndarray, np.ndarray]:
    m = a.mean(0)
    s = a.std(0).clamp_min(1e-8)
    return (a - m) / s, m.cpu().numpy(), s.cpu().numpy()


def train_npe(
    theta: Tensor,
    x: Tensor,
    *,
    n_flows: int = 5,
    hidden: int = 128,
    epochs: int = 400,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_fraction: float = 0.15,
    seed: int = 0,
    weights: Tensor | None = None,
    patience: int = 60,
    device: str | None = None,
) -> NPEResult:
    """Amortized neural posterior estimation ``q(theta | x)``."""
    seed_everything(seed)
    dev = resolve_device(device)
    theta = theta.to(dev).float()
    x = x.to(dev).float()
    th, tm, ts = _standardise(theta)
    xs, xm, xsd = _standardise(x)
    n = th.shape[0]
    n_val = max(1, int(val_fraction * n))
    perm = torch.randperm(n, device=dev)
    vi, ti = perm[:n_val], perm[n_val:]
    w = torch.ones(n, device=dev) if weights is None else weights.to(dev).float()

    flow = ConditionalMAF(th.shape[1], xs.shape[1], n_flows, hidden).to(dev)
    opt = torch.optim.Adam(flow.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    hist, val = [], []
    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        flow.train()
        idx = ti[torch.randperm(ti.numel(), device=dev)]
        tot = 0.0
        for s in range(0, idx.numel(), batch_size):
            b = idx[s : s + batch_size]
            loss = -(flow.log_prob(th[b], xs[b]) * w[b]).sum() / w[b].sum().clamp_min(1e-8)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach()) * b.numel()
        sched.step()
        hist.append(tot / max(idx.numel(), 1))
        flow.eval()
        with torch.no_grad():
            v = float(-flow.log_prob(th[vi], xs[vi]).mean())
        val.append(v)
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in flow.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        flow.load_state_dict(best_state)
    flow.eval()
    return NPEResult(flow, tm, ts, xm, xsd, hist, val, 1,
                     {"best_validation_nll": best, "n_train": int(ti.numel()),
                      "n_validation": int(vi.numel()), "epochs_run": len(hist)})


def sequential_npe(
    simulator: Callable[[np.ndarray, int], Tensor],
    prior_sample: Callable[[int, int], np.ndarray],
    prior_log_prob: Callable[[np.ndarray], np.ndarray],
    x_observed: Tensor,
    *,
    n_rounds: int = 2,
    n_per_round: int = 2000,
    seed: int = 0,
    **train_kw: Any,
) -> NPEResult:
    """SNPE-B: round-based refinement with importance weights ``p(th)/q(th)``.

    The proposal correction is explicit rather than folded into the loss, so the
    trained density is an estimate of the *prior*-conditioned posterior even
    though later rounds sample from the current posterior.
    """
    theta_all: list[np.ndarray] = []
    x_all: list[Tensor] = []
    w_all: list[np.ndarray] = []
    res: NPEResult | None = None
    for r in range(n_rounds):
        if res is None:
            th = prior_sample(n_per_round, seed + r)
            lw = np.zeros(n_per_round)
        else:
            th = res.posterior_samples(x_observed, n_per_round, seed=seed + 100 + r)
            thn = (th - res.theta_mean) / res.theta_scale
            xs = (x_observed.reshape(1, -1).cpu().numpy() - res.x_mean) / res.x_scale
            with torch.no_grad():
                dev = next(res.flow.parameters()).device
                lq = res.flow.log_prob(
                    torch.tensor(thn, dtype=torch.float32, device=dev),
                    torch.tensor(np.repeat(xs, len(th), 0), dtype=torch.float32, device=dev),
                ).cpu().numpy()
            lw = prior_log_prob(th) - lq
            lw = lw - lw.max()
        xr = simulator(th, seed + 555 + r)
        theta_all.append(th)
        x_all.append(xr)
        w_all.append(np.exp(np.clip(lw, -20, 0)))
        res = train_npe(
            torch.tensor(np.concatenate(theta_all), dtype=torch.float32),
            torch.cat(x_all, dim=0).float(),
            weights=torch.tensor(np.concatenate(w_all), dtype=torch.float32),
            seed=seed + r,
            **train_kw,
        )
        res.rounds = r + 1
    assert res is not None
    return res


# --------------------------------------------------------------------------
# Posterior calibration diagnostics
# --------------------------------------------------------------------------


def simulation_based_calibration(
    posterior_samples: Callable[[int], np.ndarray],
    theta_true: np.ndarray,
    *,
    names: Sequence[str],
    n_posterior: int = 199,
    n_bins: int = 20,
) -> dict[str, Any]:
    """SBC rank histograms.

    For each simulated dataset ``i`` the rank of ``theta_true[i]`` among
    ``n_posterior`` posterior draws is computed.  If the pipeline
    (prior + simulator + posterior) is self-consistent, ranks are **uniform**
    on ``{0, ..., n_posterior}``.  Systematic shapes have standard readings:
    U-shaped = over-confident, hump = under-confident, sloped = biased.
    """
    theta_true = np.asarray(theta_true, float)
    n_sim, dim = theta_true.shape
    ranks = np.zeros((n_sim, dim), dtype=int)
    for i in range(n_sim):
        s = np.asarray(posterior_samples(i), float)[:n_posterior]
        if s.shape[0] < n_posterior:
            raise ValueError("posterior_samples returned too few draws")
        ranks[i] = (s < theta_true[i][None, :]).sum(axis=0)
    out: dict[str, Any] = {"n_simulations": n_sim, "n_posterior_draws": n_posterior,
                           "per_parameter": {}}
    from scipy.stats import chisquare

    for j, nm in enumerate(names):
        counts, edges = np.histogram(
            ranks[:, j], bins=n_bins, range=(0, n_posterior + 1)
        )
        exp = n_sim / n_bins
        chi = chisquare(counts, f_exp=np.full(n_bins, exp))
        u = (ranks[:, j] + 0.5) / (n_posterior + 1)
        out["per_parameter"][nm] = {
            "counts": counts.tolist(),
            "bin_edges": edges.tolist(),
            "expected_per_bin": float(exp),
            "chi2_statistic": float(chi.statistic),
            "chi2_pvalue": float(chi.pvalue),
            "uniform_at_005": bool(chi.pvalue > 0.05),
            "mean_normalised_rank": float(u.mean()),
            "rank_uniformity": pit_histogram(u, n_bins=n_bins),
        }
    pv = [out["per_parameter"][n]["chi2_pvalue"] for n in names]
    out["min_pvalue"] = float(min(pv))
    out["bonferroni_pass_at_005"] = bool(min(pv) > 0.05 / len(names))
    return out


def expected_coverage(
    posterior_samples: Callable[[int], np.ndarray],
    theta_true: np.ndarray,
    *,
    names: Sequence[str],
    levels: Sequence[float] = (0.5, 0.8, 0.9, 0.95, 0.99),
) -> dict[str, Any]:
    """Empirical coverage of central credible intervals, per parameter."""
    from .types import CoverageResult

    theta_true = np.asarray(theta_true, float)
    n_sim, dim = theta_true.shape
    draws = [np.asarray(posterior_samples(i), float) for i in range(n_sim)]
    out: dict[str, Any] = {"levels": list(levels), "per_parameter": {}}
    for j, nm in enumerate(names):
        rows = []
        for lv in levels:
            lo = np.array([np.quantile(d[:, j], (1 - lv) / 2) for d in draws])
            hi = np.array([np.quantile(d[:, j], 1 - (1 - lv) / 2) for d in draws])
            n_cov = int(((lo <= theta_true[:, j]) & (theta_true[:, j] <= hi)).sum())
            rows.append(CoverageResult(nm, lv, n_sim, n_cov).to_dict())
        out["per_parameter"][nm] = rows
    return out
