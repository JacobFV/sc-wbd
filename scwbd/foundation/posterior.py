"""The amortized posterior ``p(theta | Y)`` and its calibration diagnostics.

ARCHITECTURE.md §5: "observations -> ``p(theta | Y)`` over global coupling,
conduction velocity, regional E/I balance, and observation nuisance. **This is
the 'characterize a general human brain' capability.**"

Design: a permutation-respecting summary network over the observation window
(learned multi-scale temporal convolutions **plus** explicit band-power and
covariance-spectrum statistics, because a purely learned summary is the easiest
place for a simulator shortcut to hide), followed by a conditional
affine-coupling normalizing flow with exact log-density.

Calibration is not optional and is not self-graded: :func:`sbc_ranks` and
:func:`expected_coverage` compute the diagnostics that agent H's SBC machinery
consumes, and :func:`posterior_report` refuses to call a posterior "calibrated"
on the basis of a low training loss.  A pseudo-likelihood must never be reported
as a calibrated posterior likelihood (refusal **R09**).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn

__all__ = [
    "SummaryNet",
    "ConditionalFlow",
    "AmortizedPosterior",
    "sbc_ranks",
    "expected_coverage",
    "posterior_report",
    "R09Violation",
]

BAND_EDGES = ((0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 20.0), (20.0, 30.0), (30.0, 45.0), (45.0, 60.0))


class R09Violation(RuntimeError):
    """A pseudo-likelihood was about to be reported as a calibrated posterior."""

    code = "R09"


# ======================================================================
# summary statistics
# ======================================================================
def spectral_summary(y: Tensor, fs: float, *, bands: Sequence[tuple[float, float]] = BAND_EDGES) -> Tensor:
    """Per-channel log band power + spectral edge + 1/f slope.  ``(B,T,C) -> (B,C,F)``.

    Explicit, auditable statistics.  A summary the modeller can name is a summary
    a reviewer can attack; a purely learned one is not.
    """
    B, T, C = y.shape
    x = y - y.mean(1, keepdim=True)
    win = torch.hann_window(T, device=y.device, dtype=x.dtype)
    X = torch.fft.rfft(x * win.reshape(1, T, 1), dim=1)
    P = (X.real**2 + X.imag**2) / max(T, 1)
    f = torch.fft.rfftfreq(T, d=1.0 / fs).to(y.device)
    feats = []
    tot = P.sum(1).clamp_min(1e-20)
    for lo, hi in bands:
        m = (f >= lo) & (f < hi)
        bp = P[:, m].sum(1) if m.any() else torch.zeros_like(tot)
        feats.append(torch.log(bp.clamp_min(1e-20)) - torch.log(tot))
    feats.append(torch.log(tot))
    # spectral edge frequency (95%) and aperiodic slope on log-log
    csum = P.cumsum(1) / tot.unsqueeze(1)
    edge = (csum < 0.95).sum(1).to(x.dtype) / max(len(f), 1)
    feats.append(edge)
    m = (f > 1.0) & (f < 45.0)
    if m.sum() > 3:
        lf = torch.log(f[m]).reshape(1, -1, 1)
        lp = torch.log(P[:, m].clamp_min(1e-20))
        lfc = lf - lf.mean(1, keepdim=True)
        slope = (lfc * (lp - lp.mean(1, keepdim=True))).sum(1) / (lfc**2).sum(1).clamp_min(1e-8)
        feats.append(slope)
    else:
        feats.append(torch.zeros_like(edge))
    return torch.stack(feats, dim=-1)  # (B, C, F)


def covariance_summary(y: Tensor, n_pcs: int = 16) -> Tensor:
    """Eigenvalue spectrum of the channel covariance -- the global-coupling signature."""
    B, T, C = y.shape
    x = y - y.mean(1, keepdim=True)
    sd = x.std(1, keepdim=True).clamp_min(1e-8)
    cov = torch.einsum("btc,btd->bcd", x / sd, x / sd) / max(T - 1, 1)
    ev = torch.linalg.eigvalsh(cov.float() + 1e-5 * torch.eye(C, device=y.device))
    ev = ev.flip(-1)[:, : min(n_pcs, C)]
    out = torch.log(ev.clamp_min(1e-8))
    if out.shape[1] < n_pcs:
        out = torch.cat([out, out.new_zeros(B, n_pcs - out.shape[1])], dim=1)
    offdiag = cov - torch.diag_embed(torch.diagonal(cov, dim1=-2, dim2=-1))
    return torch.cat([out, offdiag.abs().mean(dim=(1, 2), keepdim=False).reshape(B, 1)], dim=1)


class SummaryNet(nn.Module):
    """Observation window -> fixed-length summary, exchangeable over channels.

    Channels are pooled by mean+max+std so a 64-channel EEG montage and a
    454-parcel simulated field both produce the same summary shape; that is what
    lets the same posterior be applied to real data and simulator output while
    the *source card* -- not the tensor shape -- decides what the result licenses.
    """

    def __init__(self, cfg, *, fs: float = 125.0) -> None:
        super().__init__()
        C = cfg.summary_channels
        self.fs = float(fs)
        self.n_pcs = cfg.n_pcs
        layers: list[nn.Module] = [nn.Conv1d(1, C, 7, padding=3), nn.GELU()]
        for i in range(cfg.summary_layers - 1):
            layers += [nn.Conv1d(C, C, 5, padding=2 * 2**i, dilation=2**i), nn.GELU()]
        self.conv = nn.Sequential(*layers)
        n_spec = len(BAND_EDGES) + 3
        self.spec_proj = nn.Sequential(nn.Linear(n_spec, C), nn.GELU(), nn.Linear(C, C))
        self.cov_proj = nn.Sequential(nn.Linear(cfg.n_pcs + 1, C), nn.GELU(), nn.Linear(C, C))
        # +2: the log of the amplitude removed by normalisation, and the log of
        # the cross-channel amplitude spread. Normalising the window away without
        # keeping its scale would destroy exactly the information the noise-level
        # and drive parameters live in, and the posterior would then report a
        # confident, wrong marginal for them.
        self.out = nn.Sequential(nn.Linear(3 * C + C + C + 2, 2 * C), nn.GELU(), nn.Linear(2 * C, C))
        self.dim = C

    def forward(self, y: Tensor) -> Tensor:
        """``y (B,T,C)`` -> ``(B, dim)``."""
        B, T, C = y.shape
        x = y - y.mean(1, keepdim=True)
        amp = x.std(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        spread = (x.std(dim=1).std(dim=1, keepdim=True) / amp.squeeze(-1)).clamp_min(1e-8)
        x = x / amp
        h = self.conv(x.permute(0, 2, 1).reshape(B * C, 1, T))  # (B*C, K, T)
        h = h.mean(-1).reshape(B, C, -1)
        pooled = torch.cat([h.mean(1), h.amax(1), h.std(1)], dim=-1)
        spec = self.spec_proj(spectral_summary(x, self.fs)).mean(1)
        cov = self.cov_proj(covariance_summary(x, self.n_pcs).to(x.dtype))
        scale = torch.cat([amp.reshape(B, 1).log(), spread.reshape(B, 1).log()], dim=-1)
        return self.out(torch.cat([pooled, spec, cov, scale], dim=-1))


# ======================================================================
# conditional normalizing flow
# ======================================================================
class _Coupling(nn.Module):
    def __init__(self, dim: int, cond: int, hidden: int, mask: Tensor) -> None:
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = nn.Sequential(
            nn.Linear(dim + cond, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2 * dim)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: Tensor, c: Tensor) -> tuple[Tensor, Tensor]:
        m = self.mask.to(z.dtype)
        s, t = self.net(torch.cat([z * m, c], dim=-1)).chunk(2, dim=-1)
        s = torch.tanh(s) * 2.5 * (1 - m)
        t = t * (1 - m)
        return z * torch.exp(s) + t, s.sum(-1)

    def inverse(self, x: Tensor, c: Tensor) -> tuple[Tensor, Tensor]:
        """Returns ``(z, log|det dz/dx|)``.  The masked half passes through, so the
        conditioner sees exactly the same input as in the forward direction."""
        m = self.mask.to(x.dtype)
        s, t = self.net(torch.cat([x * m, c], dim=-1)).chunk(2, dim=-1)
        s = torch.tanh(s) * 2.5 * (1 - m)
        t = t * (1 - m)
        return (x - t) * torch.exp(-s), -s.sum(-1)


class ConditionalFlow(nn.Module):
    """Conditional RealNVP with exact log-density in an unconstrained space."""

    def __init__(self, dim: int, cond_dim: int, *, n_layers: int = 6, hidden: int = 256) -> None:
        super().__init__()
        self.dim = dim
        masks = []
        for i in range(n_layers):
            m = torch.zeros(dim)
            m[i % 2 :: 2] = 1.0
            if i >= 2 and dim > 2:
                perm = torch.randperm(dim, generator=torch.Generator().manual_seed(i))
                m = torch.zeros(dim)
                m[perm[: dim // 2]] = 1.0
            masks.append(m)
        self.layers = nn.ModuleList(_Coupling(dim, cond_dim, hidden, m) for m in masks)
        # Normalise the conditioning before it reaches any coupling net.
        #
        # The coupling *scale* is tanh-bounded to +-2.5, but its *translation*
        # is not: t = net([z*m, c]) is linear in c far from the origin.  Run 2's
        # pilot fed c with |c|max ~ 13-18 (against ~0.4 for the same network on
        # unmasked corpus data), which drove |z| large enough that the base term
        # -0.5*z^2 reached -4.5e9 and every batch was rejected -- from step 1, on
        # every run, bit-identically.  Measured, not inferred: the rejection dump
        # reports u|max = 3.96 and finite, so the flow's *input* was never the
        # problem; the conditioning was.
        #
        # LayerNorm makes the flow conditioning-scale invariant without changing
        # what it can represent, and it is applied in log_prob and sample alike
        # so density and samples cannot disagree.
        self.cond_norm = nn.LayerNorm(cond_dim)

    def log_prob(self, theta_u: Tensor, c: Tensor) -> Tensor:
        """``theta_u`` in unconstrained space -> log q(theta_u | c)."""
        c = self.cond_norm(c)
        z = theta_u
        ld = torch.zeros(theta_u.shape[0], device=theta_u.device, dtype=theta_u.dtype)
        for layer in reversed(self.layers):
            z, d = layer.inverse(z, c)
            ld = ld + d
        base = -0.5 * (z**2).sum(-1) - 0.5 * self.dim * math.log(2 * math.pi)
        return base + ld

    def sample(self, c: Tensor, n: int = 1) -> Tensor:
        c = self.cond_norm(c)
        B = c.shape[0]
        z = torch.randn(B * n, self.dim, device=c.device, dtype=c.dtype)
        cc = c.repeat_interleave(n, dim=0)
        for layer in self.layers:
            z, _ = layer(z, cc)
        return z.reshape(B, n, self.dim)


class AmortizedPosterior(nn.Module):
    """``q(theta | Y)`` over global coupling, velocity, regional E/I, nuisance.

    ``theta`` is carried in a **squashed unconstrained space**
    ``u = atanh(normalise(theta))`` so the flow is unbounded while the prior
    support is respected exactly; ``log_prob`` returns the density in the
    original parameterisation (the change of variables is included, not ignored).
    """

    def __init__(self, cfg, theta_dim: int, *, prior=None, fs: float = 125.0, nuisance_dim: int = 0) -> None:
        super().__init__()
        self.summary = SummaryNet(cfg, fs=fs)
        self.theta_dim = theta_dim
        self.nuisance_dim = int(nuisance_dim)
        self.total_dim = theta_dim + self.nuisance_dim
        self.flow = ConditionalFlow(self.total_dim, self.summary.dim, n_layers=cfg.flow_layers, hidden=cfg.flow_hidden)
        self.prior = prior
        self._calibrated = False
        self._calibration_evidence: dict[str, Any] = {}

    # -- parameter space --------------------------------------------------
    def to_unconstrained(self, theta: Tensor) -> tuple[Tensor, Tensor]:
        """theta -> (u, log|d theta / d u|^{-1}) so densities can be converted."""
        if self.prior is None:
            return theta, torch.zeros(theta.shape[0], device=theta.device)
        z = self.prior.normalise(theta[:, : self.theta_dim]).clamp(-1 + 1e-5, 1 - 1e-5)
        u = torch.atanh(z)
        b = self.prior.bounds().to(theta.device)
        width = (b[:, 1] - b[:, 0]).reshape(1, -1)
        # theta = b0 + (tanh(u)+1)/2 * width  ->  dtheta/du = width/2 * (1-tanh^2 u)
        logdet = (torch.log(width / 2) + torch.log((1 - z**2).clamp_min(1e-12))).sum(-1)
        if self.nuisance_dim:
            u = torch.cat([u, theta[:, self.theta_dim :]], dim=-1)
        return u, logdet

    def to_theta(self, u: Tensor) -> Tensor:
        if self.prior is None:
            return u
        th = self.prior.denormalise(torch.tanh(u[..., : self.theta_dim]))
        return torch.cat([th, u[..., self.theta_dim :]], dim=-1) if self.nuisance_dim else th

    # -- density ----------------------------------------------------------
    def log_prob(self, y: Tensor, theta: Tensor, *, in_theta_space: bool = True) -> Tensor:
        c = self.summary(y)
        u, logdet = self.to_unconstrained(theta)
        lp = self.flow.log_prob(u.float(), c.float())
        return lp - logdet.float() if in_theta_space else lp

    #: Per-sample ``-log q`` beyond which a batch is treated as pathological.
    #: The flow's coupling scale is ``tanh``-bounded to +-2.5, so with
    #: ``n_layers`` layers over ``dim`` dimensions the log-determinant cannot
    #: exceed ``2.5 * n_layers * dim``; the base term is bounded by the
    #: ``atanh`` clamp in :meth:`to_unconstrained`.  A value far outside that
    #: envelope is a degenerate batch, not a hard example.
    NPE_REJECT_ABOVE: float = 1e4

    #: How many batches :meth:`loss` has rejected, and the largest value seen.
    #: Read by the trainer -- a rejection *rate* is the diagnostic; a boolean
    #: "did it ever fire" is not.
    npe_rejected: int = 0
    npe_seen_max: float = 0.0

    def loss(self, y: Tensor, theta: Tensor) -> Tensor:
        """NPE objective: ``-E log q(theta | Y)`` in unconstrained space.

        Rejects a batch whose per-sample ``-log q`` leaves the envelope the
        flow's own bounds imply.  Run 2's first pilot trained stably for 140
        steps and then jumped seven orders of magnitude in one step, which is
        the signature of a single degenerate batch rather than a diverging
        optimiser -- the gradient was already clipped at 1.0.  Rejecting keeps
        that batch from destroying the run **and counts it**, so the rate is
        measurable instead of the failure being invisible until it is fatal.
        """
        c = self.summary(y)
        u, _ = self.to_unconstrained(theta)
        per = -self.flow.log_prob(u.float(), c.float())
        finite = torch.isfinite(per)
        keep = finite & (per.abs() < self.NPE_REJECT_ABOVE)
        with torch.no_grad():
            m = float(per[finite].abs().max()) if bool(finite.any()) else float("inf")
            type(self).npe_seen_max = max(type(self).npe_seen_max, m)
        if (not bool(keep.all())) and type(self).npe_rejected < 3:  # only real rejections
            with torch.no_grad():
                print(
                    f"[npe] REJ#{type(self).npe_rejected} all={not bool(keep.any())} "
                    f"kept={int(keep.sum())}/{keep.numel()} "
                    f"per={[round(float(v), 3) for v in per[:6]]} "
                    f"y|max={float(y.abs().max()):.4g} c|max={float(c.abs().max()):.4g} "
                    f"u|max={float(u.abs().max()):.4g} u_fin={bool(torch.isfinite(u).all())} "
                    f"th|max={float(theta.abs().max()):.4g}",
                    flush=True,
                )
        if not bool(keep.any()):
            if False:  # superseded by the dump above
                with torch.no_grad():
                    print(
                        f"[npe] FIRST REJECTION  per[:4]={[round(float(v), 4) for v in per[:4]]}  "
                        f"y|max={float(y.abs().max()):.4g} y|mean={float(y.mean()):.4g} "
                        f"c|max={float(c.abs().max()):.4g} c_finite={bool(torch.isfinite(c).all())} "
                        f"u|max={float(u.abs().max()):.4g} u_finite={bool(torch.isfinite(u).all())} "
                        f"theta|max={float(theta.abs().max()):.4g} theta[0]={[round(float(v), 4) for v in theta[0]]}",
                        flush=True,
                    )
            type(self).npe_rejected += 1
            return per.new_zeros(())
        if not bool(keep.all()):
            type(self).npe_rejected += 1
        return per[keep].mean()

    @torch.no_grad()
    def sample(self, y: Tensor, n: int = 512) -> Tensor:
        c = self.summary(y)
        u = self.flow.sample(c.float(), n)
        return self.to_theta(u)

    # -- calibration status ----------------------------------------------
    def mark_calibrated(self, evidence: dict[str, Any]) -> None:
        """Only SBC + coverage evidence may set this flag (guards R09)."""
        need = {"sbc_ks_pvalue", "coverage_mae"}
        if not need.issubset(evidence):
            raise R09Violation(
                f"a posterior may be marked calibrated only with {sorted(need)}; got {sorted(evidence)}"
            )
        self._calibrated = True
        self._calibration_evidence = dict(evidence)

    @property
    def calibrated(self) -> bool:
        return self._calibrated


# ======================================================================
# calibration diagnostics
# ======================================================================
@torch.no_grad()
def sbc_ranks(
    posterior: AmortizedPosterior,
    y: Tensor,
    theta_true: Tensor,
    *,
    n_samples: int = 256,
    chunk: int = 64,
) -> Tensor:
    """Simulation-based calibration ranks, ``(n_datasets, n_params)`` in ``[0, L]``.

    Under a correctly calibrated posterior these ranks are **uniform**.  A
    U-shape means over-confidence, an inverted-U means under-confidence, and a
    shifted histogram means bias.  Nothing about the training loss can substitute
    for this plot.
    """
    ranks = []
    for i in range(0, y.shape[0], chunk):
        yb = y[i : i + chunk]
        tb = theta_true[i : i + chunk]
        s = posterior.sample(yb, n_samples)  # (b, n, P)
        p = min(tb.shape[1], s.shape[-1])
        ranks.append((s[..., :p] < tb[:, None, :p]).sum(1))
    return torch.cat(ranks, 0)


@torch.no_grad()
def expected_coverage(
    posterior: AmortizedPosterior,
    y: Tensor,
    theta_true: Tensor,
    *,
    n_samples: int = 256,
    levels: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
    chunk: int = 64,
) -> dict[str, Any]:
    """Expected-coverage curve: nominal credible level vs realised frequency.

    Computed per parameter from central credible intervals of the marginal, plus
    a joint highest-density-region proxy from the posterior log-density rank.
    """
    lo_hits: list[Tensor] = []
    joint_hits: list[Tensor] = []
    for i in range(0, y.shape[0], chunk):
        yb, tb = y[i : i + chunk], theta_true[i : i + chunk]
        s = posterior.sample(yb, n_samples)
        p = min(tb.shape[1], s.shape[-1])
        s = s[..., :p]
        t = tb[:, None, :p]
        # marginal central intervals
        hits = []
        for lv in levels:
            a = (1 - lv) / 2
            lo = torch.quantile(s, a, dim=1)
            hi = torch.quantile(s, 1 - a, dim=1)
            hits.append(((tb[:, :p] >= lo) & (tb[:, :p] <= hi)).float())
        lo_hits.append(torch.stack(hits, 0))  # (L, b, P)
        # joint: fraction of samples closer (in whitened distance) than the truth
        mu = s.mean(1, keepdim=True)
        sd = s.std(1, keepdim=True).clamp_min(1e-8)
        ds = (((s - mu) / sd) ** 2).sum(-1)
        dt = (((t - mu) / sd) ** 2).sum(-1).squeeze(-1)
        joint_hits.append((ds < dt.unsqueeze(1)).float().mean(1))
    H = torch.cat(lo_hits, 1)  # (L, n, P)
    cov = H.mean(1)  # (L, P)
    jq = torch.cat(joint_hits, 0)
    joint = torch.stack([(jq <= lv).float().mean() for lv in levels])
    nominal = torch.tensor(list(levels))
    return {
        "levels": [float(x) for x in nominal],
        "coverage_per_param": cov.cpu().tolist(),
        "coverage_mean": cov.mean(-1).cpu().tolist(),
        "coverage_joint": joint.cpu().tolist(),
        "coverage_mae": float((cov.mean(-1).cpu() - nominal).abs().mean()),
        "coverage_mae_joint": float((joint.cpu() - nominal).abs().mean()),
        "n_datasets": int(y.shape[0]),
        "n_posterior_samples": int(n_samples),
    }


def _ks_uniform_pvalue(ranks: np.ndarray, n_bins: int) -> float:
    """Two-sided KS test of rank uniformity (asymptotic p-value)."""
    x = np.sort((ranks + 0.5) / (n_bins + 1.0))
    n = len(x)
    if n == 0:
        return float("nan")
    d = max(np.max(np.arange(1, n + 1) / n - x), np.max(x - np.arange(n) / n))
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    s = sum((-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 101))
    return float(min(max(2 * s, 0.0), 1.0))


def posterior_report(
    posterior: AmortizedPosterior,
    y: Tensor,
    theta_true: Tensor,
    *,
    param_names: Sequence[str],
    n_samples: int = 256,
) -> dict[str, Any]:
    """Full calibration report: SBC ranks, KS p-values, coverage, recovery.

    Returns machine-readable results for agent H / agent J.  It deliberately does
    **not** decide whether the posterior passes: that is a claim gate, and a gate
    is not graded by the module it grades.
    """
    ranks = sbc_ranks(posterior, y, theta_true, n_samples=n_samples).cpu().numpy()
    cov = expected_coverage(posterior, y, theta_true, n_samples=n_samples)
    P = ranks.shape[1]
    ks = [_ks_uniform_pvalue(ranks[:, j], n_samples) for j in range(P)]
    with torch.no_grad():
        s = torch.cat([posterior.sample(y[i : i + 64], 128) for i in range(0, y.shape[0], 64)], 0)
    post_mean = s.mean(1)[:, :P].cpu().numpy()
    post_sd = s.std(1)[:, :P].cpu().numpy()
    truth = theta_true[:, :P].cpu().numpy()
    r2, rmse, z_sd = [], [], []
    for j in range(P):
        v = truth[:, j].var()
        r2.append(float(1 - ((post_mean[:, j] - truth[:, j]) ** 2).mean() / max(v, 1e-12)))
        rmse.append(float(np.sqrt(((post_mean[:, j] - truth[:, j]) ** 2).mean())))
        z_sd.append(float(((post_mean[:, j] - truth[:, j]) / np.maximum(post_sd[:, j], 1e-8)).std()))
    return {
        "param_names": list(param_names)[:P],
        "sbc_ranks": ranks.tolist(),
        "sbc_n_bins": int(n_samples),
        "sbc_ks_pvalue": ks,
        "sbc_ks_pvalue_min": float(np.min(ks)) if ks else float("nan"),
        "posterior_r2": r2,
        "posterior_rmse": rmse,
        "posterior_z_sd": z_sd,  # ~1.0 iff the reported width matches the error
        "coverage_mae": cov["coverage_mae"],
        "coverage": cov,
        "note": (
            "Calibration measured against the SAME simulator that generated the "
            "training corpus. It certifies self-consistency of the amortized "
            "posterior under simulator-conditioned evidence only, and is NOT "
            "evidence of biological validity or of correct inference on real data."
        ),
    }
