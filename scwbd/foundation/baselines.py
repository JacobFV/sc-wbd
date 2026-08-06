"""Falsification baselines for the SC-WBD-001-beta held-out predictive claim.

Appendix D of the thesis (``paper/appendix.tex`` §"Heterogeneous Mixture,
Leakage, and Evaluation Protocol") requires that *every* ambitious claim ship
with a control capable of falsifying a shortcut explanation.  The foundation
model's headline claim -- that a compiled whole-brain operator with connectome
structure and structured per-region state predicts *held-out* real EEG better
than the alternatives -- has exactly four shortcut explanations, and this module
implements one control for each:

======================================  =====================================
shortcut explanation                    control that would expose it
======================================  =====================================
"the signal is autocorrelated"          :class:`PersistenceBaseline`,
                                        :class:`ARBaseline`
"the channels are linearly coupled"     :class:`VARBaseline`
"the population mean/covariance is      :class:`PopulationGaussianBaseline`
 the whole story"
"any per-person fit would do"           :class:`SubjectSpecificBaseline`
"any network of that size would do"     :class:`DenseNeuralBaseline`
======================================  =====================================

The comparison is only evidence when it is run **at matched inputs and honest
accounting**:

* *Matched inputs.*  Every baseline sees the same ``context`` window and is
  asked for the same ``target`` window.  No baseline gets extra channels,
  extra history, extra montage information or the target's own samples.
* *Matched capacity.*  :class:`DenseNeuralBaseline` sizes itself to a
  requested parameter count so that "equal capacity, no connectome" is a real
  statement and not a footnote.
* *Honest variance.*  Residual variances -- which is what a Gaussian
  log-score actually pays for -- are calibrated on *held-out training windows*,
  not on the residuals the coefficients were fitted to, because in-sample
  residual variance flatters every fitted model.
* *Honest intervals.*  Windows from one participant are correlated, so the
  bootstrap resamples **participants** (:func:`bootstrap_ci`).  Resampling
  windows would shrink every interval and manufacture significance.

The held-out metric is the mean Gaussian negative log-likelihood in **nats per
channel per time-sample** for h-step-ahead prediction of ``target`` given
``context``.  All baselines predict in the *original units of the data* and
score a diagonal (per-channel, per-time-sample) Gaussian, so that the numbers
from different baselines are members of the same likelihood family and may be
subtracted.  A model that beats these baselines on a different likelihood
family, on different inputs, or with window-level intervals has not beaten
them.

Nothing in this module fits a brain model, and nothing here licenses a claim
about any individual: these are statistical controls.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

__all__ = [
    "Baseline",
    "PersistenceBaseline",
    "ARBaseline",
    "VARBaseline",
    "PopulationGaussianBaseline",
    "SubjectSpecificBaseline",
    "DenseNeuralBaseline",
    "bootstrap_ci",
    "compare",
    "render_comparison",
]

_LOG_2PI = float(math.log(2.0 * math.pi))

#: Variance floor, in squared data units.  A baseline that is allowed an
#: arbitrarily small predictive variance can win a log-score by luck on a
#: single quiet channel; the floor makes that impossible.
_VAR_FLOOR = 1e-8

#: Minimum number of held-out training windows required before residual
#: variance is calibrated out-of-sample.  Below this the estimate is noisier
#: than the in-sample bias it removes, so the baseline falls back and *says so*
#: in :meth:`Baseline.describe`.
_MIN_CALIB_WINDOWS = 8


# ---------------------------------------------------------------------------
# tensor plumbing
# ---------------------------------------------------------------------------
def _as_windows(x: Tensor, name: str, device: torch.device | None = None) -> Tensor:
    """Validate and normalise a ``(B, T, C)`` float32 window tensor."""
    if not isinstance(x, Tensor):
        x = torch.as_tensor(x)
    if x.ndim != 3:
        raise ValueError(f"{name} must be (B, T, C); got shape {tuple(x.shape)}")
    if x.shape[0] == 0 or x.shape[1] == 0 or x.shape[2] == 0:
        raise ValueError(f"{name} has an empty axis: {tuple(x.shape)}")
    if not bool(torch.isfinite(x).all()):
        # Fail here rather than inside LAPACK: a non-finite sample silently
        # poisons a normal-equations solve and can abort the process.
        n_bad = int((~torch.isfinite(x)).sum())
        raise ValueError(f"{name} contains {n_bad} non-finite values")
    out = x.to(dtype=torch.float32)
    if device is not None:
        out = out.to(device)
    return out.contiguous()


def _as_groups(groups: Sequence[Any] | np.ndarray | None, n: int, name: str) -> np.ndarray | None:
    """Normalise participant ids to a length-``n`` object array (or ``None``)."""
    if groups is None:
        return None
    g = np.asarray(groups)
    if g.ndim != 1 or g.shape[0] != n:
        raise ValueError(f"{name} must be 1-D of length {n}; got shape {g.shape}")
    return g


def _split_indices(n: int, val_fraction: float, seed: int) -> tuple[Tensor, Tensor]:
    """Deterministic (fit, calibration) window split.

    The split is over *windows*, not participants: its only job is to keep the
    residual-variance estimate out of the sample the coefficients were fitted
    on.  Windows from the same participant remain correlated across the split,
    so this removes in-sample optimism but does not make the variance estimate
    independent -- which is why the reported intervals are still clustered by
    participant.
    """
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(int(seed)))
    n_val = int(round(val_fraction * n))
    if n_val < _MIN_CALIB_WINDOWS or n - n_val < _MIN_CALIB_WINDOWS:
        return perm, perm  # too small to split honestly; caller flags it
    return perm[n_val:], perm[:n_val]


def _ridge_solve(gram: Tensor, rhs: Tensor, ridge: float) -> Tensor:
    """Solve ``(G + lambda I) W = R`` in float64 with a least-squares fallback.

    ``ridge`` is *relative*: the penalty is ``ridge * mean(diag(G))``, so the
    same value means the same thing whether the data are in volts or in
    microvolts.  Scale-dependent regularisation is a silent way to give one
    baseline an advantage over another.
    """
    gram = gram.to(torch.float64)
    rhs = rhs.to(torch.float64)
    d = gram.shape[-1]
    scale = torch.diagonal(gram, dim1=-2, dim2=-1).mean(-1).clamp_min(1e-12)
    lam = float(ridge) * scale
    eye = torch.eye(d, dtype=gram.dtype, device=gram.device)
    a = gram + lam[..., None, None] * eye
    try:
        chol = torch.linalg.cholesky(a)
        return torch.cholesky_solve(rhs, chol)
    except (torch.linalg.LinAlgError, RuntimeError):
        sol = torch.linalg.lstsq(a.cpu(), rhs.cpu(), driver="gelsd").solution
        return sol.to(gram.device)


def _iter_lagged(windows: Tensor, order: int, chunk: int):
    """Yield ``(Z, Y)`` lag/target pairs in window chunks.

    ``Z`` is ``(N, order, C)`` with ``Z[:, j - 1]`` the sample at lag ``j``;
    ``Y`` is ``(N, C)``.  Chunked so that a 64-channel, order-32 fit does not
    materialise the whole design matrix.
    """
    b, t, c = windows.shape
    if t <= order:
        raise ValueError(f"window length {t} must exceed AR order {order}")
    for s in range(0, b, chunk):
        w = windows[s : s + chunk]
        y = w[:, order:, :]
        z = torch.stack([w[:, order - j : t - j, :] for j in range(1, order + 1)], dim=2)
        yield z.reshape(-1, order, c), y.reshape(-1, c)


def _chunk_for(order: int, channels: int) -> int:
    """Window-chunk size keeping the materialised design under ~16M floats."""
    per_window = max(1, order * channels)
    return max(1, min(4096, (1 << 24) // per_window))


def _gaussian_nll(mean: Tensor, log_var: Tensor, target: Tensor) -> Tensor:
    """Element-wise Gaussian NLL in nats, shape ``(B, T_pred, C)``."""
    var = log_var.exp().clamp_min(_VAR_FLOOR)
    return 0.5 * (_LOG_2PI + var.log() + (target - mean).pow(2) / var)


# ---------------------------------------------------------------------------
# the interface
# ---------------------------------------------------------------------------
class Baseline(ABC):
    """A fitted control that predicts a window given a context window.

    Subclasses exist to make a *specific* shortcut explanation testable.  A
    subclass that cannot state what its own existence would falsify does not
    belong in this module -- hence :meth:`describe` is abstract-by-convention
    and carries ``falsifies_comparison_if``.

    Data convention
    ---------------
    Windows are ``(B, T, C)`` float32 tensors: batch, time, channel.  A model
    is fitted on ``train_windows`` (full windows) and scored by
    :meth:`score(context, target)` where ``context`` is ``(B, T_ctx, C)`` and
    ``target`` is ``(B, T_pred, C)``.  Horizon ``h`` runs ``1..T_pred``, so
    ``score`` is simultaneously the one-step-ahead and h-step-ahead metric and
    :meth:`score` reports the per-horizon breakdown.
    """

    #: Human-readable, stable across runs; used as the table key in `compare`.
    name: str = "baseline"

    def __init__(self, *, device: str | torch.device | None = None, seed: int = 0) -> None:
        self.seed = int(seed)
        self.device = torch.device(device) if device is not None else None
        self.fitted: bool = False
        self._n_train_windows: int = 0
        self._n_train_subjects: int | None = None
        self._variance_calibration: str = "unfitted"

    # -- required surface ---------------------------------------------------
    @abstractmethod
    def fit(
        self,
        train_windows: Tensor,
        groups: Sequence[Any] | np.ndarray | None = None,
    ) -> Baseline:
        """Fit on training windows; returns ``self``.

        ``groups`` holds one participant id per window.  Most baselines ignore
        it (that is the point of them); :class:`SubjectSpecificBaseline`
        requires it.
        """

    @abstractmethod
    def predict(
        self,
        context: Tensor,
        horizon: int,
        *,
        groups: Sequence[Any] | np.ndarray | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(mean, log_var)``, both ``(B, horizon, C)``, in data units."""

    @abstractmethod
    def n_parameters(self) -> int:
        """Free parameters actually fitted from data (variances included).

        Counted honestly: a baseline that hides capacity in "preprocessing"
        would make a matched-capacity claim false.  For baselines whose
        predictive variance is a horizon-resolved table, the table is sized by
        the first :meth:`score` call, so quote this number *after* scoring.
        """

    def describe(self) -> dict[str, Any]:
        """What this baseline is, assumes, and would falsify.

        Returned keys deliberately mirror ``scwbd.bench.report.BaselineResult``
        so a run can be dropped straight into a claim report.
        """
        return {
            "name": self.name,
            "n_parameters": self.n_parameters() if self.fitted else None,
            "fitted": self.fitted,
            "assumptions": self._assumptions(),
            "falsifies_comparison_if": self._falsifies_comparison_if(),
            "n_train_windows": self._n_train_windows,
            "n_train_subjects": self._n_train_subjects,
            "variance_calibration": self._variance_calibration,
            "seed": self.seed,
        }

    @abstractmethod
    def _assumptions(self) -> list[str]:
        """Assumptions whose violation would make this control uninformative."""

    @abstractmethod
    def _falsifies_comparison_if(self) -> str:
        """The sentence this baseline exists to make refutable."""

    # -- shared scoring -----------------------------------------------------
    def score(
        self,
        context: Tensor,
        target: Tensor,
        groups: Sequence[Any] | np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Held-out Gaussian log-score of ``target`` given ``context``.

        Returns a dict containing the headline scalars *and* the per-window
        arrays, because a point estimate without the units of the bootstrap is
        not reportable: :func:`bootstrap_ci` needs ``per_window_nll`` and the
        matching participant ids.

        Keys
        ----
        ``nll_per_sample``
            Mean Gaussian NLL in nats per channel per time-sample (the
            headline metric; lower is better).
        ``mse``
            Mean squared error in squared data units.
        ``n``
            Number of scored **windows** -- the unit the bootstrap resamples
            within participants.  ``n_elements`` is the sample count.
        ``per_window_nll`` / ``per_window_mse``
            ``(B,)`` float64 arrays, one value per window.
        ``nll_per_horizon``
            Length-``T_pred`` list: how the score degrades with horizon.
        """
        if not self.fitted:
            raise RuntimeError(f"{self.name}: score() before fit()")
        ctx = _as_windows(context, "context", self.device)
        tgt = _as_windows(target, "target", self.device)
        if ctx.shape[0] != tgt.shape[0] or ctx.shape[2] != tgt.shape[2]:
            raise ValueError(
                f"{self.name}: context {tuple(ctx.shape)} and target {tuple(tgt.shape)} "
                "must share batch and channel axes"
            )
        g = _as_groups(groups, ctx.shape[0], "groups")
        horizon = int(tgt.shape[1])
        with torch.no_grad():
            mean, log_var = self.predict(ctx, horizon, groups=g)
            if mean.shape != tgt.shape or log_var.shape != tgt.shape:
                raise RuntimeError(
                    f"{self.name}: predict returned {tuple(mean.shape)} for target "
                    f"{tuple(tgt.shape)}"
                )
            nll = _gaussian_nll(mean, log_var, tgt)
            sq = (mean - tgt).pow(2)
            per_window_nll = nll.mean(dim=(1, 2)).double().cpu().numpy()
            per_window_mse = sq.mean(dim=(1, 2)).double().cpu().numpy()
            per_horizon = nll.mean(dim=(0, 2)).double().cpu().numpy()
        return {
            "model": self.name,
            "nll_per_sample": float(per_window_nll.mean()),
            "mse": float(per_window_mse.mean()),
            "n": int(ctx.shape[0]),
            "n_elements": int(tgt.numel()),
            "horizon": horizon,
            "per_window_nll": per_window_nll,
            "per_window_mse": per_window_mse,
            "nll_per_horizon": [float(v) for v in per_horizon],
            "n_parameters": self.n_parameters(),
            "units": "nats per channel per time-sample",
        }

    # -- helpers ------------------------------------------------------------
    def _prepare(
        self,
        train_windows: Tensor,
        groups: Sequence[Any] | np.ndarray | None,
    ) -> tuple[Tensor, np.ndarray | None]:
        w = _as_windows(train_windows, "train_windows")
        if self.device is None:
            self.device = w.device
        w = w.to(self.device)
        g = _as_groups(groups, w.shape[0], "groups")
        self._n_train_windows = int(w.shape[0])
        self._n_train_subjects = int(np.unique(g).size) if g is not None else None
        return w, g

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        state = "fitted" if self.fitted else "unfitted"
        return f"<{type(self).__name__} name={self.name!r} {state}>"


# ---------------------------------------------------------------------------
# linear forecasters with out-of-sample variance calibration
# ---------------------------------------------------------------------------
class _LinearForecaster(Baseline):
    """Shared machinery: fit a conditional mean, then calibrate its variance.

    The split of responsibilities matters for honesty.  Subclasses supply only
    a *conditional mean* through :meth:`_fit_mean` / :meth:`_predict_mean`;
    this class supplies the predictive variance by rolling the fitted mean
    forward on **held-out training windows** and taking the empirical residual
    variance per (horizon, channel).  A model scored with its own in-sample
    residual variance always looks better than it is, and the size of that
    illusion grows with the number of fitted coefficients -- exactly the
    quantity a matched-capacity comparison is trying to control.
    """

    #: Minimum context samples the mean model needs (AR order, or 1).
    _min_context: int = 1

    def __init__(
        self,
        *,
        val_fraction: float = 0.25,
        max_calib_windows: int = 1024,
        device: str | torch.device | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(device=device, seed=seed)
        self.val_fraction = float(val_fraction)
        self.max_calib_windows = int(max_calib_windows)
        self._calib_windows: Tensor | None = None
        self._var_cache: Tensor | None = None  # (H, C)

    # -- subclass hooks -----------------------------------------------------
    @abstractmethod
    def _fit_mean(self, windows: Tensor) -> None:
        """Fit the conditional-mean parameters on ``windows``."""

    @abstractmethod
    def _predict_mean(self, context: Tensor, horizon: int) -> Tensor:
        """Iterate the fitted mean ``horizon`` steps; ``(B, horizon, C)``."""

    # -- Baseline surface ---------------------------------------------------
    def fit(
        self,
        train_windows: Tensor,
        groups: Sequence[Any] | np.ndarray | None = None,
    ) -> _LinearForecaster:
        w, _ = self._prepare(train_windows, groups)
        fit_idx, calib_idx = _split_indices(w.shape[0], self.val_fraction, self.seed)
        self._variance_calibration = (
            "in_sample (too few windows to hold out; NLL is optimistic)"
            if fit_idx.numel() == calib_idx.numel() and bool(torch.equal(fit_idx, calib_idx))
            else "held_out_training_windows"
        )
        self._fit_mean(w[fit_idx.to(w.device)])
        keep = calib_idx[: self.max_calib_windows].to(w.device)
        self._calib_windows = w[keep].clone()
        self._var_cache = None
        self.fitted = True
        return self

    def predict(
        self,
        context: Tensor,
        horizon: int,
        *,
        groups: Sequence[Any] | np.ndarray | None = None,
    ) -> tuple[Tensor, Tensor]:
        ctx = _as_windows(context, "context", self.device)
        if ctx.shape[1] < self._min_context:
            raise ValueError(
                f"{self.name}: needs at least {self._min_context} context samples, "
                f"got {ctx.shape[1]}"
            )
        mean = self._predict_mean(ctx, horizon)
        var = self._horizon_variance(horizon, ctx.shape[2])
        log_var = var.clamp_min(_VAR_FLOOR).log().unsqueeze(0).expand_as(mean)
        return mean, log_var.contiguous()

    # -- variance calibration ----------------------------------------------
    def _horizon_variance(self, horizon: int, channels: int) -> Tensor:
        """Per-(horizon, channel) residual variance, cached, ``(horizon, C)``."""
        cache = self._var_cache
        if cache is not None and cache.shape[0] >= horizon and cache.shape[1] == channels:
            return cache[:horizon]
        var = self._calibrate_variance(horizon)
        self._var_cache = var
        return var[:horizon]

    def _calibrate_variance(self, horizon: int) -> Tensor:
        w = self._calib_windows
        if w is None:
            raise RuntimeError(f"{self.name}: no calibration windows retained")
        n, t, c = w.shape
        ctx_len = max(self._min_context, t - horizon)
        if ctx_len >= t:
            ctx_len = max(self._min_context, t - 1)
        h_eff = t - ctx_len
        if h_eff < 1:
            raise ValueError(
                f"{self.name}: calibration windows of length {t} are too short for a "
                f"model needing {self._min_context} context samples"
            )
        with torch.no_grad():
            mean = self._predict_mean(w[:, :ctx_len], h_eff)
            resid = w[:, ctx_len:] - mean
            var = resid.pow(2).sum(0) / max(n - 1, 1)  # (h_eff, C)
        if h_eff < horizon:
            # Beyond the calibrated horizon the residual variance can only be
            # extrapolated; holding it flat is optimistic, so it is recorded.
            pad = var[-1:].expand(horizon - h_eff, c)
            var = torch.cat([var, pad], dim=0)
            note = f" (+flat extrapolation beyond h={h_eff})"
            if note not in self._variance_calibration:
                self._variance_calibration += note
        return var.clamp_min(_VAR_FLOOR).contiguous()


class PersistenceBaseline(_LinearForecaster):
    """Predict the last observed sample, forever.  The trivial control.

    Why it exists: a forecaster of a smooth, heavily autocorrelated signal can
    post an impressive R^2 while having learned nothing but "tomorrow looks
    like today".  Persistence is the floor that turns "the model predicts EEG"
    into a claim with content.

    It still gets an *honest* predictive variance -- calibrated per horizon on
    held-out training windows -- so beating it on the log-score requires
    beating a properly-calibrated dumb model, not a straw man with an
    arbitrary variance.
    """

    name = "persistence"
    _min_context = 1

    def _fit_mean(self, windows: Tensor) -> None:
        """No conditional-mean parameters: that is the point of this control."""

    def _predict_mean(self, context: Tensor, horizon: int) -> Tensor:
        return context[:, -1:, :].expand(-1, horizon, -1).contiguous()

    def n_parameters(self) -> int:
        # Zero mean parameters; the variance table is fitted and counted.
        cache = self._var_cache
        return 0 if cache is None else int(cache.numel())

    def _assumptions(self) -> list[str]:
        return [
            "the signal is locally constant on the prediction horizon",
            "residual variance is stationary across participants and sessions",
        ]

    def _falsifies_comparison_if(self) -> str:
        return (
            "If the foundation model does not beat persistence by a margin whose "
            "participant-clustered interval excludes zero, the predictive claim is "
            "explained by autocorrelation alone and must be withdrawn."
        )


class ARBaseline(_LinearForecaster):
    """Per-channel autoregression of order ``p``, ridge-fitted on pooled data.

    Why it exists: EEG channels are individually near-oscillatory, and an AR(p)
    with p up to ~32 captures spectral peaks (alpha, mu) and their phase
    continuation.  Most of what looks like "the model learned rhythms" is
    available to a per-channel AR fitted on the same training set, with no
    connectome, no cross-channel term and no per-region state.

    Fitting: pooled ridge least squares over every ``(window, time)`` pair in
    the training split, per channel (``per_channel=True``) or with coefficients
    shared across channels (``per_channel=False``, a stricter control that
    denies the baseline any channel-specific spectrum).  h-step-ahead
    prediction iterates the recursion, feeding back the predicted mean; the
    resulting error growth is measured, not assumed.
    """

    def __init__(
        self,
        order: int = 8,
        *,
        per_channel: bool = True,
        ridge: float = 1e-4,
        val_fraction: float = 0.25,
        max_calib_windows: int = 1024,
        device: str | torch.device | None = None,
        seed: int = 0,
        name: str | None = None,
    ) -> None:
        super().__init__(
            val_fraction=val_fraction,
            max_calib_windows=max_calib_windows,
            device=device,
            seed=seed,
        )
        if order < 1:
            raise ValueError("AR order must be >= 1")
        self.order = int(order)
        self.per_channel = bool(per_channel)
        self.ridge = float(ridge)
        self._min_context = self.order
        self.name = name or f"ar{self.order}" + ("" if per_channel else "_shared")
        self.coef_: Tensor | None = None  # (C, p)
        self.intercept_: Tensor | None = None  # (C,)

    def _fit_mean(self, windows: Tensor) -> None:
        p, c = self.order, int(windows.shape[2])
        chunk = _chunk_for(p, c)
        dev = windows.device
        s_zz = torch.zeros(c, p, p, dtype=torch.float64, device=dev)
        s_zy = torch.zeros(c, p, dtype=torch.float64, device=dev)
        s_z = torch.zeros(c, p, dtype=torch.float64, device=dev)
        s_y = torch.zeros(c, dtype=torch.float64, device=dev)
        n = 0
        for z, y in _iter_lagged(windows, p, chunk):
            zc = z.permute(2, 0, 1).to(torch.float64)  # (C, n, p)
            yc = y.permute(1, 0).to(torch.float64)  # (C, n)
            s_zz += zc.transpose(1, 2) @ zc
            s_zy += (zc.transpose(1, 2) @ yc.unsqueeze(-1)).squeeze(-1)
            s_z += zc.sum(1)
            s_y += yc.sum(1)
            n += int(z.shape[0])
        if n <= p:
            raise ValueError(f"{self.name}: {n} samples for {p} coefficients")
        if not self.per_channel:
            s_zz = s_zz.sum(0, keepdim=True).expand(c, p, p).contiguous()
            s_zy = s_zy.sum(0, keepdim=True).expand(c, p).contiguous()
            s_z = s_z.sum(0, keepdim=True).expand(c, p).contiguous()
            s_y = s_y.sum(0, keepdim=True).expand(c).contiguous()
            n_eff = n * c
        else:
            n_eff = n
        gram = s_zz - torch.einsum("ci,cj->cij", s_z, s_z) / n_eff
        rhs = (s_zy - s_z * s_y.unsqueeze(-1) / n_eff).unsqueeze(-1)
        coef = _ridge_solve(gram, rhs, self.ridge).squeeze(-1)  # (C, p)
        intercept = s_y / n_eff - (s_z / n_eff * coef).sum(-1)
        self.coef_ = coef.to(torch.float32)
        self.intercept_ = intercept.to(torch.float32)

    def _predict_mean(self, context: Tensor, horizon: int) -> Tensor:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError(f"{self.name}: mean model not fitted")
        p = self.order
        hist = context[:, -p:, :].flip(1).clone()  # (B, p, C), lag 1 first
        coef = self.coef_.t()  # (p, C)
        out = []
        for _ in range(horizon):
            nxt = (hist * coef.unsqueeze(0)).sum(1) + self.intercept_  # (B, C)
            out.append(nxt)
            hist = torch.cat([nxt.unsqueeze(1), hist[:, :-1, :]], dim=1)
        return torch.stack(out, dim=1)

    def n_parameters(self) -> int:
        if self.coef_ is None:
            return 0
        mean_params = (
            int(self.coef_.numel()) + int(self.intercept_.numel())
            if self.per_channel
            else self.order + 1
        )
        var_params = 0 if self._var_cache is None else int(self._var_cache.numel())
        return mean_params + var_params

    def _assumptions(self) -> list[str]:
        return [
            "per-channel linear, stationary, Gaussian-innovation dynamics",
            "one set of coefficients for the whole population (no subject fitting)",
            "channels are conditionally independent given their own past",
            f"context is at least {self.order} samples long",
        ]

    def _falsifies_comparison_if(self) -> str:
        return (
            "If a per-channel AR fitted on the same training windows matches the "
            "foundation model's held-out NLL, then the claimed benefit of connectome "
            "structure and per-region state reduces to per-channel spectral "
            "continuation and the structural claim is unsupported."
        )


class VARBaseline(_LinearForecaster):
    """Vector autoregression: every channel from every channel's past.

    Why it exists: this is the strongest *linear* explanation of cross-channel
    structure.  If the foundation model's advantage is that it exploits the
    connectome, then a full VAR -- which is free to learn an arbitrary dense
    lagged coupling matrix from the same data -- is the control that says
    whether the coupling had to be anatomical or merely had to be *there*.

    ``p * C`` is large (order 8 over 64 channels is a 512-dimensional
    regressor), so the normal equations are formed in float64, ridge-penalised
    and solved by Cholesky, with an explicit least-squares fallback.  Setting
    ``rank=r`` additionally applies reduced-rank regression (the classical
    identity-weighted projection onto the top-``r`` subspace of the fitted
    values), which is the honest way to give the linear control a capacity
    knob when ``p * C`` approaches the sample count.
    """

    def __init__(
        self,
        order: int = 4,
        *,
        rank: int | None = None,
        ridge: float = 1e-3,
        val_fraction: float = 0.25,
        max_calib_windows: int = 1024,
        device: str | torch.device | None = None,
        seed: int = 0,
        name: str | None = None,
    ) -> None:
        super().__init__(
            val_fraction=val_fraction,
            max_calib_windows=max_calib_windows,
            device=device,
            seed=seed,
        )
        if order < 1:
            raise ValueError("VAR order must be >= 1")
        if rank is not None and rank < 1:
            raise ValueError("rank must be >= 1 or None")
        self.order = int(order)
        self.rank = None if rank is None else int(rank)
        self.ridge = float(ridge)
        self._min_context = self.order
        suffix = "" if self.rank is None else f"_r{self.rank}"
        self.name = name or f"var{self.order}{suffix}"
        self.coef_: Tensor | None = None  # (p*C, C)
        self.intercept_: Tensor | None = None  # (C,)
        self.effective_rank_: int | None = None

    def _fit_mean(self, windows: Tensor) -> None:
        p, c = self.order, int(windows.shape[2])
        d = p * c
        dev = windows.device
        chunk = _chunk_for(p, c)
        s_zz = torch.zeros(d, d, dtype=torch.float64, device=dev)
        s_zy = torch.zeros(d, c, dtype=torch.float64, device=dev)
        s_z = torch.zeros(d, dtype=torch.float64, device=dev)
        s_y = torch.zeros(c, dtype=torch.float64, device=dev)
        n = 0
        for z, y in _iter_lagged(windows, p, chunk):
            zf = z.reshape(z.shape[0], d).to(torch.float64)
            yf = y.to(torch.float64)
            s_zz += zf.t() @ zf
            s_zy += zf.t() @ yf
            s_z += zf.sum(0)
            s_y += yf.sum(0)
            n += int(zf.shape[0])
        if n <= d:
            # Not fatal -- that is what the ridge is for -- but it must be said.
            self._variance_calibration += (
                f" | warning: {n} samples for {d} VAR regressors (ridge-dominated)"
            )
        gram = s_zz - torch.outer(s_z, s_z) / n
        rhs = s_zy - torch.outer(s_z, s_y) / n
        coef = _ridge_solve(gram, rhs, self.ridge)  # (d, C)
        if self.rank is not None and self.rank < c:
            fitted_cov = coef.t() @ gram @ coef  # == Zc^T Zc of fitted values
            fitted_cov = 0.5 * (fitted_cov + fitted_cov.t())
            evals, evecs = torch.linalg.eigh(fitted_cov)
            top = evecs[:, -self.rank :]  # (C, r)
            coef = coef @ top @ top.t()
            self.effective_rank_ = int(self.rank)
        else:
            self.effective_rank_ = int(min(d, c))
        intercept = s_y / n - (s_z / n) @ coef
        self.coef_ = coef.to(torch.float32)
        self.intercept_ = intercept.to(torch.float32)

    def _predict_mean(self, context: Tensor, horizon: int) -> Tensor:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError(f"{self.name}: mean model not fitted")
        p, c = self.order, int(context.shape[2])
        hist = context[:, -p:, :].flip(1).clone()  # (B, p, C), lag 1 first
        out = []
        for _ in range(horizon):
            nxt = hist.reshape(hist.shape[0], p * c) @ self.coef_ + self.intercept_
            out.append(nxt)
            hist = torch.cat([nxt.unsqueeze(1), hist[:, :-1, :]], dim=1)
        return torch.stack(out, dim=1)

    def n_parameters(self) -> int:
        if self.coef_ is None:
            return 0
        d, c = self.coef_.shape
        if self.rank is not None and self.rank < c:
            mean_params = self.rank * (d + c)  # reduced-rank factorisation
        else:
            mean_params = d * c
        mean_params += c
        var_params = 0 if self._var_cache is None else int(self._var_cache.numel())
        return int(mean_params + var_params)

    def _assumptions(self) -> list[str]:
        return [
            "linear, stationary, dense lagged coupling between all channels",
            "one coupling matrix for the whole population (no subject fitting)",
            "ridge/rank regularisation is the right capacity control at this n",
            "sensor-space coupling stands in for source-space coupling",
        ]

    def _falsifies_comparison_if(self) -> str:
        return (
            "If a dense VAR fitted on the same windows matches the foundation model, "
            "then cross-channel structure did not have to be anatomical: an "
            "unconstrained linear coupling suffices, and the connectome-structured "
            "claim loses its evidential basis."
        )


class PopulationGaussianBaseline(_LinearForecaster):
    """Stationary population Gaussian: mean, channel covariance, AR(1) decay.

    Why it exists: this is the "you are just predicting the average brain"
    control demanded by the individualization claim (``appendix.tex``,
    tab:mixture-evaluation: *"Including a person's scan is not personalization
    unless it improves their future predictions beyond population and session
    baselines"*).  It carries **no subject information whatsoever**: its
    parameters are a population mean vector, a population channel covariance,
    and one lag-1 autocorrelation per channel (the PSD-matched temporal model,
    i.e. an AR(1) whose stationary variance and lag-1 autocovariance equal the
    population's).

    With ``condition_on_context=True`` (default, the matched-input setting) it
    relaxes from the last observed sample toward the population mean at the
    population rate: ``E[x_{t+h}] = mu + a^h (x_t - mu)`` with predictive
    variance ``diag(Sigma) (1 - a^{2h})``.  With ``condition_on_context=False``
    it ignores the participant's data entirely and emits the population
    marginal -- the absolute floor for any claim of individualization.

    The full channel covariance is fitted and exposed as ``covariance_``, but
    the reported score uses only its diagonal: every baseline in this module
    is scored under the *same* diagonal Gaussian likelihood so that the
    differences are subtractable.  Crediting this baseline for cross-channel
    covariance while scoring the others diagonally would be a rigged
    comparison in the other direction.
    """

    name = "population_gaussian"
    _min_context = 1

    def __init__(
        self,
        *,
        condition_on_context: bool = True,
        val_fraction: float = 0.25,
        max_calib_windows: int = 1024,
        device: str | torch.device | None = None,
        seed: int = 0,
        name: str | None = None,
    ) -> None:
        super().__init__(
            val_fraction=val_fraction,
            max_calib_windows=max_calib_windows,
            device=device,
            seed=seed,
        )
        self.condition_on_context = bool(condition_on_context)
        if name is not None:
            self.name = name
        elif not self.condition_on_context:
            self.name = "population_gaussian_marginal"
        self.mean_: Tensor | None = None  # (C,)
        self.covariance_: Tensor | None = None  # (C, C)
        self.ar1_: Tensor | None = None  # (C,)

    def _fit_mean(self, windows: Tensor) -> None:
        x = windows.reshape(-1, windows.shape[2]).to(torch.float64)
        mu = x.mean(0)
        xc = x - mu
        cov = (xc.t() @ xc) / max(x.shape[0] - 1, 1)
        # Empirical lag-1 autocovariance per channel, pooled over windows.
        w = windows.to(torch.float64) - mu
        num = (w[:, :-1, :] * w[:, 1:, :]).sum(dim=(0, 1))
        den = w.pow(2).sum(dim=(0, 1)).clamp_min(1e-12)
        a = (num / den).clamp(-0.999, 0.999)
        self.mean_ = mu.to(torch.float32)
        self.covariance_ = cov.to(torch.float32)
        self.ar1_ = a.to(torch.float32)

    def _predict_mean(self, context: Tensor, horizon: int) -> Tensor:
        if self.mean_ is None or self.ar1_ is None:
            raise RuntimeError(f"{self.name}: mean model not fitted")
        b = context.shape[0]
        mu = self.mean_.unsqueeze(0).unsqueeze(0)  # (1,1,C)
        if not self.condition_on_context:
            return mu.expand(b, horizon, -1).contiguous()
        last = context[:, -1:, :]  # (B,1,C)
        h = torch.arange(1, horizon + 1, device=context.device, dtype=torch.float32)
        decay = self.ar1_.unsqueeze(0).pow(h.unsqueeze(1))  # (H, C)
        return mu + decay.unsqueeze(0) * (last - mu)

    def n_parameters(self) -> int:
        if self.mean_ is None or self.covariance_ is None:
            return 0
        c = int(self.mean_.numel())
        # mean + full channel covariance (fitted, reported) + per-channel AR(1)
        return c + c * (c + 1) // 2 + c

    def _assumptions(self) -> list[str]:
        return [
            "one stationary Gaussian describes the whole population",
            "temporal structure is exhausted by a per-channel lag-1 autocovariance",
            "no participant identity, session, montage or anatomy is used",
        ]

    def _falsifies_comparison_if(self) -> str:
        return (
            "If the foundation model does not beat the population Gaussian on a "
            "participant's held-out future, then including that participant's data "
            "was not personalization, and no individualization claim may be made."
        )


# ---------------------------------------------------------------------------
# subject-specific control
# ---------------------------------------------------------------------------
class SubjectSpecificBaseline(Baseline):
    """One independent statistical model per participant.

    Why it exists: this is the baseline the thesis demands and the hardest one
    to beat.  A foundation model trained across many participants claims
    transfer -- that what it learned from *other* people improves *this*
    person's future.  The direct competitor is the model that ignores everyone
    else and fits this person alone.  If the per-participant AR wins, the
    population pretraining bought nothing on this port and the "foundation"
    framing is not supported by the data, however good the aggregate number is.

    Each participant's model is fitted on that participant's *training* windows
    only and scores only that participant's held-out windows; ``groups`` must
    be supplied at both fit and score time.  Participants with fewer than
    ``min_windows`` training windows fall back to a pooled model fitted on all
    training data, and the fallback set is reported in :meth:`describe` --
    silently pooling a thin participant into the population model would let a
    "subject-specific" baseline quietly become the population baseline.
    """

    def __init__(
        self,
        base: type[Baseline] | Callable[..., Baseline] = ARBaseline,
        *,
        base_kwargs: Mapping[str, Any] | None = None,
        min_windows: int = 16,
        device: str | torch.device | None = None,
        seed: int = 0,
        name: str | None = None,
    ) -> None:
        super().__init__(device=device, seed=seed)
        self.base = base
        self.base_kwargs = dict(base_kwargs or {})
        self.min_windows = int(min_windows)
        probe = self._new_base()
        self.name = name or f"subject_specific[{probe.name}]"
        self.models_: dict[Any, Baseline] = {}
        self.fallback_: Baseline | None = None
        self.fallback_subjects_: list[Any] = []

    def _new_base(self) -> Baseline:
        kwargs = dict(self.base_kwargs)
        kwargs.setdefault("seed", self.seed)
        if self.device is not None:
            kwargs.setdefault("device", self.device)
        return self.base(**kwargs)

    def fit(
        self,
        train_windows: Tensor,
        groups: Sequence[Any] | np.ndarray | None = None,
    ) -> SubjectSpecificBaseline:
        w, g = self._prepare(train_windows, groups)
        if g is None:
            raise ValueError(
                f"{self.name}: groups (one participant id per training window) are "
                "required; a subject-specific baseline without subject ids is a "
                "population baseline wearing its name"
            )
        self.fallback_ = self._new_base()
        self.fallback_.fit(w, g)
        self.models_ = {}
        self.fallback_subjects_ = []
        for subject in np.unique(g):
            idx = np.flatnonzero(g == subject)
            if idx.size < self.min_windows:
                self.fallback_subjects_.append(subject)
                continue
            sel = torch.as_tensor(idx, dtype=torch.long, device=w.device)
            model = self._new_base()
            model.fit(w[sel])
            self.models_[subject] = model
        cal = {m._variance_calibration for m in self.models_.values()} or {"none"}
        self._variance_calibration = "per-subject: " + "; ".join(sorted(cal))
        self.fitted = True
        return self

    def predict(
        self,
        context: Tensor,
        horizon: int,
        *,
        groups: Sequence[Any] | np.ndarray | None = None,
    ) -> tuple[Tensor, Tensor]:
        ctx = _as_windows(context, "context", self.device)
        g = _as_groups(groups, ctx.shape[0], "groups")
        if g is None:
            raise ValueError(
                f"{self.name}: groups are required at score time to route each "
                "window to its participant's model"
            )
        if self.fallback_ is None:
            raise RuntimeError(f"{self.name}: not fitted")
        b, _, c = ctx.shape
        mean = torch.empty(b, horizon, c, device=ctx.device, dtype=torch.float32)
        log_var = torch.empty_like(mean)
        for subject in np.unique(g):
            idx = torch.as_tensor(np.flatnonzero(g == subject), dtype=torch.long, device=ctx.device)
            model = self.models_.get(subject, self.fallback_)
            m, lv = model.predict(ctx[idx], horizon)
            mean[idx] = m
            log_var[idx] = lv
        return mean, log_var

    def n_parameters(self) -> int:
        total = sum(m.n_parameters() for m in self.models_.values())
        if self.fallback_ is not None and self.fallback_subjects_:
            total += self.fallback_.n_parameters()
        return int(total)

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d["n_subject_models"] = len(self.models_)
        d["fallback_subjects"] = [str(s) for s in self.fallback_subjects_]
        d["parameters_per_subject"] = (
            int(self.n_parameters() / max(len(self.models_), 1)) if self.models_ else 0
        )
        d["note"] = (
            "n_parameters is the SUM over participants: this baseline is not "
            "capacity-matched to a single shared model and must not be presented "
            "as if it were. It is reported so the asymmetry is visible."
        )
        return d

    def _assumptions(self) -> list[str]:
        return [
            "each participant has enough training windows to fit their own model",
            "the participant's training and held-out windows are exchangeable "
            "(same session-level nonstationarity would flatter this baseline)",
            "participant identity is known at test time",
        ]

    def _falsifies_comparison_if(self) -> str:
        return (
            "If a model fitted on a participant alone predicts that participant's "
            "held-out future as well as the population-pretrained foundation model, "
            "then cross-participant pretraining transferred nothing on this port and "
            "the foundation claim fails regardless of the aggregate score."
        )


# ---------------------------------------------------------------------------
# capacity-matched neural control
# ---------------------------------------------------------------------------
class _GRUForecaster(nn.Module):
    """Multi-layer GRU over the flat channel vector, heteroscedastic head.

    Deliberately structureless: no connectome, no adjacency, no per-region
    state, no spatial prior.  The channel vector is one flat token.  It
    predicts the *increment* to the current sample plus a per-channel
    log-variance, which is the standard strong dense forecaster and denies the
    comparison the "the baseline was badly parameterised" escape hatch.
    """

    def __init__(self, channels: int, hidden: int, layers: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.channels = int(channels)
        self.gru = nn.GRU(
            input_size=channels,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=float(dropout) if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, 2 * channels)

    def forward(self, z: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        out, state = self.gru(z, state)
        h = self.head(out)
        delta, log_var = h[..., : self.channels], h[..., self.channels :]
        return z + delta, log_var.clamp(-12.0, 12.0), state


class DenseNeuralBaseline(Baseline):
    """Equal-capacity neural forecaster with no connectome and no region state.

    Why it exists: "our model predicts held-out EEG well" is not evidence for
    *structure* unless a network of the same size without that structure does
    worse.  This baseline is the capacity control: a plain multi-layer GRU over
    the flattened channel vector with a heteroscedastic Gaussian head, sized to
    land within ``tolerance`` of ``target_parameters`` so the phrase
    "matched capacity" is checkable rather than rhetorical.  If it is not
    achievable within tolerance, construction raises instead of quietly
    reporting a mismatched control.

    Training mirrors the evaluation task rather than only the easy part of it:
    the loss is the teacher-forced one-step NLL over the context segment plus
    the **free-running** NLL over a ``rollout_len``-step continuation, so the
    variance head is calibrated on the error growth it will actually be scored
    on.  Optimisation is Adam on fp32 master weights, with bf16 autocast on
    CUDA (the loss reduction is always fp32).

    Honest accounting note: the variance head is fitted on the training data
    in-sample, unlike the linear baselines which calibrate on held-out training
    windows.  That is the *conservative* direction for the linear controls and
    the *generous* direction for this one; it is stated in ``describe()``.
    """

    def __init__(
        self,
        target_parameters: int,
        *,
        num_layers: int = 2,
        tolerance: float = 0.10,
        steps: int = 2000,
        batch_size: int = 64,
        lr: float = 3e-3,
        weight_decay: float = 0.0,
        rollout_len: int | None = None,
        grad_clip: float = 1.0,
        bf16: bool = True,
        device: str | torch.device | None = None,
        seed: int = 0,
        name: str = "dense_gru_matched",
    ) -> None:
        super().__init__(device=device, seed=seed)
        if target_parameters < 1:
            raise ValueError("target_parameters must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.target_parameters = int(target_parameters)
        self.num_layers = int(num_layers)
        self.tolerance = float(tolerance)
        self.steps = int(steps)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.rollout_len = rollout_len
        self.grad_clip = float(grad_clip)
        self.bf16 = bool(bf16)
        self.name = name
        self.module_: _GRUForecaster | None = None
        self.hidden_size_: int | None = None
        self._param_error: float | None = None
        self._loss_history: list[float] = []
        self._mean: Tensor | None = None
        self._std: Tensor | None = None
        self._rollout_used: int | None = None

    # -- capacity matching --------------------------------------------------
    @staticmethod
    def _count(channels: int, hidden: int, layers: int) -> int:
        gru = 3 * hidden * (channels + hidden) + 6 * hidden
        gru += (layers - 1) * (3 * hidden * hidden * 2 + 6 * hidden)
        head = hidden * 2 * channels + 2 * channels
        return int(gru + head)

    def _size_to_target(self, channels: int) -> int:
        """Smallest hidden width whose parameter count best matches the target."""
        lo, hi = 1, 2
        while self._count(channels, hi, self.num_layers) < self.target_parameters:
            hi *= 2
            if hi > 1 << 20:
                raise ValueError("target_parameters too large to match with a GRU")
        while lo < hi:
            mid = (lo + hi) // 2
            if self._count(channels, mid, self.num_layers) < self.target_parameters:
                lo = mid + 1
            else:
                hi = mid
        cands = [h for h in (lo - 1, lo) if h >= 1]
        best = min(
            cands,
            key=lambda h: abs(self._count(channels, h, self.num_layers) - self.target_parameters),
        )
        n = self._count(channels, best, self.num_layers)
        err = abs(n - self.target_parameters) / self.target_parameters
        if err > self.tolerance:
            raise ValueError(
                f"{self.name}: closest achievable parameter count is {n} for target "
                f"{self.target_parameters} ({err:.1%} off, tolerance {self.tolerance:.0%}); "
                "a capacity-matched comparison must not be claimed at this width -- "
                "change num_layers or the target"
            )
        self._param_error = float(err)
        return int(best)

    # -- Baseline surface ---------------------------------------------------
    def fit(
        self,
        train_windows: Tensor,
        groups: Sequence[Any] | np.ndarray | None = None,
    ) -> DenseNeuralBaseline:
        w, _ = self._prepare(train_windows, groups)
        b, t, c = w.shape
        rollout = int(self.rollout_len) if self.rollout_len is not None else max(1, t // 4)
        rollout = max(1, min(rollout, t - 2))
        self._rollout_used = rollout
        ctx_len = t - rollout
        if ctx_len < 2:
            raise ValueError(f"{self.name}: window length {t} too short for rollout {rollout}")

        self._mean = w.mean(dim=(0, 1))
        self._std = w.std(dim=(0, 1)).clamp_min(1e-6)
        z = (w - self._mean) / self._std

        with torch.random.fork_rng(devices=[w.device] if w.device.type == "cuda" else []):
            torch.manual_seed(self.seed)
            self.hidden_size_ = self._size_to_target(c)
            model = _GRUForecaster(c, self.hidden_size_, self.num_layers).to(w.device)
            opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            gen = torch.Generator(device="cpu").manual_seed(self.seed + 1)
            use_amp = self.bf16 and w.device.type == "cuda"
            model.train()
            self._loss_history = []
            for step in range(self.steps):
                idx = torch.randint(0, b, (min(self.batch_size, b),), generator=gen)
                batch = z[idx.to(z.device)]
                with torch.autocast(
                    device_type=w.device.type, dtype=torch.bfloat16, enabled=use_amp
                ):
                    loss = self._training_loss(model, batch, ctx_len, rollout)
                opt.zero_grad(set_to_none=True)
                loss.float().backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                opt.step()
                if step % max(1, self.steps // 20) == 0 or step == self.steps - 1:
                    self._loss_history.append(float(loss.detach().float()))
            model.eval()
        self.module_ = model
        self._variance_calibration = (
            "heteroscedastic head trained in-sample on free-running rollouts "
            f"(rollout_len={rollout})"
        )
        self.fitted = True
        return self

    def _training_loss(
        self, model: _GRUForecaster, batch: Tensor, ctx_len: int, rollout: int
    ) -> Tensor:
        """Teacher-forced NLL on the context + free-running NLL on the rollout."""
        ctx, tgt = batch[:, :ctx_len], batch[:, ctx_len:]
        mean, log_var, state = model(ctx)
        tf_loss = _gaussian_nll(mean[:, :-1], log_var[:, :-1], ctx[:, 1:]).mean().float()
        # `mean[:, -1]` is already the one-step prediction of tgt[:, 0].
        cur_mean, cur_lv = mean[:, -1], log_var[:, -1]
        means, lvs = [cur_mean], [cur_lv]
        for _ in range(1, rollout):
            # Stop-gradient on the fed-back sample.  Backpropagating through a
            # long free-running GRU rollout is unstable and measurably *worsens*
            # this baseline; a weak capacity control would make the whole
            # comparison worthless, so the baseline is trained the way that
            # makes it strongest.
            step_mean, step_lv, state = model(cur_mean.unsqueeze(1).detach(), state)
            cur_mean, cur_lv = step_mean[:, 0], step_lv[:, 0]
            means.append(cur_mean)
            lvs.append(cur_lv)
        roll_mean = torch.stack(means, dim=1)
        roll_lv = torch.stack(lvs, dim=1)
        roll_loss = _gaussian_nll(roll_mean, roll_lv, tgt).mean().float()
        return tf_loss + roll_loss

    def predict(
        self,
        context: Tensor,
        horizon: int,
        *,
        groups: Sequence[Any] | np.ndarray | None = None,
    ) -> tuple[Tensor, Tensor]:
        if self.module_ is None or self._mean is None or self._std is None:
            raise RuntimeError(f"{self.name}: not fitted")
        ctx = _as_windows(context, "context", self.device)
        z = (ctx - self._mean) / self._std
        with torch.no_grad():
            mean, log_var, state = self.module_(z)
            cur_mean, cur_lv = mean[:, -1], log_var[:, -1]
            means, lvs = [cur_mean], [cur_lv]
            for _ in range(1, horizon):
                step_mean, step_lv, state = self.module_(cur_mean.unsqueeze(1), state)
                cur_mean, cur_lv = step_mean[:, 0], step_lv[:, 0]
                means.append(cur_mean)
                lvs.append(cur_lv)
            z_mean = torch.stack(means, dim=1)
            z_lv = torch.stack(lvs, dim=1)
        # Back to data units: x = mean + std * z  =>  var_x = std^2 var_z.
        out_mean = self._mean + self._std * z_mean
        out_lv = z_lv + 2.0 * self._std.log()
        return out_mean.float(), out_lv.float()

    def n_parameters(self) -> int:
        if self.module_ is None:
            return 0
        return int(sum(p.numel() for p in self.module_.parameters()))

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d["target_parameters"] = self.target_parameters
        d["parameter_error"] = self._param_error
        d["hidden_size"] = self.hidden_size_
        d["num_layers"] = self.num_layers
        d["train_steps"] = self.steps
        d["rollout_len"] = self._rollout_used
        d["loss_history"] = self._loss_history
        d["capacity_matched"] = (
            self._param_error is not None and self._param_error <= self.tolerance
        )
        return d

    def _assumptions(self) -> list[str]:
        return [
            "the optimiser found a solution worth comparing against "
            "(check loss_history before quoting this baseline)",
            "parameter count is a fair capacity proxy at matched training steps "
            "and matched data",
            "no connectome, adjacency or per-region state is used -- by construction",
            "its variance head is fitted in-sample, which flatters it relative to "
            "the held-out-calibrated linear baselines",
        ]

    def _falsifies_comparison_if(self) -> str:
        return (
            "If an unstructured network with the same parameter count, data and "
            "training budget matches the foundation model's held-out NLL, then the "
            "connectome and per-region state are decorative and the structural claim "
            "must be withdrawn."
        )


# ---------------------------------------------------------------------------
# intervals and comparison
# ---------------------------------------------------------------------------
def _cluster_index(groups: np.ndarray | None, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(cluster_ids_as_int, unique_count)`` for bootstrap resampling."""
    if groups is None:
        return np.arange(n), np.arange(n)
    _, inv = np.unique(groups, return_inverse=True)
    return inv, np.unique(inv)


def _boot_draws(n_clusters: int, n_boot: int, seed: int) -> np.ndarray:
    """``(n_boot, n_clusters)`` cluster indices, shared across paired statistics."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_clusters, size=(n_boot, n_clusters))


def _cluster_means(
    values: np.ndarray, cluster: np.ndarray, n_clusters: int
) -> tuple[np.ndarray, np.ndarray]:
    sums = np.bincount(cluster, weights=values, minlength=n_clusters)
    counts = np.bincount(cluster, minlength=n_clusters).astype(float)
    return sums, counts


def bootstrap_ci(
    per_unit_values: Sequence[float] | np.ndarray,
    groups: Sequence[Any] | np.ndarray | None = None,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Cluster bootstrap over **participants**, returning ``(point, lo, hi)``.

    ``per_unit_values`` is one number per scored window (e.g. the
    ``per_window_nll`` array from :meth:`Baseline.score`) and ``groups`` is the
    matching participant id per window.  Whole participants are resampled with
    replacement; every window of a drawn participant enters the replicate.

    This is not a stylistic preference.  Windows within a participant share
    electrode placement, impedance, arousal state, drift and montage, so they
    are far from independent; resampling *windows* treats a dataset of 20
    people as if it had thousands of independent observations and produces
    intervals that can be several times too narrow.  Passing ``groups=None``
    falls back to the window bootstrap, which is correct only when each row is
    genuinely an independent participant.

    The point estimate is the unweighted mean over windows (matching
    ``nll_per_sample``); a replicate's mean is likewise unweighted over the
    windows it drew, so participants contributing more windows carry more
    weight in exactly the same way in both.
    """
    v = np.asarray(per_unit_values, dtype=float).ravel()
    if v.size == 0:
        raise ValueError("bootstrap_ci on an empty sample")
    point = float(v.mean())
    if v.size == 1:
        return point, point, point
    g = _as_groups(groups, v.size, "groups")
    cluster, uniq = _cluster_index(g, v.size)
    n_clusters = int(uniq.size)
    if n_clusters < 2:
        return point, float("nan"), float("nan")
    sums, counts = _cluster_means(v, cluster, n_clusters)
    draws = _boot_draws(n_clusters, int(n_boot), int(seed))
    num = sums[draws].sum(axis=1)
    den = counts[draws].sum(axis=1)
    reps = num / np.where(den == 0.0, np.nan, den)
    lo, hi = np.nanquantile(reps, [alpha / 2.0, 1.0 - alpha / 2.0])
    return point, float(lo), float(hi)


def _paired_ci(
    diff: np.ndarray,
    groups: np.ndarray | None,
    *,
    n_boot: int,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    """Participant-clustered interval for a paired per-window difference."""
    point, lo, hi = bootstrap_ci(diff, groups, n_boot=n_boot, alpha=alpha, seed=seed)
    return {
        "delta": point,
        "ci_lo": lo,
        "ci_hi": hi,
        "excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0.0 or hi < 0.0)),
    }


def compare(
    models: Mapping[str, Baseline],
    context: Tensor,
    target: Tensor,
    groups: Sequence[Any] | np.ndarray | None = None,
    *,
    reference: str | None = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Score every fitted model at matched inputs and tabulate the result.

    All models see the identical ``context`` and are scored on the identical
    ``target``; that is the whole point, and it is enforced here rather than
    left to the caller's discipline.  Every number ships with a
    participant-clustered 95% interval, and every model is additionally
    reported as a **paired** difference against ``reference`` using the same
    bootstrap draws, because the paired contrast is the quantity a claim
    actually rests on and it is far tighter than the difference of two marginal
    intervals.

    ``reference`` defaults to ``"persistence"`` when present, else the first
    key.  Differences are ``model - reference`` on the NLL scale, so a negative
    delta whose interval excludes zero means the model is better.

    The returned dict is machine-readable and JSON-serialisable after
    ``per_window`` arrays are dropped; it keeps ``describe()`` for every model
    so a report cannot quote the number without the assumptions.
    """
    if not models:
        raise ValueError("compare() needs at least one model")
    ctx = _as_windows(context, "context")
    tgt = _as_windows(target, "target")
    g = _as_groups(groups, ctx.shape[0], "groups")
    ref = reference or ("persistence" if "persistence" in models else next(iter(models)))
    if ref not in models:
        raise KeyError(f"reference model {ref!r} not among {sorted(models)}")

    rows: dict[str, dict[str, Any]] = {}
    scores: dict[str, dict[str, Any]] = {}
    for key, model in models.items():
        if not model.fitted:
            raise RuntimeError(f"compare(): model {key!r} is not fitted")
        s = model.score(ctx, tgt, g)
        scores[key] = s
        nll_pt, nll_lo, nll_hi = bootstrap_ci(
            s["per_window_nll"], g, n_boot=n_boot, alpha=alpha, seed=seed
        )
        mse_pt, mse_lo, mse_hi = bootstrap_ci(
            s["per_window_mse"], g, n_boot=n_boot, alpha=alpha, seed=seed
        )
        rows[key] = {
            "model": key,
            "nll_per_sample": nll_pt,
            "nll_ci": (nll_lo, nll_hi),
            "mse": mse_pt,
            "mse_ci": (mse_lo, mse_hi),
            "n_windows": s["n"],
            "n_elements": s["n_elements"],
            "n_parameters": s["n_parameters"],
            "nll_per_horizon": s["nll_per_horizon"],
            "describe": model.describe(),
        }

    ref_nll = scores[ref]["per_window_nll"]
    ref_mse = scores[ref]["per_window_mse"]
    paired: dict[str, dict[str, Any]] = {}
    for key, s in scores.items():
        if key == ref:
            continue
        paired[key] = {
            "nll": _paired_ci(
                s["per_window_nll"] - ref_nll, g, n_boot=n_boot, alpha=alpha, seed=seed
            ),
            "mse": _paired_ci(
                s["per_window_mse"] - ref_mse, g, n_boot=n_boot, alpha=alpha, seed=seed
            ),
        }

    n_subjects = int(np.unique(g).size) if g is not None else None
    return {
        "metric": "gaussian_nll",
        "units": "nats per channel per time-sample",
        "lower_is_better": True,
        "reference": ref,
        "difference_convention": "model minus reference; negative favours the model",
        "n_windows": int(ctx.shape[0]),
        "n_subjects": n_subjects,
        "n_channels": int(ctx.shape[2]),
        "context_len": int(ctx.shape[1]),
        "horizon": int(tgt.shape[1]),
        "alpha": float(alpha),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "interval_method": (
            "cluster bootstrap over participants"
            if g is not None
            else "window bootstrap (NO participant clustering: intervals are "
            "anti-conservative unless each window is an independent participant)"
        ),
        "table": rows,
        "paired_vs_reference": paired,
        "per_window": {k: s["per_window_nll"] for k, s in scores.items()},
        "groups": g,
    }


def render_comparison(result: Mapping[str, Any], *, precision: int = 4) -> str:
    """Fixed-width text rendering of :func:`compare` for logs and reports."""
    rows = result["table"]
    ref = result["reference"]
    width = max(len(k) for k in rows) + 2
    p = precision
    lines = [
        f"metric: {result['metric']} ({result['units']}), lower is better",
        f"windows={result['n_windows']} subjects={result['n_subjects']} "
        f"channels={result['n_channels']} ctx={result['context_len']} "
        f"horizon={result['horizon']}",
        f"intervals: {result['interval_method']}, {100 * (1 - result['alpha']):.0f}% "
        f"({result['n_boot']} draws)",
        "",
        f"{'model':<{width}}{'params':>10}{'NLL':>{p + 7}}{'95% CI':>{2 * p + 14}}"
        f"{'MSE':>{p + 8}}{'dNLL vs ' + ref:>{p + 14}}{'95% CI':>{2 * p + 14}}",
    ]
    for key, row in rows.items():
        lo, hi = row["nll_ci"]
        if key == ref:
            d_txt, dci_txt = "(reference)", ""
        else:
            d = result["paired_vs_reference"][key]["nll"]
            star = "*" if d["excludes_zero"] else " "
            d_txt = f"{d['delta']:+.{p}f}{star}"
            dci_txt = f"[{d['ci_lo']:+.{p}f}, {d['ci_hi']:+.{p}f}]"
        lines.append(
            f"{key:<{width}}{row['n_parameters']:>10}{row['nll_per_sample']:>{p + 7}.{p}f}"
            f"{f'[{lo:.{p}f}, {hi:.{p}f}]':>{2 * p + 14}}{row['mse']:>{p + 8}.{p}f}"
            f"{d_txt:>{p + 14}}{dci_txt:>{2 * p + 14}}"
        )
    lines.append("")
    lines.append("* paired participant-clustered interval excludes zero")
    return "\n".join(lines)
