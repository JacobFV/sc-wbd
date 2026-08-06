"""State estimators for SC-WBD, on **native clocks**.

The central object is :class:`LinearGaussianSSM`: a batched, multirate,
linear--Gaussian state-space model whose observation channels each carry their
own base-step schedule, their own read operator ``H`` (possibly with a
different number of channels/parcels -- *unequal supports*) and their own noise
covariance ``R``.

The filter never resamples.  EEG arriving every base step (1 ms) and BOLD
arriving every 1000 base steps (1 s) are consumed as two channels of the same
recursion; a step with no observation is a pure prediction, a step whose data
are missing for some batch elements applies a per-element zero gain.  This is
the executable form of ``body.tex`` sec. 7.1: *"EEG samples need not be
downsampled to the fMRI repetition time."*

Also provided, for the nonlinear backends of ``scwbd.dynamics``: extended and
unscented Kalman filters, an ensemble Kalman filter, and a bootstrap
particle/SMC filter with systematic resampling.  All reduce to the exact
Kalman result on a linear--Gaussian problem, which is a test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import torch
from torch import Tensor

from .types import DTYPE, resolve_device, torch_dtype

__all__ = [
    "EnsembleKalmanResult",
    "FilterResult",
    "LinearGaussianSSM",
    "ObservationChannel",
    "ParticleFilterResult",
    "SmootherResult",
    "ensemble_kalman_filter",
    "extended_kalman_filter",
    "kalman_filter",
    "multiepoch_kalman_filter",
    "particle_filter",
    "rts_smoother",
    "unscented_kalman_filter",
]

_LOG2PI = math.log(2.0 * math.pi)


def _expand_batch(x: Tensor, batch: int) -> Tensor:
    if x.dim() == 0:
        raise ValueError("scalar tensors are not valid state-space operators")
    if x.shape[0] == batch:
        return x
    if x.shape[0] == 1:
        return x.expand(batch, *x.shape[1:])
    raise ValueError(f"cannot broadcast leading dim {x.shape[0]} to batch {batch}")


@dataclass
class ObservationChannel:
    """One source-native read head.

    Parameters
    ----------
    name:
        Channel identifier, e.g. ``"eeg"`` / ``"bold"``.
    H:
        ``[B, p, n]`` (or ``[1, p, n]``) read operator.  ``p`` may differ
        between channels: supports are not required to be equal.
    R:
        ``[B, p, p]`` observation noise covariance on this channel's clock.
    steps:
        Long tensor of **base-step indices** at which this channel produces a
        sample.  Strictly increasing.  This *is* the native clock.
    """

    name: str
    H: Tensor
    R: Tensor
    steps: Tensor

    def __post_init__(self) -> None:
        if self.H.dim() != 3 or self.R.dim() != 3:
            raise ValueError(f"channel {self.name}: H and R must be [B, p, *]")
        if self.H.shape[-2] != self.R.shape[-1]:
            raise ValueError(f"channel {self.name}: H/R dimension mismatch")
        self.steps = self.steps.to(torch.long).reshape(-1)
        if self.steps.numel() > 1 and bool((self.steps[1:] <= self.steps[:-1]).any()):
            raise ValueError(f"channel {self.name}: steps must be strictly increasing")

    @property
    def p(self) -> int:
        return int(self.H.shape[-2])

    @property
    def n_obs(self) -> int:
        return int(self.steps.numel())


@dataclass
class LinearGaussianSSM:
    """``z_{k+1} = F z_k + b_k + w_k``, ``w_k ~ N(0, Q)``; ``z_0 ~ N(m0, P0)``.

    Observations of ``z_k`` are emitted by every channel whose schedule
    contains ``k``.  ``inputs`` holds the deterministic drive ``b_k`` and is
    ``[B, n_steps, n]`` (or ``None`` for the autonomous/zero-input model, which
    is what the Fisher whitening filter uses).
    """

    F: Tensor
    Q: Tensor
    m0: Tensor
    P0: Tensor
    channels: list[ObservationChannel]
    n_steps: int
    inputs: Tensor | None = None
    left_mul: Callable[[Tensor], Tensor] | None = None
    """Optional fast ``X -> F @ X`` exploiting known structure (delay-line shift
    registers make ``F`` mostly a permutation).  Must be numerically identical
    to ``F @ X``; ``tests/infer/test_filters.py`` asserts that."""

    def __post_init__(self) -> None:
        if self.F.dim() != 3:
            raise ValueError("F must be [B, n, n]")
        n = self.F.shape[-1]
        for nm, t, shape in (
            ("Q", self.Q, (n, n)),
            ("P0", self.P0, (n, n)),
        ):
            if tuple(t.shape[-2:]) != shape:
                raise ValueError(f"{nm} must be [B, {n}, {n}]")
        if self.m0.shape[-1] != n:
            raise ValueError("m0 must be [B, n]")
        for ch in self.channels:
            if ch.H.shape[-1] != n:
                raise ValueError(f"channel {ch.name}: H last dim must equal n={n}")
            if ch.n_obs and int(ch.steps.max()) >= self.n_steps:
                raise ValueError(f"channel {ch.name}: step index beyond n_steps")

    @property
    def n(self) -> int:
        return int(self.F.shape[-1])

    @property
    def batch(self) -> int:
        cand = [self.F.shape[0], self.Q.shape[0], self.m0.shape[0], self.P0.shape[0]]
        cand += [ch.H.shape[0] for ch in self.channels]
        cand += [ch.R.shape[0] for ch in self.channels]
        if self.inputs is not None:
            cand.append(self.inputs.shape[0])
        return max(cand)

    def channel(self, name: str) -> ObservationChannel:
        for ch in self.channels:
            if ch.name == name:
                return ch
        raise KeyError(name)

    def subset(self, names: Sequence[str]) -> "LinearGaussianSSM":
        """A design restriction: keep only the named read channels."""
        keep = [ch for ch in self.channels if ch.name in set(names)]
        missing = set(names) - {ch.name for ch in keep}
        if missing:
            raise KeyError(f"unknown channels {sorted(missing)}")
        return LinearGaussianSSM(
            self.F, self.Q, self.m0, self.P0, keep, self.n_steps, self.inputs,
            self.left_mul,
        )

    def schedule(self) -> dict[int, list[tuple[ObservationChannel, int]]]:
        """base-step -> [(channel, index within that channel's record)]."""
        sched: dict[int, list[tuple[ObservationChannel, int]]] = {}
        for ch in self.channels:
            for j, k in enumerate(ch.steps.tolist()):
                sched.setdefault(int(k), []).append((ch, j))
        return sched


@dataclass
class FilterResult:
    log_likelihood: Tensor
    log_likelihood_by_channel: dict[str, Tensor]
    n_observations_used: dict[str, Tensor]
    whitened_innovations: dict[str, Tensor] = field(default_factory=dict)
    innovations: dict[str, Tensor] = field(default_factory=dict)
    predicted_obs: dict[str, Tensor] = field(default_factory=dict)
    filtered_mean: Tensor | None = None
    filtered_cov: Tensor | None = None
    predicted_mean: Tensor | None = None
    predicted_cov: Tensor | None = None


@dataclass
class SmootherResult:
    smoothed_mean: Tensor
    smoothed_cov: Tensor
    lag_one_cov: Tensor | None = None


def _prepare(
    ssm: LinearGaussianSSM,
) -> tuple[int, int, Tensor, Tensor, Tensor, Tensor]:
    b = ssm.batch
    n = ssm.n
    F = _expand_batch(ssm.F, b)
    Q = _expand_batch(ssm.Q, b)
    m0 = _expand_batch(ssm.m0, b)
    P0 = _expand_batch(ssm.P0, b)
    return b, n, F, Q, m0, P0


def kalman_filter(
    ssm: LinearGaussianSSM,
    data: dict[str, Tensor] | None = None,
    masks: dict[str, Tensor] | None = None,
    *,
    store: str = "none",
    whiten: bool = False,
    keep_innovations: bool = False,
    jitter: float = 0.0,
) -> FilterResult:
    """Exact multirate Kalman filter.

    ``data[name]`` has shape ``[B, n_obs_name, p_name]`` and lives on that
    channel's own clock.  ``masks[name]`` is ``[B, n_obs_name]`` in ``{0,1}``
    marking *available* samples; a zero applies a zero gain for that batch
    element and contributes nothing to the likelihood -- this is how missing
    windows are handled, never by imputation (ARCHITECTURE.md sec. 7 rule 1).

    ``store='all'`` retains filtered and predicted moments for the RTS
    smoother; it is O(T n^2) memory and is meant for small problems.
    """
    data = data or {}
    masks = masks or {}
    b, n, F, Q, m0, P0 = _prepare(ssm)
    dtype, device = F.dtype, F.device
    eye = torch.eye(n, dtype=dtype, device=device).expand(b, n, n)

    m = m0.clone()
    P = P0.clone()
    if jitter:
        P = P + jitter * eye

    ll = torch.zeros(b, dtype=dtype, device=device)
    ll_ch = {ch.name: torch.zeros(b, dtype=dtype, device=device) for ch in ssm.channels}
    n_used = {ch.name: torch.zeros(b, dtype=dtype, device=device) for ch in ssm.channels}
    wh: dict[str, list[Tensor]] = {ch.name: [] for ch in ssm.channels}
    inn: dict[str, list[Tensor]] = {ch.name: [] for ch in ssm.channels}
    pred_obs: dict[str, list[Tensor]] = {ch.name: [] for ch in ssm.channels}

    keep_moments = store == "all"
    fm: list[Tensor] = []
    fc: list[Tensor] = []
    pm: list[Tensor] = []
    pc: list[Tensor] = []

    sched = ssm.schedule()
    inputs = ssm.inputs
    if inputs is not None:
        inputs = _expand_batch(inputs, b)
    fmul = ssm.left_mul if ssm.left_mul is not None else (lambda X: F @ X)

    for k in range(ssm.n_steps):
        if keep_moments:
            pm.append(m)
            pc.append(P)
        for ch, j in sched.get(k, ()):  # native-clock reads, no resampling
            H = _expand_batch(ch.H, b)
            R = _expand_batch(ch.R, b)
            y = data.get(ch.name)
            mask = masks.get(ch.name)
            mk = (
                torch.ones(b, dtype=dtype, device=device)
                if mask is None
                else _expand_batch(mask, b)[:, j].to(dtype)
            )
            PHt = P @ H.transpose(-1, -2)
            S = H @ PHt + R
            S = 0.5 * (S + S.transpose(-1, -2))
            Lc = torch.linalg.cholesky_ex(S)[0]
            yhat = (H @ m.unsqueeze(-1)).squeeze(-1)
            if y is None:
                v = torch.zeros_like(yhat)
            else:
                v = _expand_batch(y, b)[:, j, :] - yhat
            alpha = torch.cholesky_solve(v.unsqueeze(-1), Lc)  # [b,p,1]
            K = torch.cholesky_solve(PHt.transpose(-1, -2), Lc).transpose(-1, -2)
            logdet = 2.0 * torch.log(torch.diagonal(Lc, dim1=-2, dim2=-1)).sum(-1)
            quad = (v.unsqueeze(-1) * alpha).sum((-2, -1))
            term = -0.5 * (quad + logdet + ch.p * _LOG2PI) * mk
            ll = ll + term
            ll_ch[ch.name] = ll_ch[ch.name] + term
            n_used[ch.name] = n_used[ch.name] + mk
            if whiten:
                wh[ch.name].append(
                    torch.linalg.solve_triangular(
                        Lc, v.unsqueeze(-1), upper=False
                    ).squeeze(-1)
                    * mk.unsqueeze(-1)
                )
            if keep_innovations:
                inn[ch.name].append(v)
                pred_obs[ch.name].append(yhat)
            mk3 = mk.reshape(b, 1, 1)
            Km = K * mk3
            # K already contains S^{-1}; the mean update uses the raw innovation.
            m = m + (Km @ v.unsqueeze(-1)).squeeze(-1)
            IKH = eye - Km @ H
            P = IKH @ P @ IKH.transpose(-1, -2) + Km @ R @ Km.transpose(-1, -2)
            P = 0.5 * (P + P.transpose(-1, -2))
        if keep_moments:
            fm.append(m)
            fc.append(P)
        if k + 1 < ssm.n_steps:
            m = fmul(m.unsqueeze(-1)).squeeze(-1)
            if inputs is not None:
                m = m + inputs[:, k, :]
            # F P F^T computed as F (F P)^T (P symmetric) so one routine suffices
            P = fmul(fmul(P).transpose(-1, -2)) + Q
            P = 0.5 * (P + P.transpose(-1, -2))

    res = FilterResult(
        log_likelihood=ll,
        log_likelihood_by_channel=ll_ch,
        n_observations_used=n_used,
        whitened_innovations={
            k: torch.stack(v, dim=1) for k, v in wh.items() if v
        },
        innovations={k: torch.stack(v, dim=1) for k, v in inn.items() if v},
        predicted_obs={k: torch.stack(v, dim=1) for k, v in pred_obs.items() if v},
    )
    if keep_moments:
        res.filtered_mean = torch.stack(fm, dim=1)
        res.filtered_cov = torch.stack(fc, dim=1)
        res.predicted_mean = torch.stack(pm, dim=1)
        res.predicted_cov = torch.stack(pc, dim=1)
    return res


def multiepoch_kalman_filter(
    ssm: LinearGaussianSSM,
    data: dict[str, Tensor],
    masks: dict[str, Tensor] | None = None,
    *,
    n_epochs: int,
    checkpoint_every: int = 0,
    whiten: bool = False,
) -> dict[str, Tensor]:
    """Exact log-likelihood for ``n_epochs`` independent epochs of one subject.

    The Riccati/covariance recursion depends only on ``eta`` and on the
    observation *schedule*, both of which are identical across epochs, so it is
    computed **once** and its gains are reused by every epoch's mean recursion.
    This is an exact algebraic saving, not an approximation: the per-epoch
    log-likelihoods are bit-comparable with looping :func:`kalman_filter` per
    epoch (asserted in ``tests/infer/test_filters.py``).

    Shapes: ``data[name]`` is ``[B, E, n_obs, p]``; ``masks[name]`` is
    ``[B, n_obs]`` -- masks must be shared across epochs, otherwise the
    covariance recursion is no longer shared and :func:`kalman_filter` should be
    used instead.

    Returns ``{"log_likelihood": [B, E], "<channel>": [B, E]}``.  With
    ``whiten=True`` it additionally returns ``"whitened/<channel>"`` of shape
    ``[B, E, n_obs, p]``: the innovations pre-multiplied by ``chol(S_k)^{-1}``.
    Because the innovations transform is exactly the Cholesky whitening of the
    stacked residual covariance ``R_m``, these give ``J^T R_m^{-1} J`` in T4
    without ever forming ``R_m`` (which has ~10^5 rows).
    """
    masks = masks or {}
    b, n, F, Q, m0, P0 = _prepare(ssm)
    E = n_epochs
    dtype, device = F.dtype, F.device
    eye = torch.eye(n, dtype=dtype, device=device).expand(b, n, n)
    for nm, mk in masks.items():
        if mk.dim() != 2:
            raise ValueError(
                f"mask for {nm} must be [B, n_obs] (shared across epochs); use "
                "kalman_filter for per-epoch missingness"
            )
    m = m0.unsqueeze(1).expand(b, E, n).contiguous()
    P = P0.clone()
    ll = torch.zeros(b, E, dtype=dtype, device=device)
    ll_ch = {ch.name: torch.zeros(b, E, dtype=dtype, device=device) for ch in ssm.channels}
    inputs = ssm.inputs
    if inputs is not None:
        if inputs.dim() != 4:
            raise ValueError("inputs must be [B, E, T, n] for multiepoch filtering")
        inputs = _expand_batch(inputs, b)
    sched = ssm.schedule()
    fmul = ssm.left_mul if ssm.left_mul is not None else (lambda X: F @ X)

    wh: dict[str, list[Tensor]] = {ch.name: [] for ch in ssm.channels}

    def step_block(lo: int, hi: int, m: Tensor, P: Tensor, ll: Tensor, *chs: Tensor):
        acc = list(chs)
        for k in range(lo, hi):
            for ch, j in sched.get(k, ()):
                H = _expand_batch(ch.H, b)
                R = _expand_batch(ch.R, b)
                mk = masks.get(ch.name)
                w = (
                    torch.ones(b, 1, dtype=dtype, device=device)
                    if mk is None
                    else _expand_batch(mk, b)[:, j : j + 1].to(dtype)
                )
                PHt = P @ H.transpose(-1, -2)
                S = H @ PHt + R
                S = 0.5 * (S + S.transpose(-1, -2))
                Lc = torch.linalg.cholesky_ex(S)[0]
                yhat = torch.einsum("bpn,ben->bep", H, m)
                v = _expand_batch(data[ch.name], b)[:, :, j, :] - yhat
                sol = torch.cholesky_solve(v.transpose(1, 2), Lc).transpose(1, 2)
                quad = (v * sol).sum(-1)
                logdet = 2.0 * torch.log(torch.diagonal(Lc, dim1=-2, dim2=-1)).sum(-1)
                term = -0.5 * (quad + logdet.unsqueeze(1) + ch.p * _LOG2PI) * w
                ll = ll + term
                idx = [c.name for c in ssm.channels].index(ch.name)
                acc[idx] = acc[idx] + term
                if whiten:
                    wh[ch.name].append(
                        torch.linalg.solve_triangular(
                            Lc, v.transpose(1, 2), upper=False
                        ).transpose(1, 2)
                        * w.unsqueeze(-1)
                    )
                K = torch.cholesky_solve(PHt.transpose(-1, -2), Lc).transpose(-1, -2)
                K = K * w.reshape(b, 1, 1)
                m = m + torch.einsum("bnp,bep->ben", K, v)
                # P - K S K^T: symmetric by construction and O(n^2 p) rather
                # than the O(n^3) Joseph form, which matters because this
                # recursion is the whole cost of the 1 ms native-clock filter.
                KS = K @ S
                P = P - KS @ K.transpose(-1, -2)
            if k + 1 < ssm.n_steps:
                m = torch.einsum("bnm,bem->ben", F, m) if ssm.left_mul is None else (
                    fmul(m.transpose(1, 2)).transpose(1, 2)
                )
                if inputs is not None:
                    m = m + inputs[:, :, k, :]
                P = fmul(fmul(P).transpose(-1, -2)) + Q
                P = 0.5 * (P + P.transpose(-1, -2))
        return (m, P, ll, *acc)

    chs = tuple(ll_ch[c.name] for c in ssm.channels)
    if checkpoint_every and checkpoint_every > 0:
        from torch.utils.checkpoint import checkpoint

        for lo in range(0, ssm.n_steps, checkpoint_every):
            hi = min(lo + checkpoint_every, ssm.n_steps)
            out = checkpoint(
                step_block, lo, hi, m, P, ll, *chs, use_reentrant=False
            )
            m, P, ll, chs = out[0], out[1], out[2], tuple(out[3:])
    else:
        out = step_block(0, ssm.n_steps, m, P, ll, *chs)
        m, P, ll, chs = out[0], out[1], out[2], tuple(out[3:])

    res = {"log_likelihood": ll}
    for c, t in zip(ssm.channels, chs):
        res[c.name] = t
    for name, lst in wh.items():
        if lst:
            res["whitened/" + name] = torch.stack(lst, dim=2)
    return res


def rts_smoother(ssm: LinearGaussianSSM, fr: FilterResult) -> SmootherResult:
    """Rauch--Tung--Striebel smoother; exact for the linear--Gaussian case."""
    if fr.filtered_mean is None or fr.filtered_cov is None:
        raise ValueError("run kalman_filter(store='all') before smoothing")
    b, n, F, Q, _, _ = _prepare(ssm)
    fm, fc = fr.filtered_mean, fr.filtered_cov
    T = fm.shape[1]
    sm = [None] * T
    sc = [None] * T
    lag1 = [None] * T
    sm[T - 1], sc[T - 1] = fm[:, T - 1], fc[:, T - 1]
    inputs = ssm.inputs
    if inputs is not None:
        inputs = _expand_batch(inputs, b)
    for k in range(T - 2, -1, -1):
        Pk = fc[:, k]
        mk = fm[:, k]
        m_pred = (F @ mk.unsqueeze(-1)).squeeze(-1)
        if inputs is not None:
            m_pred = m_pred + inputs[:, k, :]
        P_pred = F @ Pk @ F.transpose(-1, -2) + Q
        P_pred = 0.5 * (P_pred + P_pred.transpose(-1, -2))
        Lc = torch.linalg.cholesky(P_pred)
        G = torch.cholesky_solve((Pk @ F.transpose(-1, -2)).transpose(-1, -2), Lc)
        G = G.transpose(-1, -2)  # Pk F^T P_pred^{-1}
        sm[k] = mk + (G @ (sm[k + 1] - m_pred).unsqueeze(-1)).squeeze(-1)
        sc[k] = Pk + G @ (sc[k + 1] - P_pred) @ G.transpose(-1, -2)
        sc[k] = 0.5 * (sc[k] + sc[k].transpose(-1, -2))
        lag1[k + 1] = sc[k + 1] @ G.transpose(-1, -2)
    lag1[0] = torch.zeros(b, n, n, dtype=F.dtype, device=F.device)
    return SmootherResult(
        smoothed_mean=torch.stack(sm, dim=1),
        smoothed_cov=torch.stack(sc, dim=1),
        lag_one_cov=torch.stack(lag1, dim=1),
    )


def simulate_lgssm(
    ssm: LinearGaussianSSM,
    *,
    seed: int,
    batch: int | None = None,
    masks: dict[str, Tensor] | None = None,
) -> tuple[dict[str, Tensor], Tensor]:
    """Draw states and native-clock observations.  Deterministic given ``seed``."""
    b = batch or ssm.batch
    _, n, F, Q, m0, P0 = _prepare(ssm)
    F, Q, m0, P0 = (_expand_batch(t, b) for t in (F, Q, m0, P0))
    dtype, device = F.dtype, F.device
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    def randn(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=gen, dtype=dtype).to(device)

    L0 = torch.linalg.cholesky(P0 + 1e-14 * torch.eye(n, dtype=dtype, device=device))
    LQ = torch.linalg.cholesky(Q + 1e-14 * torch.eye(n, dtype=dtype, device=device))
    z = m0 + (L0 @ randn(b, n, 1)).squeeze(-1)
    inputs = ssm.inputs
    if inputs is not None:
        inputs = _expand_batch(inputs, b)

    sched = ssm.schedule()
    out: dict[str, list[Tensor]] = {ch.name: [] for ch in ssm.channels}
    states = []
    for k in range(ssm.n_steps):
        states.append(z)
        for ch, _j in sched.get(k, ()):
            H = _expand_batch(ch.H, b)
            R = _expand_batch(ch.R, b)
            LR = torch.linalg.cholesky(R)
            y = (H @ z.unsqueeze(-1)).squeeze(-1) + (LR @ randn(b, ch.p, 1)).squeeze(-1)
            out[ch.name].append(y)
        if k + 1 < ssm.n_steps:
            z = (F @ z.unsqueeze(-1)).squeeze(-1) + (LQ @ randn(b, n, 1)).squeeze(-1)
            if inputs is not None:
                z = z + inputs[:, k, :]
    data = {k: torch.stack(v, dim=1) for k, v in out.items() if v}
    return data, torch.stack(states, dim=1)


def deterministic_response(ssm: LinearGaussianSSM) -> dict[str, Tensor]:
    """Noise-free observation means ``mu_m(eta)`` on each native clock.

    This is the object differentiated to form ``J_m = d mu_m / d eta`` in T4.
    """
    b, n, F, _, m0, _ = _prepare(ssm)
    inputs = ssm.inputs
    if inputs is not None:
        inputs = _expand_batch(inputs, b)
    sched = ssm.schedule()
    out: dict[str, list[Tensor]] = {ch.name: [] for ch in ssm.channels}
    z = m0
    for k in range(ssm.n_steps):
        for ch, _j in sched.get(k, ()):
            H = _expand_batch(ch.H, b)
            out[ch.name].append((H @ z.unsqueeze(-1)).squeeze(-1))
        if k + 1 < ssm.n_steps:
            z = (F @ z.unsqueeze(-1)).squeeze(-1)
            if inputs is not None:
                z = z + inputs[:, k, :]
    return {k: torch.stack(v, dim=1) for k, v in out.items() if v}


# --------------------------------------------------------------------------
# Nonlinear filters (for the nonlinear dynamics backends)
# --------------------------------------------------------------------------

TransitionFn = Callable[[Tensor, int], Tensor]
ObservationFn = Callable[[Tensor, int], Tensor]


def _jacobian(fn: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
    """Batched Jacobian of ``fn`` at ``x`` (``[B, n] -> [B, m]``)."""
    x = x.detach().requires_grad_(True)
    y = fn(x)
    m = y.shape[-1]
    rows = []
    for i in range(m):
        (g,) = torch.autograd.grad(y[..., i].sum(), x, retain_graph=(i < m - 1))
        rows.append(g)
    return torch.stack(rows, dim=-2)


@dataclass
class _NLSchedule:
    steps: dict[int, list[tuple[str, int]]]


def _nl_schedule(channels: Sequence[tuple[str, Tensor]]) -> _NLSchedule:
    steps: dict[int, list[tuple[str, int]]] = {}
    for name, sc in channels:
        for j, k in enumerate(sc.tolist()):
            steps.setdefault(int(k), []).append((name, j))
    return _NLSchedule(steps)


def extended_kalman_filter(
    f: TransitionFn,
    h: dict[str, ObservationFn],
    Q: Tensor,
    R: dict[str, Tensor],
    m0: Tensor,
    P0: Tensor,
    data: dict[str, Tensor],
    steps: dict[str, Tensor],
    n_steps: int,
) -> FilterResult:
    """First-order EKF.  Reduces exactly to the KF when ``f``/``h`` are affine."""
    b, n = m0.shape
    dtype, device = m0.dtype, m0.device
    eye = torch.eye(n, dtype=dtype, device=device).expand(b, n, n)
    m, P = m0.clone(), P0.clone()
    ll = torch.zeros(b, dtype=dtype, device=device)
    ll_ch = {k: torch.zeros(b, dtype=dtype, device=device) for k in h}
    n_used = {k: torch.zeros(b, dtype=dtype, device=device) for k in h}
    sched = _nl_schedule([(k, v) for k, v in steps.items()])
    for k in range(n_steps):
        for name, j in sched.steps.get(k, ()):
            hk = lambda x, _n=name, _k=k: h[_n](x, _k)  # noqa: E731
            H = _jacobian(hk, m)
            S = H @ P @ H.transpose(-1, -2) + R[name]
            S = 0.5 * (S + S.transpose(-1, -2))
            Lc = torch.linalg.cholesky_ex(S)[0]
            v = data[name][:, j, :] - hk(m)
            alpha = torch.cholesky_solve(v.unsqueeze(-1), Lc)
            K = torch.cholesky_solve((P @ H.transpose(-1, -2)).transpose(-1, -2), Lc)
            K = K.transpose(-1, -2)
            logdet = 2.0 * torch.log(torch.diagonal(Lc, dim1=-2, dim2=-1)).sum(-1)
            term = -0.5 * (
                (v.unsqueeze(-1) * alpha).sum((-2, -1)) + logdet + v.shape[-1] * _LOG2PI
            )
            ll = ll + term
            ll_ch[name] = ll_ch[name] + term
            n_used[name] = n_used[name] + 1
            m = m + (K @ v.unsqueeze(-1)).squeeze(-1)
            IKH = eye - K @ H
            P = IKH @ P @ IKH.transpose(-1, -2) + K @ R[name] @ K.transpose(-1, -2)
        if k + 1 < n_steps:
            fk = lambda x, _k=k: f(x, _k)  # noqa: E731
            Fk = _jacobian(fk, m)
            m = fk(m)
            P = Fk @ P @ Fk.transpose(-1, -2) + Q
            P = 0.5 * (P + P.transpose(-1, -2))
    return FilterResult(ll, ll_ch, n_used)


def _sigma_points(m: Tensor, P: Tensor, alpha: float, beta: float, kappa: float):
    b, n = m.shape
    lam = alpha**2 * (n + kappa) - n
    c = n + lam
    L = torch.linalg.cholesky(P * c)
    pts = [m]
    for i in range(n):
        pts.append(m + L[..., :, i])
    for i in range(n):
        pts.append(m - L[..., :, i])
    X = torch.stack(pts, dim=1)  # [b, 2n+1, n]
    wm = torch.full((2 * n + 1,), 1.0 / (2 * c), dtype=m.dtype, device=m.device)
    wc = wm.clone()
    wm[0] = lam / c
    wc[0] = lam / c + (1 - alpha**2 + beta)
    return X, wm, wc


def unscented_kalman_filter(
    f: TransitionFn,
    h: dict[str, ObservationFn],
    Q: Tensor,
    R: dict[str, Tensor],
    m0: Tensor,
    P0: Tensor,
    data: dict[str, Tensor],
    steps: dict[str, Tensor],
    n_steps: int,
    *,
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> FilterResult:
    """Unscented KF (scaled sigma points).  Exact for affine models."""
    b, n = m0.shape
    dtype, device = m0.dtype, m0.device
    m, P = m0.clone(), P0.clone()
    ll = torch.zeros(b, dtype=dtype, device=device)
    ll_ch = {k: torch.zeros(b, dtype=dtype, device=device) for k in h}
    n_used = {k: torch.zeros(b, dtype=dtype, device=device) for k in h}
    sched = _nl_schedule([(k, v) for k, v in steps.items()])
    for k in range(n_steps):
        for name, j in sched.steps.get(k, ()):
            X, wm, wc = _sigma_points(m, P, alpha, beta, kappa)
            Y = torch.stack([h[name](X[:, i, :], k) for i in range(X.shape[1])], dim=1)
            ybar = (wm.view(1, -1, 1) * Y).sum(1)
            dY = Y - ybar.unsqueeze(1)
            dX = X - m.unsqueeze(1)
            S = torch.einsum("i,bip,biq->bpq", wc, dY, dY) + R[name]
            S = 0.5 * (S + S.transpose(-1, -2))
            C = torch.einsum("i,bin,bip->bnp", wc, dX, dY)
            Lc = torch.linalg.cholesky_ex(S)[0]
            v = data[name][:, j, :] - ybar
            alpha_v = torch.cholesky_solve(v.unsqueeze(-1), Lc)
            K = torch.cholesky_solve(C.transpose(-1, -2), Lc).transpose(-1, -2)
            logdet = 2.0 * torch.log(torch.diagonal(Lc, dim1=-2, dim2=-1)).sum(-1)
            term = -0.5 * (
                (v.unsqueeze(-1) * alpha_v).sum((-2, -1))
                + logdet
                + v.shape[-1] * _LOG2PI
            )
            ll = ll + term
            ll_ch[name] = ll_ch[name] + term
            n_used[name] = n_used[name] + 1
            m = m + (K @ v.unsqueeze(-1)).squeeze(-1)
            P = P - K @ S @ K.transpose(-1, -2)
            P = 0.5 * (P + P.transpose(-1, -2))
        if k + 1 < n_steps:
            X, wm, wc = _sigma_points(m, P, alpha, beta, kappa)
            Xp = torch.stack([f(X[:, i, :], k) for i in range(X.shape[1])], dim=1)
            m = (wm.view(1, -1, 1) * Xp).sum(1)
            dX = Xp - m.unsqueeze(1)
            P = torch.einsum("i,bin,bim->bnm", wc, dX, dX) + Q
            P = 0.5 * (P + P.transpose(-1, -2))
    return FilterResult(ll, ll_ch, n_used)


@dataclass
class EnsembleKalmanResult:
    ensemble_mean: Tensor
    ensemble_cov: Tensor
    ensembles: Tensor | None = None


def ensemble_kalman_filter(
    f: TransitionFn,
    h: dict[str, ObservationFn],
    Q: Tensor,
    R: dict[str, Tensor],
    m0: Tensor,
    P0: Tensor,
    data: dict[str, Tensor],
    steps: dict[str, Tensor],
    n_steps: int,
    *,
    n_ensemble: int = 256,
    seed: int = 0,
    inflation: float = 1.0,
) -> EnsembleKalmanResult:
    """Stochastic (perturbed-observation) ensemble Kalman filter."""
    b, n = m0.shape
    dtype, device = m0.dtype, m0.device
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    def randn(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=gen, dtype=dtype).to(device)

    L0 = torch.linalg.cholesky(P0)
    LQ = torch.linalg.cholesky(Q)
    E = m0.unsqueeze(1) + torch.einsum(
        "bnm,bem->ben", L0, randn(b, n_ensemble, n)
    )
    sched = _nl_schedule([(k, v) for k, v in steps.items()])
    means, covs = [], []
    for k in range(n_steps):
        for name, j in sched.steps.get(k, ()):
            Y = torch.stack(
                [h[name](E[:, e, :], k) for e in range(n_ensemble)], dim=1
            )
            p = Y.shape[-1]
            LR = torch.linalg.cholesky(R[name])
            pert = torch.einsum("bpq,beq->bep", LR, randn(b, n_ensemble, p))
            Ebar = E.mean(1, keepdim=True)
            Ybar = Y.mean(1, keepdim=True)
            dE = (E - Ebar) * inflation
            dY = Y - Ybar
            Cxy = torch.einsum("ben,bep->bnp", dE, dY) / (n_ensemble - 1)
            Cyy = torch.einsum("bep,beq->bpq", dY, dY) / (n_ensemble - 1) + R[name]
            Lc = torch.linalg.cholesky_ex(0.5 * (Cyy + Cyy.transpose(-1, -2)))[0]
            innov = data[name][:, j, :].unsqueeze(1) + pert - Y
            gain = torch.cholesky_solve(Cxy.transpose(-1, -2), Lc).transpose(-1, -2)
            E = E + torch.einsum("bnp,bep->ben", gain, innov)
        means.append(E.mean(1))
        dE = E - E.mean(1, keepdim=True)
        covs.append(torch.einsum("ben,bem->bnm", dE, dE) / (n_ensemble - 1))
        if k + 1 < n_steps:
            E = torch.stack([f(E[:, e, :], k) for e in range(n_ensemble)], dim=1)
            E = E + torch.einsum("bnm,bem->ben", LQ, randn(b, n_ensemble, n))
    return EnsembleKalmanResult(
        torch.stack(means, dim=1), torch.stack(covs, dim=1), None
    )


@dataclass
class ParticleFilterResult:
    log_likelihood: Tensor
    mean: Tensor
    cov: Tensor
    ess: Tensor


def particle_filter(
    f: TransitionFn,
    h_logpdf: dict[str, Callable[[Tensor, Tensor, int], Tensor]],
    m0: Tensor,
    P0: Tensor,
    Q: Tensor,
    data: dict[str, Tensor],
    steps: dict[str, Tensor],
    n_steps: int,
    *,
    n_particles: int = 1024,
    seed: int = 0,
    resample_threshold: float = 0.5,
) -> ParticleFilterResult:
    """Bootstrap particle filter / SMC with systematic resampling.

    ``h_logpdf[name](particles, y, k)`` returns ``log p(y | x)`` per particle.
    Multirate by construction: a step with no scheduled observation simply
    propagates.
    """
    b, n = m0.shape
    dtype, device = m0.dtype, m0.device
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    def randn(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=gen, dtype=dtype).to(device)

    def rand(*shape: int) -> Tensor:
        return torch.rand(*shape, generator=gen, dtype=dtype).to(device)

    L0 = torch.linalg.cholesky(P0)
    LQ = torch.linalg.cholesky(Q)
    X = m0.unsqueeze(1) + torch.einsum("bnm,bem->ben", L0, randn(b, n_particles, n))
    logw = torch.full(
        (b, n_particles), -math.log(n_particles), dtype=dtype, device=device
    )
    ll = torch.zeros(b, dtype=dtype, device=device)
    sched = _nl_schedule([(k, v) for k, v in steps.items()])
    means, covs, esss = [], [], []
    for k in range(n_steps):
        for name, j in sched.steps.get(k, ()):
            lp = h_logpdf[name](X, data[name][:, j, :], k)
            logw = logw + lp
            mx = logw.max(dim=1, keepdim=True).values
            lse = mx.squeeze(1) + torch.log(torch.exp(logw - mx).sum(1))
            ll = ll + lse
            logw = logw - lse.unsqueeze(1)
        w = torch.exp(logw)
        ess = 1.0 / (w**2).sum(1)
        mean = (w.unsqueeze(-1) * X).sum(1)
        dX = X - mean.unsqueeze(1)
        cov = torch.einsum("be,ben,bem->bnm", w, dX, dX)
        means.append(mean)
        covs.append(cov)
        esss.append(ess)
        need = ess < resample_threshold * n_particles
        if bool(need.any()):
            u0 = rand(b, 1) / n_particles
            positions = u0 + torch.arange(
                n_particles, dtype=dtype, device=device
            ).unsqueeze(0) / n_particles
            cum = torch.cumsum(w, dim=1)
            idx = torch.searchsorted(cum.contiguous(), positions.contiguous())
            idx = idx.clamp(max=n_particles - 1)
            keep = need.view(b, 1, 1)
            Xr = torch.gather(X, 1, idx.unsqueeze(-1).expand(-1, -1, n))
            X = torch.where(keep, Xr, X)
            logw = torch.where(
                need.view(b, 1),
                torch.full_like(logw, -math.log(n_particles)),
                logw,
            )
        if k + 1 < n_steps:
            X = torch.stack([f(X[:, e, :], k) for e in range(n_particles)], dim=1)
            X = X + torch.einsum("bnm,bem->ben", LQ, randn(b, n_particles, n))
    return ParticleFilterResult(
        ll,
        torch.stack(means, dim=1),
        torch.stack(covs, dim=1),
        torch.stack(esss, dim=1),
    )
