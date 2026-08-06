"""Equations **T1--T3** of ``thesis_contract.tex`` sec. 0.3, as an exact
batched linear--Gaussian state-space model on native clocks.

T1 -- latent dynamics (3 regions, directed coupling, one conduction delay)::

    dx(t) = A(theta) x(t) dt + B u(t - tau_u) dt + Q^{1/2} dW,   x in R^3

realised here as a delay-differential drift

    xdot_i = -x_i / T_i + sum_j C_ij(theta) x_j(t - tau) + [B u(t - tau_u)]_i

with ``C`` the directed coupling ``1->2``, ``2->3``, ``3->1`` and a single
unknown network conduction delay ``tau`` (thesis: *"one unknown conduction
delay"*; ``theta`` holds *"selected coupling gains and the network delay"*).

T2 -- EEG-like instantaneous mixing at ``Delta_E = 1 ms``::

    y_E[k] = L(ell) x(k Delta_E) + eps_E[k]

T3 -- fMRI-like hemodynamic convolution at ``Delta_B = 1 s``::

    y_B[n] = M \\int_0^{T_h} h(s; rho) x(n Delta_B - s) ds + eps_B[n]

Discretisation
--------------
The *discrete* model is the ground truth (thesis: *"For a linear--Gaussian
discretization ..."*).  Over one base step ``Delta`` the local (diagonal) part
is integrated exactly by matrix exponential and the delayed-coupling plus input
terms are held constant (zero-order hold):

    x_{k+1} = Phi x_k + Gamma (C x^tau_k + B u^d_k) + w_k
    Phi     = diag(exp(-Delta/T_i))
    Gamma   = diag(T_i (1 - exp(-Delta/T_i)))
    Cov(w)  = diag(q_i T_i / 2 (1 - exp(-2 Delta/T_i)))     (Van Loan, exact)

``x^tau_k = sum_p w_p(tau) x_{k-p}`` uses a Gaussian-windowed-sinc fractional
delay so that ``tau`` is a *continuous* parameter with a smooth derivative --
required for the Fisher information of a delay to exist.  ``h(s; rho)`` is
realised exactly as a cascade of first-order lags: a chain of ``K`` stages with
time constant ``beta`` has stage-``k`` impulse response
``s^{k-1} e^{-s/beta} / (Gamma(k) beta^k)``, i.e. the Gamma density, so
``h(s;rho) = g_B [gamma(s; k_peak, beta) - c gamma(s; k_under, beta)]`` needs
no explicit ``T_h`` truncation (the rational realisation is the ``T_h -> inf``
limit; truncation error is below 1e-9 for the configured horizons).

Parameter vector ``eta = (theta, ell, rho)``, 9 entries, carried in an
**unconstrained** coordinate ``u`` with an independent Gaussian prior.  All
Fisher matrices are reported in the *prior-standardised* basis
``(u - u0)/sd_prior``: this makes the condition number meaningful across
parameters with different physical units, and makes ``I_prior`` exactly the
identity so that ``lambda_min(I_likelihood)`` -- the *minimum non-prior
eigenvalue* -- is directly readable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Sequence

import numpy as np
import torch
from torch import Tensor

from .filters import LinearGaussianSSM, ObservationChannel, deterministic_response
from .types import DTYPE, resolve_device, torch_dtype

__all__ = [
    "PARAM_NAMES",
    "THETA_NAMES",
    "ParamSpec",
    "Protocol",
    "SystemConfig",
    "SystemModel",
    "build_protocol",
    "calibrate_observation_noise",
    "calibrate_stimulus_amplitude",
    "coarse_config",
    "coarsen_protocol",
    "structured_left_mul",
    "decimate_eeg",
    "default_eta",
    "make_model",
    "natural_from_unconstrained",
    "prior_mean",
    "prior_sd",
    "unconstrained_from_natural",
]


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------

Transform = Literal["identity", "log", "scaled_logit"]


@dataclass(frozen=True)
class ParamSpec:
    name: str
    group: Literal["theta", "ell", "rho"]
    transform: Transform
    prior_mean_u: float          # prior mean on the unconstrained scale
    prior_sd_u: float            # prior sd on the unconstrained scale
    scale: float = 1.0           # for scaled_logit: natural = scale * sigmoid(u)
    units: str = "dimensionless"
    description: str = ""


PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec("a21", "theta", "identity", 30.0, 10.0, units="1/s",
              description="directed coupling gain region 1 -> region 2"),
    ParamSpec("a32", "theta", "identity", 25.0, 10.0, units="1/s",
              description="directed coupling gain region 2 -> region 3"),
    ParamSpec("a13", "theta", "identity", -18.0, 10.0, units="1/s",
              description="directed coupling gain region 3 -> region 1"),
    ParamSpec("tau", "theta", "log", math.log(0.012), 0.25, units="s",
              description="network conduction delay"),
    ParamSpec("gain_eeg", "ell", "log", 0.0, 0.30, units="dimensionless",
              description="EEG lead-field global gain"),
    ParamSpec("tilt_eeg", "ell", "identity", 0.0, 0.20, units="dimensionless",
              description="EEG lead-field geometric (electrode-placement) tilt"),
    ParamSpec("beta_hrf", "rho", "log", math.log(1.6), 0.20, units="s",
              description="hemodynamic cascade time constant"),
    ParamSpec("c_under", "rho", "scaled_logit", 0.0, 0.60, scale=0.5,
              units="dimensionless", description="HRF undershoot weight"),
    ParamSpec("gain_bold", "rho", "log", 0.0, 0.30, units="dimensionless",
              description="BOLD global gain"),
)

PARAM_NAMES: tuple[str, ...] = tuple(p.name for p in PARAMS)
THETA_NAMES: tuple[str, ...] = tuple(p.name for p in PARAMS if p.group == "theta")
PARAM_INDEX = {p.name: i for i, p in enumerate(PARAMS)}
N_PARAM = len(PARAMS)


def prior_mean_u() -> np.ndarray:
    return np.array([p.prior_mean_u for p in PARAMS], dtype=float)


def prior_sd_u() -> np.ndarray:
    return np.array([p.prior_sd_u for p in PARAMS], dtype=float)


prior_mean = prior_mean_u
prior_sd = prior_sd_u


def natural_from_unconstrained(u: Tensor | np.ndarray) -> dict[str, Any]:
    """Map the unconstrained coordinate to physical parameters."""
    out: dict[str, Any] = {}
    for i, p in enumerate(PARAMS):
        ui = u[..., i]
        if p.transform == "identity":
            out[p.name] = ui
        elif p.transform == "log":
            out[p.name] = torch.exp(ui) if isinstance(ui, Tensor) else np.exp(ui)
        else:  # scaled_logit
            sig = torch.sigmoid(ui) if isinstance(ui, Tensor) else 1 / (1 + np.exp(-ui))
            out[p.name] = p.scale * sig
    return out


def unconstrained_from_natural(nat: dict[str, float]) -> np.ndarray:
    u = np.zeros(N_PARAM)
    for i, p in enumerate(PARAMS):
        v = float(nat[p.name])
        if p.transform == "identity":
            u[i] = v
        elif p.transform == "log":
            u[i] = math.log(v)
        else:
            r = v / p.scale
            u[i] = math.log(r / (1.0 - r))
    return u


def default_eta() -> np.ndarray:
    """The generative truth used by the benchmark (== the prior mean)."""
    return prior_mean_u()


# --------------------------------------------------------------------------
# Fixed instrument geometry (deterministic, not fitted)
# --------------------------------------------------------------------------

LEAD_FIELD_L0 = np.array(
    [[1.00, 0.30, 0.10],
     [0.20, 1.00, 0.25],
     [0.15, 0.35, 1.00],
     [0.60, 0.55, 0.50]],
    dtype=float,
)
LEAD_FIELD_L1 = np.array(
    [[0.00, 0.00, 0.00],
     [0.50, -0.50, 0.00],
     [0.00, 0.50, -0.50],
     [0.30, 0.00, -0.30]],
    dtype=float,
)
BOLD_MIXING_M = np.array(
    [[1.00, 0.12, 0.05],
     [0.10, 1.00, 0.12],
     [0.06, 0.10, 1.00]],
    dtype=float,
)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SystemConfig:
    """Everything about the instrument and protocol that is *not* estimated."""

    dt: float = 1e-3                       # base clock, seconds
    dt_eeg: float = 1e-3                   # Delta_E (T2)
    dt_bold: float = 1.0                   # Delta_B (T3)
    n_regions: int = 3
    membrane_tau: tuple[float, ...] = (0.020, 0.022, 0.025)
    q_process: tuple[float, ...] = (1.0, 1.0, 1.0)
    n_delay_taps: int = 26                 # history depth D (0 => no delay line)
    sinc_sigma: float = 2.0
    hrf_stages: int = 8
    hrf_peak_stage: int = 4                # 1-based tap giving the positive lobe
    hrf_under_stage: int = 8               # 1-based tap giving the undershoot
    epoch_seconds: float = 12.0
    n_epochs: int = 10
    input_delay: float = 0.005             # tau_u, known/calibrated
    input_region: int = 0
    impulse_region: int = 1
    sigma_eeg: float = 1.0
    sigma_bold: float = 1.0
    bold_first_sample: float = 1.0         # seconds
    p0_jitter: float = 1e-9
    dtype: str = "float64"
    device: str | None = None

    @property
    def n_steps(self) -> int:
        return int(round(self.epoch_seconds / self.dt))

    @property
    def eeg_stride(self) -> int:
        return max(1, int(round(self.dt_eeg / self.dt)))

    @property
    def bold_stride(self) -> int:
        return max(1, int(round(self.dt_bold / self.dt)))

    @property
    def n_state(self) -> int:
        return self.n_regions * (self.n_delay_taps + 1) + self.n_regions * self.hrf_stages

    @property
    def hrf_offset(self) -> int:
        return self.n_regions * (self.n_delay_taps + 1)

    def eeg_steps(self) -> np.ndarray:
        return np.arange(0, self.n_steps, self.eeg_stride, dtype=np.int64)

    def bold_steps(self) -> np.ndarray:
        first = int(round(self.bold_first_sample / self.dt)) - 1
        first = max(first, 0)
        return np.arange(first, self.n_steps, self.bold_stride, dtype=np.int64)


def coarse_config(cfg: SystemConfig) -> SystemConfig:
    """The **naive resampling** configuration: one common 1 s tensor.

    Everything is forced onto ``Delta_B``.  With a 1 s base step there is no
    delay line at all (``n_delay_taps = 0``): a 12 ms conduction delay is not
    representable, so ``d mu / d tau == 0`` and the design is *structurally*
    rank deficient in ``tau``.  That is the failure this baseline exists to
    expose, not a strawman added to it.
    """
    return replace(
        cfg,
        dt=cfg.dt_bold,
        dt_eeg=cfg.dt_bold,
        n_delay_taps=0,
        bold_first_sample=cfg.bold_first_sample,
    )


# --------------------------------------------------------------------------
# Stimulus protocol
# --------------------------------------------------------------------------


@dataclass
class Protocol:
    """Known deterministic drive ``u(t)`` per epoch, on the base clock."""

    u: np.ndarray               # [n_epochs, n_steps] baseline drive into input_region
    impulse: np.ndarray         # [n_epochs, n_steps] extra calibrated write
    description: str
    event_times: list[list[float]] = field(default_factory=list)
    impulse_times: list[list[float]] = field(default_factory=list)

    @property
    def energy(self) -> float:
        return float((self.u**2).sum() + (self.impulse**2).sum())


def build_protocol(
    cfg: SystemConfig,
    *,
    seed: int,
    mean_isi: float = 1.6,
    event_duration: float = 0.020,
    amplitude: float = 1.0,
    impulse_amplitude: float = 8.0,
    impulse_duration: float = 0.005,
    impulses_per_epoch: int = 1,
    scale_baseline: float = 1.0,
) -> Protocol:
    """Deterministic pseudo-random event train + optional calibrated impulses."""
    rng = np.random.default_rng(seed)
    n_steps = cfg.n_steps
    u = np.zeros((cfg.n_epochs, n_steps))
    imp = np.zeros((cfg.n_epochs, n_steps))
    ev_times: list[list[float]] = []
    imp_times: list[list[float]] = []
    dur = max(1, int(round(event_duration / cfg.dt)))
    idur = max(1, int(round(impulse_duration / cfg.dt)))
    for e in range(cfg.n_epochs):
        t = 0.5
        times = []
        while t < cfg.epoch_seconds - 0.5:
            k = int(round(t / cfg.dt))
            u[e, k : k + dur] = amplitude * scale_baseline
            times.append(t)
            t += float(rng.exponential(mean_isi)) + event_duration + 0.2
        ev_times.append(times)
        its = []
        for m in range(impulses_per_epoch):
            frac = (m + 1) / (impulses_per_epoch + 1)
            ti = 0.35 + frac * (cfg.epoch_seconds - 0.9)
            k = int(round(ti / cfg.dt))
            imp[e, k : k + idur] = impulse_amplitude
            its.append(ti)
        imp_times.append(its)
    return Protocol(u, imp, "poisson_events", ev_times, imp_times)


def coarsen_protocol(proto: Protocol, cfg: SystemConfig, coarse: SystemConfig) -> Protocol:
    """1 s block-average of the drive -- what naive resampling actually keeps."""
    stride = int(round(coarse.dt / cfg.dt))
    n_out = coarse.n_steps

    def blk(a: np.ndarray) -> np.ndarray:
        out = np.zeros((a.shape[0], n_out))
        for n in range(n_out):
            seg = a[:, n * stride : (n + 1) * stride]
            if seg.size:
                out[:, n] = seg.mean(axis=1)
        return out

    return Protocol(blk(proto.u), blk(proto.impulse), proto.description + "|1s_block_mean",
                    proto.event_times, proto.impulse_times)


# --------------------------------------------------------------------------
# Fractional delay kernel
# --------------------------------------------------------------------------


def _sinc(x: Tensor) -> Tensor:
    return torch.sinc(x)


def _dsinc(x: Tensor) -> Tensor:
    """d/dx sinc(x) with a stable series near 0."""
    pi = math.pi
    small = x.abs() < 1e-4
    xs = torch.where(small, torch.ones_like(x), x)
    big = (torch.cos(pi * xs) * pi * xs - torch.sin(pi * xs)) / (pi * xs * xs)
    ser = -(pi**2) * x / 3.0
    return torch.where(small, ser, big)


def delay_weights(tau: Tensor, cfg: SystemConfig) -> Tensor:
    """Normalised Gaussian-windowed-sinc fractional-delay weights ``w_p(tau)``.

    Returns ``[..., D+1]``.  With ``D = 0`` the weight is identically 1 and the
    kernel has *zero* derivative in ``tau`` -- the structural signature of a
    clock too coarse to represent the delay.
    """
    D = cfg.n_delay_taps
    if D == 0:
        return torch.ones(*tau.shape, 1, dtype=tau.dtype, device=tau.device)
    p = torch.arange(D + 1, dtype=tau.dtype, device=tau.device)
    d = (tau / cfg.dt).unsqueeze(-1)
    x = d - p
    raw = _sinc(x) * torch.exp(-0.5 * (x / cfg.sinc_sigma) ** 2)
    return raw / raw.sum(-1, keepdim=True)


def delay_weights_grad(tau: Tensor, cfg: SystemConfig) -> Tensor:
    """Analytic ``d w_p / d tau`` (hand-derived; cross-checked against autodiff)."""
    D = cfg.n_delay_taps
    if D == 0:
        return torch.zeros(*tau.shape, 1, dtype=tau.dtype, device=tau.device)
    p = torch.arange(D + 1, dtype=tau.dtype, device=tau.device)
    d = (tau / cfg.dt).unsqueeze(-1)
    x = d - p
    g = torch.exp(-0.5 * (x / cfg.sinc_sigma) ** 2)
    raw = _sinc(x) * g
    draw = (_dsinc(x) - _sinc(x) * x / cfg.sinc_sigma**2) * g
    s = raw.sum(-1, keepdim=True)
    ds = draw.sum(-1, keepdim=True)
    return (draw * s - raw * ds) / (s * s) / cfg.dt


# --------------------------------------------------------------------------
# Model assembly
# --------------------------------------------------------------------------


def structured_left_mul(F: Tensor, cfg: SystemConfig):
    """A fast, exact ``X -> F @ X`` that uses the known structure of ``F``.

    Only ``n_regions`` rows (the ``x`` update) are dense; the delay line is a
    permutation (shift register) and the hemodynamic cascade is two-diagonal.
    For the default configuration this is ~45x fewer multiplications than the
    dense product and it dominates the cost of the Riccati recursion.
    ``tests/infer/test_filters.py`` asserts exact agreement with ``F @ X``.
    """
    R, D, K = cfg.n_regions, cfg.n_delay_taps, cfg.hrf_stages
    nh = R * (D + 1)
    off = cfg.hrf_offset
    Fx = F[:, 0:R, 0:nh]
    a = F[:, off, off].reshape(-1, 1, 1)

    def mul(X: Tensor) -> Tensor:
        parts = [Fx @ X[:, 0:nh, :]]
        if D:
            parts.append(X[:, 0 : R * D, :])
        cur = X[:, off : off + R * K, :]
        prev = torch.cat([X[:, 0:R, :], cur[:, 0 : R * (K - 1), :]], dim=1)
        parts.append(torch.addcmul(prev, a, cur - prev))
        return torch.cat(parts, dim=1)

    return mul


@dataclass
class SystemModel:
    """Assembled batched operators for one parameter batch and one protocol."""

    cfg: SystemConfig
    F: Tensor                    # [B, n, n]
    Q: Tensor                    # [B, n, n]
    H_eeg: Tensor                # [B, p_E, n]
    H_bold: Tensor               # [B, p_B, n]
    R_eeg: Tensor
    R_bold: Tensor
    m0: Tensor
    P0: Tensor
    inputs: Tensor               # [B*, n_epochs, n_steps, n]
    eeg_steps: Tensor
    bold_steps: Tensor

    @property
    def n(self) -> int:
        return int(self.F.shape[-1])

    def ssm(
        self,
        channels: Sequence[str] = ("eeg", "bold"),
        *,
        epoch: int | None = None,
        eeg_steps: Tensor | None = None,
        with_inputs: bool = True,
        fast: bool = True,
    ) -> LinearGaussianSSM:
        chans = []
        if "eeg" in channels:
            chans.append(
                ObservationChannel(
                    "eeg", self.H_eeg, self.R_eeg,
                    self.eeg_steps if eeg_steps is None else eeg_steps,
                )
            )
        if "bold" in channels:
            chans.append(ObservationChannel("bold", self.H_bold, self.R_bold, self.bold_steps))
        F, Q, m0, P0 = self.F, self.Q, self.m0, self.P0
        if not with_inputs:
            inp = None
        elif epoch is not None:
            inp = self.inputs[:, epoch]
        else:
            # epoch=None means "batch over epochs": flatten [B, E, T, n] to
            # [B*E, T, n] and repeat the operators to match, so that row b*E+e
            # is subject b's epoch e.
            E = self.inputs.shape[1]
            inp = self.inputs.reshape(-1, *self.inputs.shape[2:])
            if E > 1:
                F = F.repeat_interleave(E, 0)
                Q = Q.repeat_interleave(E, 0)
                m0 = m0.repeat_interleave(E, 0)
                P0 = P0.repeat_interleave(E, 0)
        return LinearGaussianSSM(
            F=F, Q=Q, m0=m0, P0=P0,
            channels=chans, n_steps=self.cfg.n_steps, inputs=inp,
            left_mul=structured_left_mul(F, self.cfg) if fast else None,
        )

    def multiepoch_ssm(
        self, channels: Sequence[str] = ("eeg", "bold"), *, eeg_steps: Tensor | None = None
    ) -> LinearGaussianSSM:
        """SSM whose ``inputs`` keep the epoch axis, for the shared-Riccati filter."""
        ssm = self.ssm(channels, epoch=0, eeg_steps=eeg_steps)
        ssm.inputs = self.inputs
        return ssm


def _discrete_lyapunov(F: Tensor, Q: Tensor, n_doubling: int = 26) -> Tensor:
    """Stationary covariance ``P = F P F^T + Q`` by squaring (differentiable)."""
    A = F
    P = Q
    for _ in range(n_doubling):
        P = P + A @ P @ A.transpose(-1, -2)
        A = A @ A
        if float(A.detach().abs().max()) < 1e-30:
            break
    return 0.5 * (P + P.transpose(-1, -2))


def _as_batch(u: Tensor) -> Tensor:
    return u if u.dim() == 2 else u.reshape(1, -1)


def build_operators(
    u: Tensor,
    cfg: SystemConfig,
) -> dict[str, Tensor]:
    """Assemble ``F, Q, H_eeg, H_bold`` for a batch of unconstrained parameters."""
    u = _as_batch(u)
    B = u.shape[0]
    dt = cfg.dt
    R = cfg.n_regions
    D = cfg.n_delay_taps
    K = cfg.hrf_stages
    n = cfg.n_state
    dtype, device = u.dtype, u.device
    nat = natural_from_unconstrained(u)

    Tm = torch.tensor(cfg.membrane_tau, dtype=dtype, device=device)
    qv = torch.tensor(cfg.q_process, dtype=dtype, device=device)
    phi = torch.exp(-dt / Tm)                       # [R]
    gam = Tm * (1.0 - phi)                          # [R]
    sw = qv * Tm / 2.0 * (1.0 - torch.exp(-2.0 * dt / Tm))

    # directed coupling C(theta): 1->2, 2->3, 3->1
    C = torch.zeros(B, R, R, dtype=dtype, device=device)
    C[:, 1, 0] = nat["a21"]
    C[:, 2, 1] = nat["a32"]
    C[:, 0, 2] = nat["a13"]

    w = delay_weights(nat["tau"], cfg)              # [B, D+1]
    GamC = gam.reshape(1, R, 1) * C                 # [B,R,R]

    F = torch.zeros(B, n, n, dtype=dtype, device=device)
    # x_{k+1} row block
    F[:, 0:R, 0:R] = torch.diag(phi).unsqueeze(0)
    for p in range(D + 1):
        F[:, 0:R, R * p : R * (p + 1)] = (
            F[:, 0:R, R * p : R * (p + 1)] + GamC * w[:, p].reshape(B, 1, 1)
        )
    # history shift register
    eyeR = torch.eye(R, dtype=dtype, device=device).unsqueeze(0)
    for j in range(D):
        F[:, R * (j + 1) : R * (j + 2), R * j : R * (j + 1)] = eyeR
    # hemodynamic cascade (exact ZOH per stage)
    a = torch.exp(-dt / nat["beta_hrf"]).reshape(B, 1, 1)
    off = cfg.hrf_offset
    F[:, off : off + R, 0:R] = (1.0 - a) * eyeR
    F[:, off : off + R, off : off + R] = a * eyeR
    for s in range(1, K):
        F[:, off + R * s : off + R * (s + 1), off + R * (s - 1) : off + R * s] = (
            1.0 - a
        ) * eyeR
        F[:, off + R * s : off + R * (s + 1), off + R * s : off + R * (s + 1)] = a * eyeR

    Q = torch.zeros(B, n, n, dtype=dtype, device=device)
    Q[:, 0:R, 0:R] = torch.diag(sw).unsqueeze(0)

    # EEG head (T2): instantaneous mixing of x_k only
    L0 = torch.tensor(LEAD_FIELD_L0, dtype=dtype, device=device)
    L1 = torch.tensor(LEAD_FIELD_L1, dtype=dtype, device=device)
    gE = nat["gain_eeg"].reshape(B, 1, 1)
    L = gE * (L0.unsqueeze(0) + nat["tilt_eeg"].reshape(B, 1, 1) * L1.unsqueeze(0))
    H_eeg = torch.zeros(B, L0.shape[0], n, dtype=dtype, device=device)
    H_eeg[:, :, 0:R] = L

    # BOLD head (T3): M [gamma(k_peak) - c gamma(k_under)] * g_B
    Mmix = torch.tensor(BOLD_MIXING_M, dtype=dtype, device=device)
    gB = nat["gain_bold"].reshape(B, 1, 1)
    c = nat["c_under"].reshape(B, 1, 1)
    H_bold = torch.zeros(B, Mmix.shape[0], n, dtype=dtype, device=device)
    sp = off + R * (cfg.hrf_peak_stage - 1)
    su = off + R * (cfg.hrf_under_stage - 1)
    H_bold[:, :, sp : sp + R] = gB * Mmix.unsqueeze(0)
    H_bold[:, :, su : su + R] = H_bold[:, :, su : su + R] - gB * c * Mmix.unsqueeze(0)

    return {"F": F, "Q": Q, "H_eeg": H_eeg, "H_bold": H_bold, "gam": gam}


def build_operator_derivatives(u: Tensor, cfg: SystemConfig) -> dict[str, Tensor]:
    """**Analytic** ``dF/du_i``, ``dH/du_i`` (hand-derived closed forms).

    ``tests/infer/test_fisher.py`` asserts these agree with reverse-mode
    autodiff to ~1e-9; the two are then genuinely independent code paths for
    the analytic and numerical Fisher computations of T4.
    """
    u = _as_batch(u)
    B = u.shape[0]
    R = cfg.n_regions
    D = cfg.n_delay_taps
    K = cfg.hrf_stages
    n = cfg.n_state
    dt = cfg.dt
    dtype, device = u.dtype, u.device
    nat = natural_from_unconstrained(u)
    Tm = torch.tensor(cfg.membrane_tau, dtype=dtype, device=device)
    gam = Tm * (1.0 - torch.exp(-dt / Tm))
    w = delay_weights(nat["tau"], cfg)
    dw_dtau = delay_weights_grad(nat["tau"], cfg)

    dF = torch.zeros(B, N_PARAM, n, n, dtype=dtype, device=device)
    dH_e = torch.zeros(B, N_PARAM, LEAD_FIELD_L0.shape[0], n, dtype=dtype, device=device)
    dH_b = torch.zeros(B, N_PARAM, BOLD_MIXING_M.shape[0], n, dtype=dtype, device=device)

    # --- coupling gains: dC/da is a single unit entry, identity transform
    for name, (i, j) in (("a21", (1, 0)), ("a32", (2, 1)), ("a13", (0, 2))):
        pi = PARAM_INDEX[name]
        E = torch.zeros(R, R, dtype=dtype, device=device)
        E[i, j] = 1.0
        GamE = (gam.reshape(R, 1) * E).unsqueeze(0)     # [1,R,R]
        for p in range(D + 1):
            dF[:, pi, 0:R, R * p : R * (p + 1)] = GamE * w[:, p].reshape(B, 1, 1)

    # --- delay: enters only through w_p(tau); chain rule for u = log tau
    pi = PARAM_INDEX["tau"]
    C = torch.zeros(B, R, R, dtype=dtype, device=device)
    C[:, 1, 0] = nat["a21"]
    C[:, 2, 1] = nat["a32"]
    C[:, 0, 2] = nat["a13"]
    GamC = gam.reshape(1, R, 1) * C
    dtau_du = nat["tau"].reshape(B, 1, 1)               # d tau / d log tau = tau
    for p in range(D + 1):
        dF[:, pi, 0:R, R * p : R * (p + 1)] = (
            GamC * dw_dtau[:, p].reshape(B, 1, 1) * dtau_du
        )

    # --- HRF time constant: a = exp(-dt/beta); da/dlogbeta = (dt/beta) a
    pi = PARAM_INDEX["beta_hrf"]
    beta = nat["beta_hrf"]
    a = torch.exp(-dt / beta)
    da = (dt / beta) * a
    da = da.reshape(B, 1, 1)
    eyeR = torch.eye(R, dtype=dtype, device=device).unsqueeze(0)
    off = cfg.hrf_offset
    dF[:, pi, off : off + R, 0:R] = -da * eyeR
    dF[:, pi, off : off + R, off : off + R] = da * eyeR
    for s in range(1, K):
        dF[:, pi, off + R * s : off + R * (s + 1), off + R * (s - 1) : off + R * s] = (
            -da * eyeR
        )
        dF[:, pi, off + R * s : off + R * (s + 1), off + R * s : off + R * (s + 1)] = (
            da * eyeR
        )

    # --- observation nuisances
    L0 = torch.tensor(LEAD_FIELD_L0, dtype=dtype, device=device)
    L1 = torch.tensor(LEAD_FIELD_L1, dtype=dtype, device=device)
    gE = nat["gain_eeg"].reshape(B, 1, 1)
    tilt = nat["tilt_eeg"].reshape(B, 1, 1)
    dH_e[:, PARAM_INDEX["gain_eeg"], :, 0:R] = gE * (L0.unsqueeze(0) + tilt * L1.unsqueeze(0))
    dH_e[:, PARAM_INDEX["tilt_eeg"], :, 0:R] = gE * L1.unsqueeze(0)

    Mmix = torch.tensor(BOLD_MIXING_M, dtype=dtype, device=device)
    gB = nat["gain_bold"].reshape(B, 1, 1)
    c = nat["c_under"]
    sp = off + R * (cfg.hrf_peak_stage - 1)
    su = off + R * (cfg.hrf_under_stage - 1)
    pi = PARAM_INDEX["gain_bold"]
    dH_b[:, pi, :, sp : sp + R] = gB * Mmix.unsqueeze(0)
    dH_b[:, pi, :, su : su + R] = -gB * c.reshape(B, 1, 1) * Mmix.unsqueeze(0)
    pi = PARAM_INDEX["c_under"]
    spec = PARAMS[PARAM_INDEX["c_under"]]
    sig = torch.sigmoid(u[:, PARAM_INDEX["c_under"]])
    dc_du = (spec.scale * sig * (1 - sig)).reshape(B, 1, 1)
    dH_b[:, pi, :, su : su + R] = -gB * dc_du * Mmix.unsqueeze(0)

    return {"dF": dF, "dH_eeg": dH_e, "dH_bold": dH_b}


def build_inputs(
    cfg: SystemConfig,
    proto: Protocol,
    *,
    include_impulse: bool = False,
    dtype: torch.dtype = DTYPE,
    device: torch.device | str | None = None,
) -> Tensor:
    """Deterministic state increment ``b_k = Gamma B u(k - k_u)`` per epoch."""
    device = resolve_device(device)
    R = cfg.n_regions
    n = cfg.n_state
    Tm = np.asarray(cfg.membrane_tau, dtype=float)
    gam = Tm * (1.0 - np.exp(-cfg.dt / Tm))
    lag = int(round(cfg.input_delay / cfg.dt))
    drive = np.zeros((cfg.n_epochs, cfg.n_steps, R))
    for e in range(cfg.n_epochs):
        ub = np.zeros(cfg.n_steps)
        if lag:
            ub[lag:] = proto.u[e, : cfg.n_steps - lag]
        else:
            ub[:] = proto.u[e]
        drive[e, :, cfg.input_region] += ub
        if include_impulse:
            ui = np.zeros(cfg.n_steps)
            if lag:
                ui[lag:] = proto.impulse[e, : cfg.n_steps - lag]
            else:
                ui[:] = proto.impulse[e]
            drive[e, :, cfg.impulse_region] += ui
    drive = drive * gam.reshape(1, 1, R)
    out = np.zeros((cfg.n_epochs, cfg.n_steps, n))
    out[:, :, 0:R] = drive
    return torch.tensor(out, dtype=dtype, device=device)


def make_model(
    u: Tensor | np.ndarray,
    cfg: SystemConfig,
    proto: Protocol,
    *,
    include_impulse: bool = False,
    device: torch.device | str | None = None,
    dtype: torch.dtype | str | None = None,
) -> SystemModel:
    dtype = torch_dtype(dtype or cfg.dtype)
    device = resolve_device(device if device is not None else cfg.device)
    if not isinstance(u, Tensor):
        u = torch.tensor(np.asarray(u, dtype=float), dtype=dtype, device=device)
    u = _as_batch(u.to(dtype=dtype, device=device))
    ops = build_operators(u, cfg)
    B = u.shape[0]
    n = cfg.n_state
    P0 = _discrete_lyapunov(ops["F"], ops["Q"])
    P0 = P0 + cfg.p0_jitter * torch.eye(n, dtype=dtype, device=device)
    m0 = torch.zeros(B, n, dtype=dtype, device=device)
    pE = LEAD_FIELD_L0.shape[0]
    pB = BOLD_MIXING_M.shape[0]
    R_eeg = (cfg.sigma_eeg**2) * torch.eye(pE, dtype=dtype, device=device).expand(1, pE, pE)
    R_bold = (cfg.sigma_bold**2) * torch.eye(pB, dtype=dtype, device=device).expand(1, pB, pB)
    inputs = build_inputs(cfg, proto, include_impulse=include_impulse, dtype=dtype, device=device)
    return SystemModel(
        cfg=cfg,
        F=ops["F"], Q=ops["Q"],
        H_eeg=ops["H_eeg"], H_bold=ops["H_bold"],
        R_eeg=R_eeg.contiguous(), R_bold=R_bold.contiguous(),
        m0=m0, P0=P0,
        inputs=inputs.unsqueeze(0),
        eeg_steps=torch.tensor(cfg.eeg_steps(), device=device),
        bold_steps=torch.tensor(cfg.bold_steps(), device=device),
    )


def spectral_radius(cfg: SystemConfig, u: np.ndarray) -> float:
    ops = build_operators(torch.tensor(u, dtype=torch.float64).reshape(1, -1), cfg)
    ev = torch.linalg.eigvals(ops["F"][0])
    return float(ev.abs().max())


def calibrate_observation_noise(
    cfg: SystemConfig,
    u: np.ndarray,
    proto: Protocol,
    *,
    eeg_noise_ratio: float = 0.5,
    bold_noise_ratio: float = 0.5,
) -> SystemConfig:
    """Set ``sigma_E``/``sigma_B`` to a fixed fraction of the *signal* sd.

    The signal sd is the total (stochastic + input-driven) standard deviation of
    the noiseless observation at ``u``.  Fixing the ratio -- rather than the
    absolute noise -- keeps the EEG and BOLD arms on a comparable footing so
    that the five-design comparison is not decided by an arbitrary SNR choice.
    An explicit SNR sweep is part of the benchmark regimes.
    """
    mdl = make_model(u, replace(cfg, sigma_eeg=1.0, sigma_bold=1.0), proto)
    Pst = mdl.P0[0]
    var_e = torch.diagonal(mdl.H_eeg[0] @ Pst @ mdl.H_eeg[0].T).mean()
    var_b = torch.diagonal(mdl.H_bold[0] @ Pst @ mdl.H_bold[0].T).mean()
    ssm = mdl.ssm(("eeg", "bold"), epoch=0)
    det = deterministic_response(ssm)
    var_e = var_e + det["eeg"][0].var()
    var_b = var_b + det["bold"][0].var()
    return replace(
        cfg,
        sigma_eeg=float(eeg_noise_ratio * torch.sqrt(var_e)),
        sigma_bold=float(bold_noise_ratio * torch.sqrt(var_b)),
    )


def calibrate_stimulus_amplitude(
    cfg: SystemConfig,
    u: np.ndarray,
    proto: Protocol,
    *,
    evoked_ratio: float = 1.0,
) -> float:
    """Scale the known drive so the *evoked* EEG variance is a declared multiple
    of the ongoing (process-noise) EEG variance.

    T4 is a **mean-Jacobian** information: it is carried entirely by the
    deterministic, input-driven part of the response.  A design whose evoked
    response is negligible relative to ongoing activity has, by construction,
    almost no T4 information -- so the evoked fraction is a design decision that
    must be declared, not left to an arbitrary amplitude.  ``evoked_ratio=1``
    means evoked and ongoing EEG variance are equal.
    """
    mdl = make_model(u, replace(cfg, sigma_eeg=1.0, sigma_bold=1.0), proto)
    stoch = float(torch.diagonal(mdl.H_eeg[0] @ mdl.P0[0] @ mdl.H_eeg[0].T).mean())
    det = deterministic_response(mdl.ssm(("eeg",), epoch=None))
    evoked = float(det["eeg"].var())
    if evoked <= 0:
        raise ValueError("protocol produces no evoked response")
    return math.sqrt(evoked_ratio * stoch / evoked)


def decimate_eeg(y_eeg: Tensor, cfg: SystemConfig, coarse: SystemConfig) -> Tensor:
    """Take one EEG sample per BOLD repetition -- *naive* resampling."""
    fine_steps = cfg.eeg_steps()
    want = coarse.bold_steps() * int(round(coarse.dt / cfg.dt))
    idx = np.searchsorted(fine_steps, np.clip(want, 0, fine_steps[-1]))
    idx = np.clip(idx, 0, len(fine_steps) - 1)
    return y_eeg[..., idx, :]


def block_average_eeg(y_eeg: Tensor, cfg: SystemConfig, coarse: SystemConfig) -> Tensor:
    """Block-mean EEG within each repetition -- the *tuned* resampling control."""
    stride = int(round(coarse.dt / cfg.dt / cfg.eeg_stride))
    marks = coarse.bold_steps() * int(round(coarse.dt / cfg.dt))
    out = []
    for m in marks:
        lo = max(0, int(m) - stride + 1)
        hi = int(m) + 1
        out.append(y_eeg[..., lo:hi, :].mean(dim=-2))
    return torch.stack(out, dim=-2)
