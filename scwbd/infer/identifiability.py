"""The linear identifiability laboratory -- build-order item 2 of
``thesis_contract.tex`` sec. 0.6, and the *first gate* of sec. 11.

Five designs are compared, exactly as specified in sec. 0.3:

===========================  ==========================================
``eeg_only``                 (i)   EEG alone, 1 ms native clock
``fmri_only``                (ii)  fMRI alone, 1 s native clock
``joint_native``             (iii) joint inference on both native clocks
``joint_resampled``          (iv)  joint inference after naive resampling
``joint_native_impulse``     (v)   joint native-clock + one calibrated impulse
===========================  ==========================================

Two supplementary controls are added so the headline comparison cannot be won
by an artefact:

``joint_resampled_exactmodel``
    The *charitable* resampling baseline: EEG is still thrown away down to the
    1 s grid, but the correct fine-grained forward model is used.  The gap
    between this and ``joint_resampled`` separates **information lost** from
    **bias introduced by the wrong discretisation**.
``joint_native_impulse_matched``
    The impulse design with total input energy matched to ``joint_native`` by
    scaling the background drive.  Without this, design (v) could win merely by
    injecting more energy.

Reported per design: rank, condition number, minimum non-prior eigenvalue,
parameter-profile likelihoods, posterior correlations, interval coverage and
delay error -- with the prior contribution always separated.

An algebraic warning that the report repeats: with the modality-block-diagonal
form of T4, ``I_{EEG+BOLD} = I_EEG + I_BOLD >= I_EEG`` is an *identity*, not
evidence.  Adding a modality cannot decrease T4 information.  The discriminating
comparisons are therefore (a) native versus resampled, (b) the *size* of the
fMRI contribution to the preregistered subset, (c) impulse versus no impulse at
matched energy, and (d) whether extra information becomes calibrated recovery.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .calibration import interval_coverage
from .fisher import FisherReport, expected_fisher, monte_carlo_fisher, schur_information
from .filters import multiepoch_kalman_filter
from .linear_gaussian import (
    N_PARAM,
    PARAM_INDEX,
    PARAM_NAMES,
    PARAMS,
    THETA_NAMES,
    Protocol,
    SystemConfig,
    build_protocol,
    calibrate_observation_noise,
    calibrate_stimulus_amplitude,
    coarse_config,
    coarsen_protocol,
    decimate_eeg,
    default_eta,
    make_model,
    natural_from_unconstrained,
    prior_mean_u,
    prior_sd_u,
    structured_left_mul,
)
from .types import CoverageResult, as_builtin, seed_everything

__all__ = [
    "DESIGNS",
    "DesignSpec",
    "REGIMES",
    "Regime",
    "RecoveryResult",
    "build_design",
    "profile_likelihood",
    "recover",
    "run_benchmark",
    "run_fisher_table",
]


# --------------------------------------------------------------------------
# Designs and regimes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DesignSpec:
    name: str
    channels: tuple[str, ...]
    include_impulse: bool = False
    resample: bool = False          # decimate EEG onto the BOLD clock
    coarse_model: bool = False      # fit the 1 s discretisation (naive)
    energy_matched: bool = False
    primary: bool = True
    label: str = ""


DESIGNS: tuple[DesignSpec, ...] = (
    DesignSpec("eeg_only", ("eeg",), label="(i) EEG alone"),
    DesignSpec("fmri_only", ("bold",), label="(ii) fMRI alone"),
    DesignSpec("joint_native", ("eeg", "bold"), label="(iii) joint native-clock"),
    DesignSpec("joint_resampled", ("eeg", "bold"), resample=True, coarse_model=True,
               label="(iv) joint after naive resampling"),
    DesignSpec("joint_native_impulse", ("eeg", "bold"), include_impulse=True,
               label="(v) joint native-clock + one calibrated impulse"),
    DesignSpec("joint_resampled_exactmodel", ("eeg", "bold"), resample=True,
               primary=False, label="(iv-b) tuned resampling, exact forward model"),
    DesignSpec("joint_native_impulse_matched", ("eeg", "bold"), include_impulse=True,
               energy_matched=True, primary=False,
               label="(v-b) impulse at matched input energy"),
)


@dataclass(frozen=True)
class Regime:
    """A held-out simulation regime: a truth and an instrument SNR."""

    name: str
    theta_scale: float = 1.0
    tau_seconds: float = 0.012
    eeg_noise_ratio: float = 0.5
    bold_noise_ratio: float = 0.5
    evoked_ratio: float = 1.0
    description: str = ""

    def eta_true(self) -> np.ndarray:
        u = prior_mean_u().copy()
        for nm in ("a21", "a32", "a13"):
            u[PARAM_INDEX[nm]] *= self.theta_scale
        u[PARAM_INDEX["tau"]] = math.log(self.tau_seconds)
        return u


REGIMES: tuple[Regime, ...] = (
    Regime("reference", 1.0, 0.012, 0.5, 0.5, 1.0,
           "prior-mean coupling, 12 ms delay, evoked == ongoing variance"),
    Regime("weak_coupling_long_delay", 0.55, 0.017, 0.5, 0.5, 1.0,
           "coupling gains x0.55, 17 ms delay"),
    Regime("low_snr_short_delay", 1.25, 0.0085, 1.2, 1.2, 0.6,
           "coupling gains x1.25, 8.5 ms delay, 2.4x noise sd, weaker evoked drive"),
)


@dataclass
class BuiltDesign:
    spec: DesignSpec
    cfg: SystemConfig
    proto: Protocol
    channels: tuple[str, ...]
    include_impulse: bool
    eeg_steps: Tensor | None
    fit_cfg: SystemConfig
    fit_proto: Protocol
    fit_eeg_steps: Tensor | None
    notes: list[str] = field(default_factory=list)


def build_design(
    spec: DesignSpec, base_cfg: SystemConfig, regime: Regime, *, seed: int
) -> BuiltDesign:
    """Instantiate one design: instrument, protocol and estimator model."""
    u_true = regime.eta_true()
    proto = build_protocol(base_cfg, seed=seed)
    amp = calibrate_stimulus_amplitude(
        base_cfg, u_true, proto, evoked_ratio=regime.evoked_ratio
    )
    scale = 1.0
    if spec.energy_matched:
        # match total input energy to the impulse-free design by shrinking the
        # background drive; the impulse then costs information elsewhere.
        ref = build_protocol(base_cfg, seed=seed, amplitude=amp)
        imp = build_protocol(base_cfg, seed=seed, amplitude=amp,
                             impulse_amplitude=8.0 * amp)
        e_ref = float((ref.u**2).sum())
        e_imp = float((imp.impulse**2).sum())
        scale = math.sqrt(max(e_ref - e_imp, 1e-12) / max(e_ref, 1e-12))
    proto = build_protocol(
        base_cfg, seed=seed, amplitude=amp * scale, impulse_amplitude=8.0 * amp,
    )
    cfg = calibrate_observation_noise(
        base_cfg, u_true, proto,
        eeg_noise_ratio=regime.eeg_noise_ratio,
        bold_noise_ratio=regime.bold_noise_ratio,
    )
    notes = []
    if spec.energy_matched:
        notes.append(f"background drive scaled by {scale:.4f} to match input energy")

    fit_cfg, fit_proto = cfg, proto
    eeg_steps = None
    fit_eeg_steps = None
    if spec.resample:
        cc = coarse_config(cfg)
        want = cc.bold_steps() * int(round(cc.dt / cfg.dt))
        eeg_steps = torch.tensor(
            np.clip(want, 0, cfg.n_steps - 1), dtype=torch.long,
            device=torch.device(cfg.device or "cpu"),
        )
        notes.append("EEG decimated to one sample per repetition (naive resampling)")
    if spec.coarse_model:
        fit_cfg = calibrate_observation_noise(
            coarse_config(cfg), u_true, coarsen_protocol(proto, cfg, coarse_config(cfg)),
            eeg_noise_ratio=regime.eeg_noise_ratio,
            bold_noise_ratio=regime.bold_noise_ratio,
        )
        fit_cfg = replace(fit_cfg, sigma_eeg=cfg.sigma_eeg, sigma_bold=cfg.sigma_bold)
        fit_proto = coarsen_protocol(proto, cfg, fit_cfg)
        notes.append(
            "estimator uses the 1 s discretisation: with dt = Delta_B there is no "
            "delay line, so d mu / d tau == 0 and tau is structurally "
            "unidentifiable"
        )
    return BuiltDesign(
        spec, cfg, proto, spec.channels, spec.include_impulse, eeg_steps,
        fit_cfg, fit_proto, fit_eeg_steps, notes,
    )


# --------------------------------------------------------------------------
# Fisher table
# --------------------------------------------------------------------------


def run_fisher_table(
    base_cfg: SystemConfig,
    regime: Regime,
    *,
    seed: int,
    designs: Sequence[DesignSpec] = DESIGNS,
    theta_names: Sequence[str] = THETA_NAMES,
    with_joint_whitening: bool = True,
) -> dict[str, Any]:
    """T4 for every design in one regime."""
    u_true = regime.eta_true()
    out: dict[str, Any] = {}
    for spec in designs:
        bd = build_design(spec, base_cfg, regime, seed=seed)
        rec: dict[str, Any] = {"label": spec.label, "notes": list(bd.notes),
                               "primary": spec.primary}
        rep = expected_fisher(
            u_true, bd.cfg, bd.proto, design=spec.name, channels=bd.channels,
            include_impulse=bd.include_impulse, eeg_steps=bd.eeg_steps,
            method="analytic", joint_whitening=False, theta_names=theta_names,
        )
        rec["T4_block_diagonal"] = rep.to_dict()
        if with_joint_whitening and len(bd.channels) > 1:
            repj = expected_fisher(
                u_true, bd.cfg, bd.proto, design=spec.name, channels=bd.channels,
                include_impulse=bd.include_impulse, eeg_steps=bd.eeg_steps,
                method="analytic", joint_whitening=True, theta_names=theta_names,
            )
            rec["exact_joint_whitening"] = repj.to_dict()
        if spec.coarse_model:
            # the estimator's *own* information: this is where naive resampling
            # loses the delay structurally.
            repc = expected_fisher(
                u_true, bd.fit_cfg, bd.fit_proto, design=spec.name + "/coarse_model",
                channels=bd.channels, include_impulse=bd.include_impulse,
                method="analytic", joint_whitening=False, theta_names=theta_names,
            )
            rec["coarse_estimator_information"] = repc.to_dict()
        out[spec.name] = rec
    return out


# --------------------------------------------------------------------------
# Simulation + MAP recovery
# --------------------------------------------------------------------------


def _simulate(
    bd: BuiltDesign, u_true: np.ndarray, *, n_replicates: int, seed: int
) -> dict[str, Tensor]:
    """Simulate ``n_replicates`` independent records on native clocks."""
    mdl = make_model(u_true, bd.cfg, bd.proto, include_impulse=bd.include_impulse)
    ssm = mdl.ssm(("eeg", "bold"), epoch=0)
    E = bd.cfg.n_epochs
    from .filters import LinearGaussianSSM, replicate_chunks, simulate_epoched

    lm = structured_left_mul(mdl.F, bd.cfg)

    def make_ssm(c: int) -> LinearGaussianSSM:
        inp = mdl.inputs[0].unsqueeze(0).expand(c, E, bd.cfg.n_steps, mdl.n)
        return LinearGaussianSSM(
            mdl.F, mdl.Q, mdl.m0, mdl.P0, ssm.channels, bd.cfg.n_steps,
            inp.reshape(c * E, bd.cfg.n_steps, mdl.n), lm,
        )

    return simulate_epoched(
        make_ssm, n_replicates=n_replicates, n_epochs=E, seed=seed,
        chunks=replicate_chunks(n_replicates, E, bd.cfg.n_steps, mdl.n),
    )


def _prepare_fit_data(bd: BuiltDesign, data: dict[str, Tensor]) -> dict[str, Tensor]:
    out = {}
    if "eeg" in bd.channels:
        y = data["eeg"]
        if bd.spec.resample:
            y = decimate_eeg(y, bd.cfg, coarse_config(bd.cfg))
        out["eeg"] = y
    if "bold" in bd.channels:
        out["bold"] = data["bold"]
    return out


def _objective(
    bd: BuiltDesign, fit_data: dict[str, Tensor], *, n_replicates: int
) -> Callable[[Tensor], Tensor]:
    cfg = bd.fit_cfg
    proto = bd.fit_proto
    E = cfg.n_epochs
    u0 = torch.tensor(prior_mean_u(), dtype=getattr(torch, cfg.dtype))
    sd = torch.tensor(prior_sd_u(), dtype=getattr(torch, cfg.dtype))

    def neg_log_posterior(u: Tensor, checkpoint_every: int = 0) -> Tensor:
        mdl = make_model(u, cfg, proto, include_impulse=bd.include_impulse)
        ssm = mdl.ssm(bd.channels, epoch=0, eeg_steps=bd.fit_eeg_steps)
        ssm.inputs = mdl.inputs.expand(u.shape[0], E, cfg.n_steps, mdl.n)
        res = multiepoch_kalman_filter(
            ssm, fit_data, n_epochs=E, checkpoint_every=checkpoint_every
        )
        ll = res["log_likelihood"].sum(1)
        z = (u - u0.to(u)) / sd.to(u)
        return -(ll - 0.5 * (z**2).sum(-1))

    return neg_log_posterior


def _grad(f, u: Tensor, checkpoint_every: int) -> tuple[Tensor, Tensor]:
    u = u.detach().requires_grad_(True)
    val = f(u, checkpoint_every)
    (g,) = torch.autograd.grad(val.sum(), u)
    return val.detach(), g.detach()


@dataclass
class RecoveryResult:
    design: str
    regime: str
    n_replicates: int
    parameter_names: list[str]
    eta_true: np.ndarray
    estimates: np.ndarray                 # [R, P] unconstrained
    posterior_sd: np.ndarray              # [R, P]
    coverage: dict[str, CoverageResult]
    bias: dict[str, float]
    rmse: dict[str, float]
    delay_error_seconds: dict[str, float]
    posterior_correlation: np.ndarray
    heldout_log_loss: dict[str, float]
    converged_fraction: float
    optimiser: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return as_builtin(
            {
                "design": self.design,
                "regime": self.regime,
                "n_replicates": self.n_replicates,
                "parameter_names": self.parameter_names,
                "eta_true_unconstrained": self.eta_true,
                "estimate_mean": self.estimates.mean(0),
                "estimate_sd": self.estimates.std(0, ddof=1),
                "posterior_sd_mean": self.posterior_sd.mean(0),
                "coverage": {k: v.to_dict() for k, v in self.coverage.items()},
                "bias_in_prior_sd": self.bias,
                "rmse_in_prior_sd": self.rmse,
                "delay_error": self.delay_error_seconds,
                "posterior_correlation": self.posterior_correlation,
                "heldout_log_loss": self.heldout_log_loss,
                "converged_fraction": self.converged_fraction,
                "optimiser": self.optimiser,
            }
        )


def recover(
    bd: BuiltDesign,
    regime: Regime,
    *,
    n_replicates: int = 128,
    seed: int = 0,
    n_newton: int = 6,
    newton_tol: float = 0.05,
    checkpoint_every: int = 250,
    level: float = 0.95,
    heldout_data: dict[str, Tensor] | None = None,
    verbose: bool = False,
) -> RecoveryResult:
    """MAP recovery with Laplace (observed-information) intervals.

    The estimator is deliberately *blind*: every replicate starts from the prior
    mean and never sees ``eta_true``.  Newton steps are preconditioned by the
    expected information at the start point (shared, data-independent) with
    backtracking on the exact objective; the final interval uses the **observed**
    information from forward differences of the autodiff gradient at the
    estimate.
    """
    u_true = regime.eta_true()
    data = _simulate(bd, u_true, n_replicates=n_replicates, seed=seed)
    fit_data = _prepare_fit_data(bd, data)
    f = _objective(bd, fit_data, n_replicates=n_replicates)
    dt = getattr(torch, bd.fit_cfg.dtype)
    u = torch.tensor(
        np.tile(prior_mean_u(), (n_replicates, 1)), dtype=dt
    ).to(make_model(prior_mean_u(), bd.fit_cfg, bd.fit_proto).F.device)
    sd = torch.tensor(prior_sd_u(), dtype=u.dtype, device=u.device)

    # preconditioner: expected information at the start point (shared)
    rep0 = expected_fisher(
        prior_mean_u(), bd.fit_cfg, bd.fit_proto, channels=bd.channels,
        include_impulse=bd.include_impulse, eeg_steps=bd.fit_eeg_steps,
        method="analytic",
    )
    Hpre = torch.tensor(rep0.I_total, dtype=u.dtype, device=u.device)
    Hpre = Hpre / sd.unsqueeze(0) / sd.unsqueeze(1)   # back to unconstrained basis
    Hpre = Hpre + 1e-9 * torch.eye(N_PARAM, dtype=u.dtype, device=u.device)

    t0 = time.time()
    def _decrement(gr: Tensor) -> Tensor:
        """sqrt(g^T H^-1 g): distance to the optimum in posterior sds."""
        s = torch.linalg.solve(Hpre, gr.unsqueeze(-1)).squeeze(-1)
        return torch.sqrt(torch.clamp((gr * s).sum(-1), min=0.0))

    val, g = _grad(f, u, checkpoint_every)
    history = [float(val.mean())]
    decrements = [float(_decrement(g).max())]
    n_used = 0
    # The preconditioner is the expected information at the *prior mean* and is
    # never refreshed, so this iteration converges linearly, not quadratically.
    # A fixed step count therefore cannot certify convergence -- stop on the
    # Newton decrement instead, and record how many steps it actually took.
    for _it in range(n_newton):
        step = torch.linalg.solve(Hpre, g.unsqueeze(-1)).squeeze(-1)
        alpha = torch.ones(n_replicates, 1, dtype=u.dtype, device=u.device)
        for _bt in range(6):
            cand = u - alpha * step
            with torch.no_grad():
                vc = f(cand)
            worse = (vc > val) | ~torch.isfinite(vc)
            if not bool(worse.any()):
                break
            alpha = torch.where(worse.unsqueeze(-1), alpha * 0.4, alpha)
        u = (u - alpha * step).detach()
        val, g = _grad(f, u, checkpoint_every)
        history.append(float(val.mean()))
        n_used = _it + 1
        dec = _decrement(g)
        decrements.append(float(dec.max()))
        if verbose:
            print(f"  newton {_it}: obj {history[-1]:.3f} "
                  f"max decrement {decrements[-1]:.4g}")
        if decrements[-1] < newton_tol:
            break
    grad_norm = _decrement(g)

    # observed information by forward differences of the analytic gradient
    H = torch.zeros(n_replicates, N_PARAM, N_PARAM, dtype=u.dtype, device=u.device)
    for i in range(N_PARAM):
        h = 1e-4 * float(sd[i])
        du = torch.zeros_like(u)
        du[:, i] = h
        _, g2 = _grad(f, u + du, checkpoint_every)
        H[:, :, i] = (g2 - g) / h
    H = 0.5 * (H + H.transpose(-1, -2))
    ev = torch.linalg.eigvalsh(H)
    ok = ev[:, 0] > 0
    Hs = torch.where(
        ok.view(-1, 1, 1), H,
        Hpre.unsqueeze(0).expand_as(H),
    )
    cov = torch.linalg.inv(Hs)
    post_sd = torch.sqrt(torch.clamp(torch.diagonal(cov, dim1=-2, dim2=-1), min=0))

    uh = u.double().cpu().numpy()
    ps = post_sd.double().cpu().numpy()
    sdn = prior_sd_u()
    from scipy.stats import norm

    z = float(norm.ppf(0.5 * (1 + level)))
    cov_res, bias, rmse = {}, {}, {}
    for i, nm in enumerate(PARAM_NAMES):
        lo, hi = uh[:, i] - z * ps[:, i], uh[:, i] + z * ps[:, i]
        n_cov = int(((lo <= u_true[i]) & (u_true[i] <= hi)).sum())
        cov_res[nm] = CoverageResult(nm, level, n_replicates, n_cov)
        bias[nm] = float((uh[:, i] - u_true[i]).mean() / sdn[i])
        rmse[nm] = float(np.sqrt(((uh[:, i] - u_true[i]) ** 2).mean()) / sdn[i])
    tau_hat = np.exp(uh[:, PARAM_INDEX["tau"]])
    tau_true = float(np.exp(u_true[PARAM_INDEX["tau"]]))
    delay = {
        "true_seconds": tau_true,
        "mean_estimate_seconds": float(tau_hat.mean()),
        "bias_seconds": float(tau_hat.mean() - tau_true),
        "rmse_seconds": float(np.sqrt(((tau_hat - tau_true) ** 2).mean())),
        "rmse_seconds_se": _delay_rmse_se(tau_hat, tau_true, n_replicates),
        "mad_seconds": float(np.abs(tau_hat - tau_true).mean()),
    }
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(uh, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)

    hl: dict[str, float] = {}
    if heldout_data is not None:
        fh = _objective(bd, _prepare_fit_data(bd, heldout_data), n_replicates=n_replicates)
        with torch.no_grad():
            nlp = fh(u)
        n_obs = sum(
            v.shape[1] * v.shape[2] * v.shape[3]
            for v in _prepare_fit_data(bd, heldout_data).values()
        )
        hl = {
            "total_negative_log_posterior": float(nlp.mean()),
            "per_observation": float(nlp.mean()) / n_obs,
            "se": float(nlp.std(unbiased=True) / math.sqrt(n_replicates)),
            "n_observations": int(n_obs),
        }
    return RecoveryResult(
        design=bd.spec.name,
        regime=regime.name,
        n_replicates=n_replicates,
        parameter_names=list(PARAM_NAMES),
        eta_true=u_true,
        estimates=uh,
        posterior_sd=ps,
        coverage=cov_res,
        bias=bias,
        rmse=rmse,
        delay_error_seconds=delay,
        posterior_correlation=corr,
        heldout_log_loss=hl,
        converged_fraction=float((grad_norm < newton_tol).double().mean()),
        optimiser={
            "n_newton_max": n_newton,
            "n_newton_used": n_used,
            "newton_tol": newton_tol,
            "stopped_early": bool(n_used < n_newton),
            "objective_history_mean": history,
            "max_decrement_history": decrements,
            "convergence_metric": "newton_decrement_in_posterior_sd",
            "max_newton_decrement": float(grad_norm.max()),
            "median_newton_decrement": float(grad_norm.median()),
            "positive_definite_hessian_fraction": float(ok.double().mean()),
            "seconds": time.time() - t0,
        },
    )


def profile_likelihood(
    bd: BuiltDesign,
    regime: Regime,
    *,
    names: Sequence[str] = THETA_NAMES,
    n_grid: int = 7,
    span: float = 2.0,
    seed: int = 12345,
    n_newton: int = 5,
    checkpoint_every: int = 250,
) -> dict[str, Any]:
    """Profile log-likelihood over a grid for each preregistered parameter.

    All parameters except the profiled one are re-optimised at every grid point
    (batched: one filter run covers the whole grid for all parameters).
    """
    u_true = regime.eta_true()
    data = _simulate(bd, u_true, n_replicates=1, seed=seed)
    fit_data = {k: v.expand(len(names) * n_grid, *v.shape[1:])
                for k, v in _prepare_fit_data(bd, data).items()}
    f = _objective(bd, fit_data, n_replicates=len(names) * n_grid)
    sdn = prior_sd_u()
    dtt = getattr(torch, bd.fit_cfg.dtype)
    dev = make_model(u_true, bd.fit_cfg, bd.fit_proto).F.device
    grids = {}
    rows = []
    fixed_idx = []
    for nm in names:
        i = PARAM_INDEX[nm]
        gs = u_true[i] + np.linspace(-span, span, n_grid) * sdn[i]
        grids[nm] = gs
        for v in gs:
            r = u_true.copy()
            r[i] = v
            rows.append(r)
            fixed_idx.append(i)
    u = torch.tensor(np.stack(rows), dtype=dtt, device=dev)
    mask = torch.ones_like(u)
    mask[torch.arange(u.shape[0]), torch.tensor(fixed_idx, device=dev)] = 0.0
    sd = torch.tensor(sdn, dtype=dtt, device=dev)
    rep0 = expected_fisher(
        u_true, bd.fit_cfg, bd.fit_proto, channels=bd.channels,
        include_impulse=bd.include_impulse, method="analytic",
    )
    Hpre = torch.tensor(rep0.I_total, dtype=dtt, device=dev)
    Hpre = Hpre / sd.unsqueeze(0) / sd.unsqueeze(1)
    Hpre = Hpre + 1e-9 * torch.eye(N_PARAM, dtype=dtt, device=dev)
    val, g = _grad(f, u, checkpoint_every)
    for _ in range(n_newton):
        step = torch.linalg.solve(Hpre, (g * mask).unsqueeze(-1)).squeeze(-1) * mask
        alpha = torch.ones(u.shape[0], 1, dtype=dtt, device=dev)
        for _bt in range(6):
            cand = u - alpha * step
            with torch.no_grad():
                vc = f(cand)
            worse = (vc > val) | ~torch.isfinite(vc)
            if not bool(worse.any()):
                break
            alpha = torch.where(worse.unsqueeze(-1), alpha * 0.4, alpha)
        u = (u - alpha * step).detach()
        val, g = _grad(f, u, checkpoint_every)
    v = val.double().cpu().numpy().reshape(len(names), n_grid)
    out = {}
    for k, nm in enumerate(names):
        prof = -(v[k] - v[k].min())          # profile log posterior, peak at 0
        spec = PARAMS[PARAM_INDEX[nm]]
        out[nm] = {
            "grid_unconstrained": grids[nm].tolist(),
            "grid_natural": [
                float(x) for x in np.asarray(
                    natural_from_unconstrained(
                        torch.tensor(
                            np.stack([
                                np.where(
                                    np.arange(N_PARAM) == PARAM_INDEX[nm], gv, u_true
                                )
                                for gv in grids[nm]
                            ])
                        )
                    )[nm]
                ).reshape(-1)
            ],
            "profile_log_posterior": prof.tolist(),
            "curvature_at_truth": float(
                -np.gradient(np.gradient(prof, grids[nm]), grids[nm])[n_grid // 2]
            ),
            "units": spec.units,
        }
    return out


# --------------------------------------------------------------------------
# Full benchmark
# --------------------------------------------------------------------------


def run_benchmark(
    base_cfg: SystemConfig,
    *,
    regimes: Sequence[Regime] = REGIMES,
    designs: Sequence[DesignSpec] = DESIGNS,
    seed: int = 20260805,
    n_replicates: int = 128,
    n_newton: int = 6,
    newton_tol: float = 0.05,
    with_recovery: bool = True,
    with_profiles: bool = True,
    with_monte_carlo_fisher: bool = True,
    mc_replicates: int = 256,
    recovery_designs: Sequence[str] | None = None,
    heavy_regimes: Sequence[str] | None = None,
    checkpoint_path: str | None = None,
    resume: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the benchmark.

    ``recovery_designs`` restricts the (expensive) MAP-recovery arm to a subset
    of designs; the (cheap) T4 arm always covers every design.
    ``heavy_regimes`` restricts profile likelihoods and the Monte-Carlo complete
    information to named regimes.  Both restrictions are recorded in the
    manifest so the report states exactly what was computed where.

    ``resume`` reloads ``checkpoint_path`` and skips any (regime, design) pair
    already present.  On a shared machine this run *will* be interrupted; every
    interruption should cost the design in flight, not the whole sweep.
    """
    seed_everything(seed)
    sig = run_signature(
        base_cfg, seed=seed, n_replicates=n_replicates, n_newton=n_newton,
        mc_replicates=mc_replicates, with_recovery=with_recovery,
        newton_tol=newton_tol, with_profiles=with_profiles,
        with_monte_carlo_fisher=with_monte_carlo_fisher,
    )
    results: dict[str, Any] = {"regimes": {}, "run_signature": sig}
    done = _load_checkpoint(checkpoint_path, sig) if resume else {}
    if done and verbose:
        n = sum(len(r.get("designs", {})) for r in done.values())
        print(f"[resume] reusing {n} completed design(s) from {checkpoint_path}",
              flush=True)
    for regime in regimes:
        u_true = regime.eta_true()
        rres: dict[str, Any] = {
            "description": regime.description,
            "eta_true_unconstrained": u_true.tolist(),
            "eta_true_natural": {
                k: float(np.asarray(v).reshape(-1)[0])
                for k, v in natural_from_unconstrained(
                    torch.tensor(u_true).unsqueeze(0)
                ).items()
            },
            "designs": {},
        }
        if verbose:
            print(f"[regime {regime.name}]", flush=True)
        prior = done.get(regime.name, {}).get("designs", {})
        for spec in designs:
            if spec.name in prior:
                rres["designs"][spec.name] = prior[spec.name]
                if verbose:
                    print(f"  {spec.name:32s} [resumed from checkpoint]", flush=True)
                results["regimes"][regime.name] = rres
                continue
            t0 = time.time()
            bd = build_design(spec, base_cfg, regime, seed=seed)
            do_rec = with_recovery and (
                recovery_designs is None or spec.name in set(recovery_designs)
            )
            do_heavy = heavy_regimes is None or regime.name in set(heavy_regimes)
            entry: dict[str, Any] = {
                "label": spec.label, "primary": spec.primary, "notes": list(bd.notes),
                "sigma_eeg": bd.cfg.sigma_eeg, "sigma_bold": bd.cfg.sigma_bold,
                "n_eeg_samples_used": int(
                    (len(bd.eeg_steps) if bd.eeg_steps is not None
                     else len(bd.cfg.eeg_steps())) * bd.cfg.n_epochs
                ) if "eeg" in bd.channels else 0,
                "n_bold_samples_used": int(
                    len(bd.cfg.bold_steps()) * bd.cfg.n_epochs
                ) if "bold" in bd.channels else 0,
                "input_energy": bd.proto.energy,
            }
            rep = expected_fisher(
                u_true, bd.cfg, bd.proto, design=spec.name, channels=bd.channels,
                include_impulse=bd.include_impulse, eeg_steps=bd.eeg_steps,
                method="analytic", joint_whitening=False,
            )
            entry["fisher_T4"] = rep.to_dict()
            if len(bd.channels) > 1:
                entry["fisher_exact_joint"] = expected_fisher(
                    u_true, bd.cfg, bd.proto, design=spec.name, channels=bd.channels,
                    include_impulse=bd.include_impulse, eeg_steps=bd.eeg_steps,
                    method="analytic", joint_whitening=True,
                ).to_dict()
            if spec.coarse_model:
                entry["fisher_coarse_estimator"] = expected_fisher(
                    u_true, bd.fit_cfg, bd.fit_proto,
                    design=spec.name + "/coarse", channels=bd.channels,
                    include_impulse=bd.include_impulse, method="analytic",
                ).to_dict()
            if with_monte_carlo_fisher and do_heavy:
                mc = monte_carlo_fisher(
                    u_true, bd.cfg, bd.proto, channels=bd.channels,
                    include_impulse=bd.include_impulse, eeg_steps=bd.eeg_steps,
                    n_replicates=mc_replicates, seed=seed + 991,
                )
                entry["fisher_monte_carlo_complete"] = as_builtin(mc)
            if do_rec:
                heldout = _simulate(
                    bd, u_true, n_replicates=n_replicates, seed=seed + 77777
                )
                rec = recover(
                    bd, regime, n_replicates=n_replicates, seed=seed + 1,
                    n_newton=n_newton, newton_tol=newton_tol,
                    heldout_data=heldout, verbose=False,
                )
                entry["recovery"] = rec.to_dict()
            if with_profiles and do_heavy:
                entry["profile_likelihood"] = profile_likelihood(bd, regime, seed=seed + 3)
            entry["seconds"] = time.time() - t0
            if verbose:
                m = rep.metrics
                print(
                    f"  {spec.name:32s} rank={m['rank_likelihood']} "
                    f"cond={m['condition_number_total']:.3g} "
                    f"lmin_nonprior={m['min_eigenvalue_nonprior']:.4g} "
                    f"({entry['seconds']:.0f}s)"
                )
            rres["designs"][spec.name] = entry
            # checkpoint after EVERY design, not every regime: an interruption
            # must cost one design (minutes), never a whole regime (an hour).
            results["regimes"][regime.name] = rres
            _checkpoint(checkpoint_path, results)
        results["regimes"][regime.name] = rres
        _checkpoint(checkpoint_path, results)
    return results


def run_signature(
    cfg: SystemConfig, *, seed: int, n_replicates: int, n_newton: int,
    mc_replicates: int, with_recovery: bool, with_profiles: bool,
    with_monte_carlo_fisher: bool, newton_tol: float = 0.05,
) -> str:
    """Hash of everything that changes what a design entry *means*.

    Resuming across a settings change would silently splice together results
    computed under different budgets, which is exactly the kind of quiet
    inconsistency a claim report must never contain.

    Switching an *optional* arm on or off deliberately does **not** change the
    signature: profiles and the Monte-Carlo Fisher are additive diagnostics
    stored under their own keys, and their presence or absence cannot alter any
    other number in the entry.  Their sizes enter only while they are enabled.
    """
    import hashlib
    import json

    payload = {
        "epoch_seconds": cfg.epoch_seconds, "n_epochs": cfg.n_epochs,
        "dtype": cfg.dtype, "dt": cfg.dt, "n_regions": cfg.n_regions,
        "n_delay_taps": cfg.n_delay_taps, "hrf_stages": cfg.hrf_stages,
        "seed": seed, "with_recovery": with_recovery,
    }
    if with_recovery:
        payload |= {"n_replicates": n_replicates, "n_newton": n_newton,
                    "newton_tol": newton_tol}
    if with_monte_carlo_fisher:
        payload["mc_replicates"] = mc_replicates
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:16]


def _delay_rmse_se(tau_hat: np.ndarray, tau_true: float, n: int) -> float:
    """Delta-method standard error of the delay RMSE.

    A design carrying no delay information leaves every replicate at the prior
    mean, so the squared error can be identically zero and the delta-method
    denominator vanishes.  Return 0.0 there rather than a NaN, which JSON
    serialises to ``null`` and every downstream consumer then trips over.
    """
    sq = (np.asarray(tau_hat, float) - float(tau_true)) ** 2
    root = float(np.sqrt(sq.mean()))
    if root <= 0.0 or n < 2:
        return 0.0
    return float(np.std(sq, ddof=1) / (2.0 * root * math.sqrt(n)))


def _load_checkpoint(path: str | None, signature: str) -> dict[str, Any]:
    """Completed ``{regime: {designs: {...}}}`` from a previous run, or ``{}``.

    A corrupt, unreadable, or differently-configured checkpoint is treated as
    absent rather than fatal: losing the cache costs time, but refusing to
    start costs the whole run.
    """
    if not path:
        return {}
    import json
    from pathlib import Path as _P

    p = _P(path)
    if not p.exists():
        return {}
    try:
        blob = json.loads(p.read_text())
    except Exception as exc:                       # noqa: BLE001 - see docstring
        print(f"[resume] ignoring unreadable checkpoint {p}: {exc}", flush=True)
        return {}
    got = blob.get("run_signature")
    if got != signature:
        print(
            f"[resume] ignoring checkpoint {p}: run signature {got} != "
            f"{signature} (settings changed since it was written)",
            flush=True,
        )
        return {}
    return blob.get("regimes", {})


def _checkpoint(path: str | None, results: Mapping[str, Any]) -> None:
    if not path:
        return
    import json
    from pathlib import Path as _P

    p = _P(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(as_builtin(results), indent=1))
    tmp.replace(p)          # atomic: a kill mid-write cannot corrupt the file
