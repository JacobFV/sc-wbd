"""Expected Fisher information, **exactly T4** of ``thesis_contract.tex`` sec. 0.3.

    I_d(eta) = sum_{m in d} J_m(eta)^T R_m^{-1} J_m(eta) + I_prior,
    J_m = d mu_m / d eta,        eta = (theta, ell, rho)

Three independent code paths compute this and are cross-validated against one
another (``tests/infer/test_fisher.py``):

``analytic``
    Forward **sensitivity** propagation ``S_{k+1} = F S_k + (dF/deta) z_k``
    through the state-space recursion, using hand-derived closed forms for
    ``dF/deta`` and ``dH/deta`` (``linear_gaussian.build_operator_derivatives``).

``autodiff``
    Reverse-mode differentiation of the deterministic response ``mu_m(eta)``.

``monte_carlo``
    The *complete* expected information ``E[grad l grad l^T]`` of the exact
    multirate Kalman log-likelihood, estimated over simulated datasets.  Unlike
    T4 this includes the Slepian--Bangs covariance term
    ``1/2 tr(Sigma^-1 dSigma_i Sigma^-1 dSigma_j)``, so it is an *upper*
    reference: T4 is the mean-only, modality-block-diagonal part.  Reporting
    both keeps the approximation in T4 visible instead of hidden.

``R_m^{-1}`` is never formed.  ``R_m`` -- the marginal covariance of modality
``m``'s whole record, including process noise propagated through the delay
line and the hemodynamic cascade -- has ~10^5 rows here.  Instead the columns
of ``J_m`` are pushed through a Kalman filter for the *zero-input* model, whose
innovations transform is exactly the Cholesky whitening of ``R_m``.  Then
``J^T R^{-1} J = (chol(R)^{-1} J)^T (chol(R)^{-1} J)`` is an accumulation over
native-clock samples.

**The prior contribution is always reported separately** (thesis: *"Prior
contribution is shown separately so a full-rank posterior cannot disguise a
prior-dominated likelihood"*).  All matrices are in the **prior-standardised**
basis, in which ``I_prior`` is exactly the identity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import Tensor

from . import linear_gaussian as lg
from .filters import LinearGaussianSSM, multiepoch_kalman_filter
from .linear_gaussian import (
    N_PARAM,
    assert_delay_line_adequate,
    PARAM_NAMES,
    THETA_NAMES,
    Protocol,
    SystemConfig,
    SystemModel,
    build_operator_derivatives,
    make_model,
    prior_sd_u,
    structured_left_mul,
)
from .types import as_builtin

__all__ = [
    "DESIGN_ALIASES",
    "PARAM_NAMES",
    "THETA_NAMES",
    "FisherReport",
    "expected_fisher",
    "fisher_metrics",
    "mean_jacobian",
    "monte_carlo_fisher",
    "prior_information",
    "resolve_design",
    "schur_information",
]

_EIG_RTOL = 1e-10

#: ``design name -> (channels, include_impulse, decimate_eeg)``.
#:
#: ``scwbd.bench.gates.run_g4`` binds ``expected_fisher`` as a
#: ``design -> information`` map (``adapters.fisher_design_map``), so the design
#: name alone must be sufficient to select the read channels and the write.
#: ``PARAM_NAMES`` ordering is stable across every design, which is the
#: precondition for the gate's Schur complement over ``THETA_NAMES``.
DESIGN_ALIASES: dict[str, tuple[tuple[str, ...], bool, bool]] = {
    "eeg": (("eeg",), False, False),
    "eeg_only": (("eeg",), False, False),
    "fmri": (("bold",), False, False),
    "fmri_only": (("bold",), False, False),
    "bold": (("bold",), False, False),
    "joint": (("eeg", "bold"), False, False),
    "joint_native": (("eeg", "bold"), False, False),
    "joint_plus_impulse": (("eeg", "bold"), True, False),
    "joint_native_impulse": (("eeg", "bold"), True, False),
    "joint_native_impulse_matched": (("eeg", "bold"), True, False),
    "joint_resampled": (("eeg", "bold"), False, True),
    "joint_resampled_exactmodel": (("eeg", "bold"), False, True),
    "prior": ((), False, False),
}


def resolve_design(
    design: str, cfg: SystemConfig
) -> tuple[tuple[str, ...], bool, Tensor | None]:
    """``design name -> (channels, include_impulse, eeg_steps)``.

    Unknown names raise rather than silently defaulting to the joint design: a
    gate that silently evaluates the wrong design is worse than one that fails.
    """
    key = design.split("/")[0]
    if key not in DESIGN_ALIASES:
        raise KeyError(
            f"unknown design {design!r}; known designs: {sorted(DESIGN_ALIASES)}"
        )
    chans, impulse, decimate = DESIGN_ALIASES[key]
    steps = None
    if decimate:
        from .linear_gaussian import coarse_config

        cc = coarse_config(cfg)
        want = cc.bold_steps() * int(round(cc.dt / cfg.dt))
        steps = torch.tensor(np.clip(want, 0, cfg.n_steps - 1), dtype=torch.long)
    return chans, impulse, steps


def prior_information(standardised: bool = True) -> np.ndarray:
    """``I_prior``.  Independent Gaussian prior on the unconstrained coordinate."""
    sd = prior_sd_u()
    if standardised:
        return np.eye(N_PARAM)
    return np.diag(1.0 / sd**2)


# --------------------------------------------------------------------------
# Mean Jacobians  J_m = d mu_m / d eta
# --------------------------------------------------------------------------


def mean_jacobian(
    u: np.ndarray,
    cfg: SystemConfig,
    proto: Protocol,
    *,
    channels: Sequence[str] = ("eeg", "bold"),
    include_impulse: bool = False,
    method: str = "analytic",
    eeg_steps: Tensor | None = None,
    device: str | None = None,
) -> dict[str, Tensor]:
    """``J_m`` for every channel, shape ``[n_param, n_epochs, n_obs, p]``."""
    if method == "analytic":
        return _jacobian_forward_sensitivity(
            u, cfg, proto, channels, include_impulse, eeg_steps, device
        )
    if method == "autodiff":
        return _jacobian_autodiff(
            u, cfg, proto, channels, include_impulse, eeg_steps, device
        )
    if method == "finite_difference":
        return _jacobian_finite_difference(
            u, cfg, proto, channels, include_impulse, eeg_steps, device
        )
    raise ValueError(f"unknown Jacobian method {method!r}")


def _channels_of(mdl: SystemModel, channels, eeg_steps):
    out = []
    if "eeg" in channels:
        out.append(("eeg", mdl.H_eeg, mdl.eeg_steps if eeg_steps is None else eeg_steps))
    if "bold" in channels:
        out.append(("bold", mdl.H_bold, mdl.bold_steps))
    return out


def _jacobian_forward_sensitivity(
    u, cfg, proto, channels, include_impulse, eeg_steps, device
) -> dict[str, Tensor]:
    mdl = make_model(u, cfg, proto, include_impulse=include_impulse, device=device)
    devt, dt = mdl.F.device, mdl.F.dtype
    ders = build_operator_derivatives(
        torch.tensor(np.asarray(u, float), dtype=dt, device=devt).reshape(1, -1), cfg
    )
    dF = ders["dF"][0]                       # [P, n, n]
    dH = {"eeg": ders["dH_eeg"][0], "bold": ders["dH_bold"][0]}
    n = mdl.n
    E = cfg.n_epochs
    P = N_PARAM
    fmul = structured_left_mul(mdl.F, cfg)
    b = mdl.inputs[0]                        # [E, T, n]
    Z = torch.zeros(n, E, dtype=dt, device=devt)
    S = torch.zeros(P, n, E, dtype=dt, device=devt)
    chans = _channels_of(mdl, channels, eeg_steps)
    want = {nm: {int(k): j for j, k in enumerate(st.tolist())} for nm, _H, st in chans}
    out = {
        nm: torch.zeros(P, E, len(st), H.shape[-2], dtype=dt, device=devt)
        for nm, H, st in chans
    }
    for k in range(cfg.n_steps):
        for nm, H, _st in chans:
            j = want[nm].get(k)
            if j is None:
                continue
            # dJ = H S^i + (dH/deta_i) z
            t1 = torch.einsum("pn,ine->ipe", H[0], S)
            t2 = torch.einsum("ipn,ne->ipe", dH[nm], Z)
            out[nm][:, :, j, :] = (t1 + t2).permute(0, 2, 1)
        if k + 1 < cfg.n_steps:
            drive = torch.einsum("inm,me->ine", dF, Z)
            # one structured transition for the state and all P sensitivities:
            # columns are [E state columns | P*E sensitivity columns]
            cols = torch.cat(
                [Z, S.permute(1, 0, 2).reshape(n, P * E)], dim=1
            ).unsqueeze(0)
            newcols = fmul(cols)[0]
            Z = newcols[:, :E] + b[:, k, :].transpose(0, 1)
            S = newcols[:, E:].reshape(n, P, E).permute(1, 0, 2) + drive
    return out


def _jacobian_autodiff(
    u, cfg, proto, channels, include_impulse, eeg_steps, device
) -> dict[str, Tensor]:
    u0 = np.asarray(u, dtype=float)
    dt = getattr(torch, cfg.dtype)
    ut = torch.tensor(u0, dtype=dt, device=lg.resolve_device(device or cfg.device))
    ut.requires_grad_(True)

    def response(uu: Tensor) -> dict[str, Tensor]:
        mdl = make_model(uu, cfg, proto, include_impulse=include_impulse, device=device)
        chans = _channels_of(mdl, channels, eeg_steps)
        want = {nm: {int(k): j for j, k in enumerate(st.tolist())} for nm, _H, st in chans}
        n, E = mdl.n, cfg.n_epochs
        Z = torch.zeros(E, n, dtype=uu.dtype, device=uu.device)
        b = mdl.inputs[0]
        fmul = structured_left_mul(mdl.F, cfg)
        acc = {nm: [] for nm, _H, _s in chans}
        for k in range(cfg.n_steps):
            for nm, H, _st in chans:
                if k in want[nm]:
                    acc[nm].append(torch.einsum("pn,en->ep", H[0], Z))
            if k + 1 < cfg.n_steps:
                Z = fmul(Z.transpose(0, 1).unsqueeze(0))[0].transpose(0, 1) + b[:, k, :]
        return {nm: torch.stack(v, dim=1) for nm, v in acc.items() if v}

    # Forward-mode: N_PARAM directional derivatives.  The tangent propagation is
    # produced by autodiff from build_operators, so this path never touches the
    # hand-derived closed forms in build_operator_derivatives -- which is what
    # makes the analytic/autodiff agreement test meaningful.
    cols: dict[str, list[Tensor]] = {}
    base = ut.detach()
    for i in range(N_PARAM):
        e = torch.zeros_like(base)
        e[i] = 1.0
        _, tang = torch.func.jvp(response, (base,), (e,))
        for nm, t in tang.items():
            cols.setdefault(nm, []).append(t.detach())
    return {nm: torch.stack(v, dim=0) for nm, v in cols.items()}


def _jacobian_finite_difference(
    u, cfg, proto, channels, include_impulse, eeg_steps, device, rel_step=1e-4
) -> dict[str, Tensor]:
    """Central differences on ``mu_m(eta)`` -- a third, derivative-free check."""
    from .linear_gaussian import prior_sd_u as _sd

    sd = _sd()
    u0 = np.asarray(u, dtype=float)

    def resp(uu):
        mdl = make_model(uu, cfg, proto, include_impulse=include_impulse, device=device)
        from .filters import deterministic_response

        out = {}
        for nm in channels:
            ssm = mdl.ssm((nm,), epoch=None, eeg_steps=eeg_steps)
            ssm.inputs = mdl.inputs[0]
            out[nm] = deterministic_response(ssm)[nm]
        return out

    cols: dict[str, list[Tensor]] = {}
    for i in range(N_PARAM):
        h = rel_step * sd[i]
        up, dn = u0.copy(), u0.copy()
        up[i] += h
        dn[i] -= h
        ru, rd = resp(up), resp(dn)
        for nm in ru:
            cols.setdefault(nm, []).append((ru[nm] - rd[nm]) / (2 * h))
    return {nm: torch.stack(v, dim=0) for nm, v in cols.items()}


# --------------------------------------------------------------------------
# Whitening and assembly
# --------------------------------------------------------------------------


def _whiten(
    mdl: SystemModel,
    J: dict[str, Tensor],
    *,
    whiten_channels: Sequence[str],
    eeg_steps: Tensor | None,
) -> dict[str, Tensor]:
    """Push Jacobian columns through the zero-input filter of ``whiten_channels``.

    The resulting innovations are ``chol(R)^{-1} J`` where ``R`` is the marginal
    residual covariance of the *stacked* record over ``whiten_channels``.
    """
    ssm = mdl.ssm(whiten_channels, epoch=0, eeg_steps=eeg_steps, with_inputs=False)
    ssm.inputs = None
    P, E = next(iter(J.values())).shape[:2]
    data = {}
    for ch in ssm.channels:
        if ch.name in J:
            data[ch.name] = J[ch.name].reshape(1, P * E, J[ch.name].shape[2], -1)
        else:
            data[ch.name] = torch.zeros(
                1, P * E, ch.n_obs, ch.p, dtype=mdl.F.dtype, device=mdl.F.device
            )
    res = multiepoch_kalman_filter(ssm, data, n_epochs=P * E, whiten=True)
    return {
        ch.name: res["whitened/" + ch.name].reshape(P, E, ch.n_obs, ch.p)
        for ch in ssm.channels
    }


def _gram(w: Tensor) -> np.ndarray:
    """``sum over epochs and native-clock samples`` of ``w_i . w_j``."""
    flat = w.reshape(w.shape[0], -1)
    return (flat @ flat.transpose(0, 1)).double().cpu().numpy()


@dataclass
class FisherReport:
    design: str
    method: str
    parameter_names: list[str]
    basis: str
    I_likelihood: np.ndarray
    I_by_modality: dict[str, np.ndarray]
    I_prior: np.ndarray
    joint_whitening: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def I_total(self) -> np.ndarray:
        return self.I_likelihood + self.I_prior

    def to_dict(self) -> dict[str, Any]:
        return as_builtin(
            {
                "design": self.design,
                "method": self.method,
                "basis": self.basis,
                "parameter_names": self.parameter_names,
                "joint_whitening": self.joint_whitening,
                "I_likelihood": self.I_likelihood,
                "I_by_modality": self.I_by_modality,
                "I_prior": self.I_prior,
                "I_total": self.I_total,
                "metrics": self.metrics,
                "notes": self.notes,
            }
        )


def expected_fisher(
    u: np.ndarray,
    cfg: SystemConfig,
    proto: Protocol,
    *,
    design: str = "joint_native",
    channels: Sequence[str] | None = None,
    include_impulse: bool | None = None,
    eeg_steps: Tensor | None = None,
    method: str = "analytic",
    joint_whitening: bool = False,
    standardised: bool = True,
    device: str | None = None,
    theta_names: Sequence[str] = THETA_NAMES,
) -> FisherReport:
    """T4 for one design.

    ``joint_whitening=False`` reproduces T4 *literally*: a **sum over
    modalities** of separate quadratic forms, i.e. each modality is whitened by
    its own residual covariance and the EEG/BOLD cross-covariance induced by
    shared process noise is ignored.  ``joint_whitening=True`` whitens the
    stacked record instead and is the exact joint information.  The benchmark
    reports both, because the difference is precisely the amount by which T4
    over-counts joint information.
    """
    # Design-by-name resolution: this is what makes `expected_fisher` usable as
    # the `design -> information` map that scwbd.bench.gates.run_g4 binds.
    d_chans, d_imp, d_steps = resolve_design(design, cfg)
    if channels is None:
        channels = d_chans
    if include_impulse is None:
        include_impulse = d_imp
    if eeg_steps is None:
        eeg_steps = d_steps
    if eeg_steps is not None and not isinstance(eeg_steps, Tensor):
        eeg_steps = torch.as_tensor(eeg_steps, dtype=torch.long)
    sd = prior_sd_u()
    if not channels:  # the "prior" design: no likelihood contribution at all
        Z = np.zeros((N_PARAM, N_PARAM))
        rep = FisherReport(
            design=design, method=method, parameter_names=list(PARAM_NAMES),
            basis="prior_standardised" if standardised else "unconstrained",
            I_likelihood=Z, I_by_modality={}, I_prior=prior_information(standardised),
            joint_whitening=joint_whitening,
            notes=["prior-only design: I_likelihood is identically zero"],
        )
        rep.metrics = fisher_metrics(rep, theta_names=theta_names)
        return rep

    assert_delay_line_adequate(cfg, u)
    mdl = make_model(u, cfg, proto, include_impulse=include_impulse, device=device)
    if eeg_steps is not None:
        eeg_steps = eeg_steps.to(mdl.F.device)
    J = mean_jacobian(
        u, cfg, proto, channels=channels, include_impulse=include_impulse,
        method=method, eeg_steps=eeg_steps, device=device,
    )
    by_mod: dict[str, np.ndarray] = {}
    if joint_whitening:
        w = _whiten(mdl, J, whiten_channels=channels, eeg_steps=eeg_steps)
        I_like = sum(_gram(w[c]) for c in channels)
        by_mod["joint"] = I_like
    else:
        for c in channels:
            w = _whiten(mdl, {c: J[c]}, whiten_channels=(c,), eeg_steps=eeg_steps)
            by_mod[c] = _gram(w[c])
        I_like = sum(by_mod.values())

    if standardised:
        D = np.diag(sd)
        I_like = D @ I_like @ D
        by_mod = {k: D @ v @ D for k, v in by_mod.items()}
    I_pr = prior_information(standardised)
    rep = FisherReport(
        design=design,
        method=method,
        parameter_names=list(PARAM_NAMES),
        basis="prior_standardised" if standardised else "unconstrained",
        I_likelihood=np.asarray(I_like, float),
        I_by_modality={k: np.asarray(v, float) for k, v in by_mod.items()},
        I_prior=I_pr,
        joint_whitening=joint_whitening,
    )
    rep.metrics = fisher_metrics(rep, theta_names=theta_names)
    return rep


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def _sym_eig(M: np.ndarray) -> np.ndarray:
    return np.linalg.eigvalsh(0.5 * (M + M.T))


def _numerical_rank(M: np.ndarray, rtol: float = _EIG_RTOL) -> int:
    ev = _sym_eig(M)
    mx = float(ev.max()) if ev.size else 0.0
    if mx <= 0:
        return 0
    return int((ev > rtol * mx).sum())


def _cond(M: np.ndarray) -> float:
    ev = _sym_eig(M)
    lo, hi = float(ev.min()), float(ev.max())
    if lo <= 0 or hi <= 0:
        return float("inf")
    return hi / lo


def schur_information(I: np.ndarray, keep: Sequence[int]) -> np.ndarray:
    """Profile (Schur-complement) information about ``keep`` after the rest.

    ``I_kk - I_kn I_nn^{-1} I_nk``.  This -- not the naive submatrix -- is the
    information about the preregistered subset once observation nuisances have
    been profiled out, and it is the quantity the claim gate uses.
    """
    keep = list(keep)
    rest = [i for i in range(I.shape[0]) if i not in keep]
    Ikk = I[np.ix_(keep, keep)]
    if not rest:
        return Ikk
    Ikn = I[np.ix_(keep, rest)]
    Inn = I[np.ix_(rest, rest)]
    try:
        sol = np.linalg.solve(Inn, Ikn.T)
    except np.linalg.LinAlgError:
        sol = np.linalg.pinv(Inn) @ Ikn.T
    return Ikk - Ikn @ sol


#: Relative reproducibility of a *well-conditioned* eigenvalue of these Gram
#: matrices, measured directly by recomputing the whole pipeline under three
#: BLAS thread counts (1 / 8 / 20), which changes summation order:
#: eeg_only and joint_native theta-profile lambda_min reproduced to 1.32e-12
#: relative (~12 significant figures).  A near-cancelling eigenvalue inherits
#: this amplified by lambda_max/lambda_min: fmri_only reproduced to only
#: 9.27e-08 (~7 figures), which the ratio predicts.  Printing 15 digits of a
#: number reproducible to 7 is not a rounding preference, it is a claim about
#: the measurement that the measurement does not support.
_EIG_REL_REPRODUCIBILITY = 1.32e-12


def _eig_uncertainty(ev: np.ndarray) -> dict[str, Any]:
    """Estimated reproducibility of ``min(ev)`` and whether it is a real value."""
    if ev.size == 0:
        return {"relative_uncertainty": float("nan"), "significant_figures": 0,
                "numerically_zero": True}
    lo, hi = float(ev.min()), float(np.abs(ev).max())
    absu = _EIG_REL_REPRODUCIBILITY * hi
    rel = absu / abs(lo) if lo != 0 else float("inf")
    return {
        "absolute_uncertainty": absu,
        "relative_uncertainty": rel,
        "significant_figures": (
            0 if not np.isfinite(rel) or rel >= 1 else max(0, int(-math.log10(rel)))
        ),
        "numerically_zero": bool(abs(lo) <= absu),
    }


def _report_eig(ev: np.ndarray) -> float:
    """``min(ev)`` reported as exactly 0 when it is inside its own noise floor."""
    u = _eig_uncertainty(ev)
    return 0.0 if u["numerically_zero"] else float(ev.min())


def fisher_metrics(
    rep: FisherReport, *, theta_names: Sequence[str] = THETA_NAMES
) -> dict[str, Any]:
    names = rep.parameter_names
    idx = [names.index(t) for t in theta_names]
    I_like = rep.I_likelihood
    I_tot = rep.I_total
    ev_like = _sym_eig(I_like)
    ev_tot = _sym_eig(I_tot)
    cov = np.linalg.inv(I_tot)
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    corr = cov / np.outer(np.where(sd > 0, sd, np.inf), np.where(sd > 0, sd, np.inf))
    prior_frac = {
        n: float(1.0 / (1.0 + I_like[i, i])) for i, n in enumerate(names)
    }
    Sl = schur_information(I_like, idx)
    St = schur_information(I_tot, idx)
    off = np.abs(corr - np.eye(len(names)))
    i, j = np.unravel_index(np.argmax(off), off.shape)
    return {
        "rank_likelihood": _numerical_rank(I_like),
        "rank_total": _numerical_rank(I_tot),
        "n_parameters": len(names),
        "condition_number_likelihood": _cond(I_like),
        "condition_number_total": _cond(I_tot),
        "min_eigenvalue_nonprior": _report_eig(ev_like),
        "min_eigenvalue_nonprior_raw": float(ev_like.min()),
        "min_eigenvalue_nonprior_numerics": _eig_uncertainty(ev_like),
        "max_eigenvalue_nonprior": float(ev_like.max()),
        "eigenvalues_likelihood": ev_like.tolist(),
        "eigenvalues_total": ev_tot.tolist(),
        "log10_det_likelihood": float(
            np.sum(np.log10(np.clip(ev_like, 1e-300, None)))
        ),
        "log10_det_total": float(np.sum(np.log10(np.clip(ev_tot, 1e-300, None)))),
        "posterior_sd": {n: float(sd[k]) for k, n in enumerate(names)},
        "posterior_correlation": corr.tolist(),
        "max_abs_posterior_correlation": float(off[i, j]),
        "max_abs_posterior_correlation_pair": [names[i], names[j]],
        "prior_variance_fraction": prior_frac,
        "theta_subset": list(theta_names),
        "theta_profile_information_likelihood": Sl.tolist(),
        "theta_profile_min_eigenvalue_nonprior": _report_eig(_sym_eig(Sl)),
        "theta_profile_min_eigenvalue_nonprior_raw": float(_sym_eig(Sl).min()),
        "theta_profile_min_eigenvalue_numerics": _eig_uncertainty(_sym_eig(Sl)),
        "theta_profile_log10_det_likelihood": float(
            np.sum(np.log10(np.clip(np.abs(_sym_eig(Sl)), 1e-300, None)))
        ),
        "theta_profile_min_eigenvalue_total": float(_sym_eig(St).min()),
        "theta_profile_rank_likelihood": _numerical_rank(Sl),
        "theta_profile_condition_number_total": _cond(St),
    }


# --------------------------------------------------------------------------
# Monte-Carlo *complete* expected information (mean + covariance terms)
# --------------------------------------------------------------------------


def _loglik_per_replicate(
    u_row: np.ndarray,
    cfg: SystemConfig,
    proto: Protocol,
    data: dict[str, Tensor],
    *,
    channels: Sequence[str],
    include_impulse: bool,
    eeg_steps: Tensor | None,
    n_replicates: int,
    device: str | None,
) -> Tensor:
    mdl = make_model(u_row, cfg, proto, include_impulse=include_impulse, device=device)
    ssm = mdl.ssm(channels, epoch=0, eeg_steps=eeg_steps)
    E = cfg.n_epochs
    # tiled lazily by the filter: E distinct rows, not n_replicates*E copies
    ssm.inputs = mdl.inputs
    res = multiepoch_kalman_filter(
        ssm,
        {k: v.reshape(1, n_replicates * E, *v.shape[2:]) for k, v in data.items()},
        n_epochs=n_replicates * E,
    )
    return res["log_likelihood"].reshape(n_replicates, E).sum(1)


def monte_carlo_fisher(
    u: np.ndarray,
    cfg: SystemConfig,
    proto: Protocol,
    *,
    channels: Sequence[str] = ("eeg", "bold"),
    include_impulse: bool = False,
    eeg_steps: Tensor | None = None,
    n_replicates: int = 256,
    seed: int = 0,
    fd_step: float = 1e-3,
    standardised: bool = True,
    device: str | None = None,
) -> dict[str, Any]:
    """``I = E[grad l grad l^T]`` for the exact multirate log-likelihood.

    This is the *complete* Fisher information: unlike T4 it retains the
    covariance-sensitivity (Slepian--Bangs) term and the EEG/BOLD cross
    covariance.  Estimated by central finite differences of the exact filter
    log-likelihood over ``n_replicates`` simulated records; the Monte-Carlo
    standard error of each entry is returned alongside.
    """
    from .filters import simulate_lgssm

    mdl = make_model(u, cfg, proto, include_impulse=include_impulse, device=device)
    E = cfg.n_epochs
    ssm = mdl.ssm(channels, epoch=0, eeg_steps=eeg_steps)
    sim = LinearGaussianSSM(
        mdl.F, mdl.Q, mdl.m0, mdl.P0, ssm.channels, cfg.n_steps,
        mdl.inputs[0],          # [E, T, n]; tiled per step by the simulator
        structured_left_mul(mdl.F, cfg),
    )
    data, _ = simulate_lgssm(sim, seed=seed, batch=n_replicates * E)
    data = {k: v.reshape(n_replicates, E, *v.shape[1:]) for k, v in data.items()}

    sd = prior_sd_u()
    g = np.zeros((n_replicates, N_PARAM))
    for i in range(N_PARAM):
        h = fd_step * sd[i]
        up = np.asarray(u, float).copy(); up[i] += h
        dn = np.asarray(u, float).copy(); dn[i] -= h
        lp = _loglik_per_replicate(up, cfg, proto, data, channels=channels,
                                   include_impulse=include_impulse, eeg_steps=eeg_steps,
                                   n_replicates=n_replicates, device=device)
        lm = _loglik_per_replicate(dn, cfg, proto, data, channels=channels,
                                   include_impulse=include_impulse, eeg_steps=eeg_steps,
                                   n_replicates=n_replicates, device=device)
        g[:, i] = ((lp - lm) / (2 * h)).double().cpu().numpy()
    if standardised:
        g = g * sd[None, :]
    I = g.T @ g / n_replicates
    outer = np.einsum("ri,rj->rij", g, g)
    se = outer.std(axis=0, ddof=1) / math.sqrt(n_replicates)
    return {
        "I_likelihood": I,
        "standard_error": se,
        "n_replicates": n_replicates,
        "mean_score": g.mean(axis=0).tolist(),
        "mean_score_se": (g.std(axis=0, ddof=1) / math.sqrt(n_replicates)).tolist(),
        "basis": "prior_standardised" if standardised else "unconstrained",
    }
