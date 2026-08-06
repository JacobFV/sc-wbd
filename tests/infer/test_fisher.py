"""T4: analytic Fisher == autodiff Fisher, prior kept separate, and the
structural facts the benchmark's conclusions rest on."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scwbd.infer.fisher import (
    DESIGN_ALIASES,
    PARAM_NAMES,
    THETA_NAMES,
    expected_fisher,
    mean_jacobian,
    monte_carlo_fisher,
    prior_information,
    resolve_design,
    schur_information,
)
from scwbd.infer.linear_gaussian import (
    N_PARAM,
    PARAM_INDEX,
    build_operator_derivatives,
    build_operators,
    coarse_config,
    delay_weights,
    delay_weights_grad,
    make_model,
)

torch.set_default_dtype(torch.float64)


def test_analytic_operator_derivatives_match_autodiff(tiny_setup):
    """The hand-derived dF/deta and dH/deta must equal autodiff exactly."""
    cfg, _proto, u0 = tiny_setup
    u = torch.tensor(u0, dtype=torch.float64).reshape(1, -1)
    ders = build_operator_derivatives(u, cfg)
    for key, out_key in (("F", "dF"), ("H_eeg", "dH_eeg"), ("H_bold", "dH_bold")):
        for i in range(N_PARAM):
            e = torch.zeros_like(u)
            e[0, i] = 1.0
            _, tang = torch.func.jvp(
                lambda uu: build_operators(uu, cfg)[key], (u,), (e,)
            )
            got = ders[out_key][0, i]
            scale = max(float(tang[0].abs().max()), 1e-12)
            assert float((got - tang[0]).abs().max()) / scale < 1e-9, (key, i)


def test_delay_weight_gradient_matches_autodiff(tiny_setup):
    cfg, _p, _u = tiny_setup
    tau = torch.tensor([0.012, 0.0075, 0.0165], dtype=torch.float64)
    ana = delay_weights_grad(tau, cfg)
    num = torch.autograd.functional.jacobian(
        lambda t: delay_weights(t, cfg).sum(0), tau
    )
    # column-wise: d w[:, p] / d tau_i is diagonal in i
    j = torch.autograd.functional.jacobian(lambda t: delay_weights(t, cfg), tau)
    for i in range(tau.numel()):
        assert float((ana[i] - j[i, :, i]).abs().max()) < 1e-9


def test_delay_interpolation_is_accurate(tiny_setup):
    """The windowed-sinc fractional delay must reproduce a true shift on a
    signal band-limited well below Nyquist, otherwise the delay parameter is
    measuring the interpolator, not the conduction time."""
    cfg, _p, _u = tiny_setup
    D = cfg.n_delay_taps
    tau = 0.0074
    w = delay_weights(torch.tensor([tau]), cfg)[0].numpy()
    t = np.arange(0, 2.0, cfg.dt)
    for f in (5.0, 20.0, 40.0):
        x = np.sin(2 * np.pi * f * t)
        lags = np.arange(D + 1)
        idx = np.arange(D + 5, len(t))
        approx = sum(w[p] * x[idx - p] for p in lags)
        exact = np.sin(2 * np.pi * f * (t[idx] - tau))
        assert np.abs(approx - exact).max() < 2e-3, f


def test_analytic_and_autodiff_jacobians_agree(tiny_setup):
    cfg, proto, u0 = tiny_setup
    Ja = mean_jacobian(u0, cfg, proto, method="analytic")
    Jn = mean_jacobian(u0, cfg, proto, method="autodiff")
    Jf = mean_jacobian(u0, cfg, proto, method="finite_difference")
    for k in Ja:
        s = float(Ja[k].abs().max())
        assert float((Ja[k] - Jn[k]).abs().max()) / s < 1e-10, k
        assert float((Ja[k] - Jf[k]).abs().max()) / s < 1e-5, k


def test_analytic_fisher_equals_autodiff_fisher(tiny_setup):
    cfg, proto, u0 = tiny_setup
    a = expected_fisher(u0, cfg, proto, method="analytic")
    b = expected_fisher(u0, cfg, proto, method="autodiff")
    s = float(np.abs(a.I_likelihood).max())
    assert np.abs(a.I_likelihood - b.I_likelihood).max() / s < 1e-9


def test_prior_is_reported_separately_and_is_the_identity(tiny_setup):
    """thesis sec. 0.3: a full-rank posterior must not disguise a prior-dominated
    likelihood."""
    cfg, proto, u0 = tiny_setup
    rep = expected_fisher(u0, cfg, proto, design="fmri_only")
    assert np.allclose(rep.I_prior, np.eye(N_PARAM))
    assert np.allclose(rep.I_total, rep.I_likelihood + rep.I_prior)
    # the total is full rank purely because of the prior
    assert rep.metrics["rank_total"] == N_PARAM
    assert rep.metrics["rank_likelihood"] < N_PARAM
    # and that is visible in the reported prior variance fraction
    assert max(rep.metrics["prior_variance_fraction"].values()) > 0.9


def test_t4_is_additive_over_modalities(tiny_setup):
    """The block-diagonal form of T4 makes I_joint = I_EEG + I_BOLD an identity.

    This is asserted, not assumed, because the report must be able to say that
    "joint beats single" is algebra rather than evidence.
    """
    cfg, proto, u0 = tiny_setup
    e = expected_fisher(u0, cfg, proto, design="eeg_only")
    b = expected_fisher(u0, cfg, proto, design="fmri_only")
    j = expected_fisher(u0, cfg, proto, design="joint_native")
    assert np.abs(j.I_likelihood - (e.I_likelihood + b.I_likelihood)).max() < 1e-8


def test_joint_whitening_differs_from_block_diagonal(tiny_setup):
    """The exact joint information is NOT the sum of per-modality quadratic
    forms, because EEG and BOLD share process noise."""
    cfg, proto, u0 = tiny_setup
    a = expected_fisher(u0, cfg, proto, joint_whitening=False)
    b = expected_fisher(u0, cfg, proto, joint_whitening=True)
    rel = np.abs(a.I_likelihood - b.I_likelihood).max() / np.abs(a.I_likelihood).max()
    assert rel > 1e-6


def test_naive_resampling_is_structurally_rank_deficient_in_the_delay(tiny_setup):
    """With a 1 s base clock the conduction delay has identically zero
    sensitivity -- the failure that design (iv) exists to expose."""
    cfg, proto, u0 = tiny_setup
    from scwbd.infer.linear_gaussian import coarsen_protocol

    cc = coarse_config(cfg)
    cp = coarsen_protocol(proto, cfg, cc)
    J = mean_jacobian(u0, cc, cp, method="analytic")
    i = PARAM_INDEX["tau"]
    for k, v in J.items():
        assert float(v[i].abs().max()) == 0.0, k
    rep = expected_fisher(u0, cc, cp, design="joint_native")
    assert float(rep.I_likelihood[i, i]) == 0.0
    assert rep.metrics["theta_profile_min_eigenvalue_nonprior"] < 1e-12


def test_schur_complement_is_the_conditional_information(tiny_setup):
    cfg, proto, u0 = tiny_setup
    rep = expected_fisher(u0, cfg, proto)
    idx = [PARAM_NAMES.index(t) for t in THETA_NAMES]
    S = schur_information(rep.I_total, idx)
    inv = np.linalg.inv(rep.I_total)
    # the Schur complement is the inverse of the theta block of the inverse
    assert np.abs(S - np.linalg.inv(inv[np.ix_(idx, idx)])).max() < 1e-8


def test_monte_carlo_complete_information_dominates_t4(tiny_setup):
    """T4 keeps only the mean-Jacobian term; the complete expected information
    also contains the covariance-sensitivity term, so it must be larger."""
    cfg, proto, u0 = tiny_setup
    t4 = expected_fisher(u0, cfg, proto, design="eeg_only")
    mc = monte_carlo_fisher(u0, cfg, proto, channels=("eeg",), n_replicates=64, seed=5)
    for i in (PARAM_INDEX["a21"], PARAM_INDEX["gain_eeg"]):
        se = float(mc["standard_error"][i, i])
        assert mc["I_likelihood"][i, i] > t4.I_likelihood[i, i] - 3 * se


def test_design_names_resolve_for_the_gate(tiny_setup):
    cfg, proto, u0 = tiny_setup
    for name in ("joint_native", "joint_plus_impulse", "eeg", "fmri",
                 "joint_resampled", "prior"):
        chans, imp, steps = resolve_design(name, cfg)
        rep = expected_fisher(u0, cfg, proto, design=name)
        assert rep.I_likelihood.shape == (N_PARAM, N_PARAM)
        assert rep.parameter_names == list(PARAM_NAMES)
    with pytest.raises(KeyError):
        resolve_design("not_a_design", cfg)
    assert np.allclose(expected_fisher(u0, cfg, proto, design="prior").I_likelihood, 0)


def test_impulse_increases_theta_information(tiny_setup):
    """G4's statistic: the write must raise the profile information about theta,
    not merely about the observation nuisances."""
    cfg, proto, u0 = tiny_setup
    base = expected_fisher(u0, cfg, proto, design="joint_native")
    imp = expected_fisher(u0, cfg, proto, design="joint_plus_impulse")
    b = base.metrics["theta_profile_min_eigenvalue_nonprior"]
    i = imp.metrics["theta_profile_min_eigenvalue_nonprior"]
    assert i > b, "the calibrated impulse did not increase theta information"


def test_fisher_is_deterministic(tiny_setup):
    cfg, proto, u0 = tiny_setup
    a = expected_fisher(u0, cfg, proto)
    b = expected_fisher(u0, cfg, proto)
    assert np.array_equal(a.I_likelihood, b.I_likelihood)
