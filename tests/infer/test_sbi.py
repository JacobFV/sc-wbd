"""Simulation-based inference: SBC ranks must be uniform for a correctly
specified model, and non-uniform when the posterior is wrong."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from scwbd.infer.sbi import (
    ConditionalMAF,
    expected_coverage,
    multirate_summary_statistics,
    simulation_based_calibration,
    train_npe,
)
from scwbd.infer.types import seed_everything

DEFAULT_DTYPE = torch.float32   # consumed by the conftest autouse fixture


def test_sbc_ranks_are_uniform_for_an_exact_posterior():
    """Conjugate Gaussian: prior N(0,1), one observation with sd 0.7.

    The exact posterior is known in closed form, so a non-uniform rank
    histogram here would be a defect in the SBC implementation itself.
    """
    rng = np.random.default_rng(0)
    n_sim, n_draws = 600, 199
    tau = 0.7
    theta = rng.normal(0, 1, size=(n_sim, 1))
    y = theta[:, 0] + rng.normal(0, tau, n_sim)
    post_var = 1 / (1 + 1 / tau**2)
    post_mean = post_var * y / tau**2
    draws = post_mean[:, None] + math.sqrt(post_var) * rng.normal(size=(n_sim, n_draws))
    out = simulation_based_calibration(
        lambda i: draws[i][:, None], theta, names=["theta"], n_posterior=n_draws,
        n_bins=20,
    )
    assert out["per_parameter"]["theta"]["uniform_at_005"], out


@pytest.mark.parametrize("factor,expect", [(0.4, False), (2.5, False)])
def test_sbc_detects_over_and_under_confidence(factor, expect):
    rng = np.random.default_rng(1)
    n_sim, n_draws = 600, 199
    tau = 0.7
    theta = rng.normal(0, 1, size=(n_sim, 1))
    y = theta[:, 0] + rng.normal(0, tau, n_sim)
    post_var = 1 / (1 + 1 / tau**2)
    post_mean = post_var * y / tau**2
    draws = post_mean[:, None] + factor * math.sqrt(post_var) * rng.normal(
        size=(n_sim, n_draws)
    )
    out = simulation_based_calibration(
        lambda i: draws[i][:, None], theta, names=["theta"], n_posterior=n_draws
    )
    assert out["per_parameter"]["theta"]["uniform_at_005"] is expect


def test_expected_coverage_of_an_exact_posterior_is_nominal():
    rng = np.random.default_rng(2)
    n_sim = 500
    theta = rng.normal(0, 1, size=(n_sim, 1))
    y = theta[:, 0] + rng.normal(0, 0.7, n_sim)
    pv = 1 / (1 + 1 / 0.7**2)
    pm = pv * y / 0.7**2
    draws = pm[:, None] + math.sqrt(pv) * rng.normal(size=(n_sim, 2000))
    out = expected_coverage(lambda i: draws[i][:, None], theta, names=["theta"])
    for row in out["per_parameter"]["theta"]:
        assert row["nominal_inside_wilson95"], row


def test_conditional_flow_learns_a_linear_gaussian_posterior():
    """NPE on a problem whose posterior is analytic: theta ~ N(0,1),
    x = theta + noise(0.5).  The learned mean and sd must match."""
    seed_everything(3)
    n = 12000
    th = torch.randn(n, 1)
    x = th + 0.5 * torch.randn(n, 1)
    res = train_npe(th, x, epochs=180, hidden=64, n_flows=4, seed=3, batch_size=512)
    pv = 1 / (1 + 1 / 0.25)
    for x0 in (-1.0, 0.0, 1.2):
        s = res.posterior_samples(torch.tensor([x0]), n=4000, seed=1)[:, 0]
        want_mean = pv * x0 / 0.25
        assert abs(s.mean() - want_mean) < 0.12, (x0, s.mean(), want_mean)
        assert abs(s.std() - math.sqrt(pv)) / math.sqrt(pv) < 0.25, (x0, s.std())


def test_npe_posterior_passes_sbc():
    """The amortized posterior agent I inherits must itself be calibrated."""
    seed_everything(4)
    n = 12000
    th = torch.randn(n, 1)
    x = th + 0.5 * torch.randn(n, 1)
    res = train_npe(th, x, epochs=180, hidden=64, n_flows=4, seed=4, batch_size=512)
    n_sim, n_draws = 300, 99
    th_t = torch.randn(n_sim, 1)
    x_t = th_t + 0.5 * torch.randn(n_sim, 1)
    cache = [res.posterior_samples(x_t[i], n=n_draws, seed=1000 + i) for i in range(n_sim)]
    out = simulation_based_calibration(
        lambda i: cache[i], th_t.numpy(), names=["theta"], n_posterior=n_draws, n_bins=10
    )
    # a trained flow is only approximately calibrated; require the rank mean to
    # be centred and the histogram not to be grossly U-shaped
    r = out["per_parameter"]["theta"]
    assert abs(r["mean_normalised_rank"] - 0.5) < 0.08, r["mean_normalised_rank"]
    assert r["chi2_pvalue"] > 1e-4, r["chi2_pvalue"]


def test_multirate_summaries_keep_millisecond_lags():
    """Summaries must be computed on native clocks: block-averaging EEG onto the
    BOLD grid destroys exactly the lags that carry a conduction delay."""
    g = torch.Generator().manual_seed(0)
    B, E, T, p = 3, 2, 2000, 4
    base = torch.randn(B, E, T, p, generator=g)
    shifted = torch.roll(base, shifts=12, dims=2)
    a = multirate_summary_statistics({"eeg": base})
    b = multirate_summary_statistics({"eeg": shifted})
    assert a.shape[0] == B and a.shape[1] > 10
    # a 12 ms shift of one record changes the lagged cross-covariances
    mixed = base.clone()
    mixed[..., 1] = shifted[..., 0]
    c = multirate_summary_statistics({"eeg": mixed})
    assert float((c - a).abs().max()) > 1e-3
    # BOLD contributes its own block on its own clock
    d = multirate_summary_statistics({"eeg": base,
                                      "bold": torch.randn(B, E, 8, 3, generator=g)})
    assert d.shape[1] > a.shape[1]


def test_flow_log_prob_integrates_to_one():
    """A normalising flow that is not normalised is not a posterior."""
    seed_everything(5)
    flow = ConditionalMAF(dim=1, ctx=2, n_flows=3, hidden=32)
    ctx = torch.zeros(1, 2)
    grid = torch.linspace(-8, 8, 4001).unsqueeze(-1)
    with torch.no_grad():
        lp = flow.log_prob(grid, ctx.expand(grid.shape[0], -1))
    mass = float(torch.trapz(torch.exp(lp), grid[:, 0]))
    assert abs(mass - 1.0) < 1e-2, mass
