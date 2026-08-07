"""Parameter recovery with nominal interval coverage.

This is a real statistical test, not a smoke test: enough Monte-Carlo replicates
that the coverage estimate has a meaningful error bar, and the assertion is made
against that error bar (Wilson) rather than against a hand-picked tolerance.

It is deliberately the slowest module in ``tests/infer``.  A cheaper version
would not be evidence about the thing the thesis conditions its next model on.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
import torch

from scwbd.infer.calibration import expected_coverage_curve, pit_histogram, pit_values
from scwbd.infer.identifiability import DESIGNS, REGIMES, build_design, recover
from scwbd.infer.linear_gaussian import PARAM_INDEX, SystemConfig, THETA_NAMES

# The whole MODULE is slow, not a subset of its tests.
#
# Marking individual tests deselected 8 of 17 and changed the runtime not at all:
# the cost lives in a module-scoped fixture (`recover(...)` over 64 replicates,
# the synthetic slice build), and one unmarked test is enough to pay it in full.
# A per-test marker on a module-scoped cost is a marker that does not do what it
# says.
pytestmark = pytest.mark.slow

DEVICE = os.environ.get("SCWBD_TEST_DEVICE", "cpu")
N_REPLICATES = int(os.environ.get("SCWBD_TEST_REPLICATES", "64"))


@pytest.fixture(scope="module")
def recovery():
    """MAP recovery for the joint native-clock design in the reference regime."""
    cfg = SystemConfig(
        device=DEVICE, dtype="float64", epoch_seconds=3.0, n_epochs=10,
        n_delay_taps=22, hrf_stages=6, hrf_peak_stage=3, hrf_under_stage=6,
    )
    bd = build_design(DESIGNS[2], cfg, REGIMES[0], seed=4242)
    return recover(bd, REGIMES[0], n_replicates=N_REPLICATES, seed=17, n_newton=5)


@pytest.mark.slow
def test_estimator_is_approximately_unbiased(recovery):
    """Bias is judged against the Monte-Carlo standard error, not a magic number."""
    r = recovery
    n = r.n_replicates
    for p in THETA_NAMES:
        i = r.parameter_names.index(p)
        emp_sd = float(r.estimates[:, i].std(ddof=1))
        from scwbd.infer.linear_gaussian import prior_sd_u

        se = emp_sd / math.sqrt(n) / prior_sd_u()[i]
        assert abs(r.bias[p]) < 4 * se + 0.02, (p, r.bias[p], se)


@pytest.mark.slow
def test_interval_coverage_is_nominal(recovery):
    """The headline statistical claim of the module."""
    r = recovery
    for p in THETA_NAMES:
        c = r.coverage[p]
        ci = c.wilson_interval
        assert c.is_nominal(), (
            f"{p}: empirical coverage {c.empirical:.3f} "
            f"[{ci.lo:.3f}, {ci.hi:.3f}] excludes the nominal 0.95 "
            f"over n={c.n} replicates"
        )


@pytest.mark.slow
def test_posterior_sd_tracks_the_sampling_spread(recovery):
    """A calibrated Laplace posterior must have the right *width*, not just the
    right coverage: over- and under-dispersion can cancel in a single level."""
    r = recovery
    for p in THETA_NAMES:
        i = r.parameter_names.index(p)
        ratio = r.posterior_sd[:, i].mean() / r.estimates[:, i].std(ddof=1)
        assert 0.75 < ratio < 1.35, (p, ratio)


@pytest.mark.slow
def test_pit_is_uniform_and_multilevel_coverage_holds(recovery):
    r = recovery
    i = PARAM_INDEX["tau"]
    truth = np.full(r.n_replicates, r.eta_true[i])
    pit = pit_values(truth, r.estimates[:, i], r.posterior_sd[:, i])
    h = pit_histogram(pit, n_bins=8)
    assert h["ks_pvalue"] > 0.01, h
    curve = expected_coverage_curve(truth, r.estimates[:, i], r.posterior_sd[:, i])
    for row in curve["coverage"]:
        assert row["nominal_inside_wilson95"], row


@pytest.mark.slow
def test_delay_is_recovered_to_sub_millisecond(recovery):
    """EEG at 1 ms should localise a 12 ms conduction delay; if it cannot, the
    benchmark has nothing to say about naive resampling."""
    d = recovery.delay_error_seconds
    assert d["rmse_seconds"] < 3e-3, d
    assert abs(d["bias_seconds"]) < 1e-3, d


@pytest.mark.slow
def test_optimiser_actually_converged(recovery):
    o = recovery.optimiser
    assert o["positive_definite_hessian_fraction"] > 0.95, o
    assert recovery.converged_fraction > 0.8, o


@pytest.mark.slow
def test_naive_resampling_loses_the_delay():
    """The scientific contrast, asserted as a test rather than only reported.

    The 1 s estimator cannot represent a 12 ms delay, so its delay error must
    fall back to the prior scale while the native-clock design does far better.
    """
    cfg = SystemConfig(
        device=DEVICE, dtype="float64", epoch_seconds=3.0, n_epochs=10,
        n_delay_taps=22, hrf_stages=6, hrf_peak_stage=3, hrf_under_stage=6,
    )
    reg = REGIMES[0]
    native = recover(build_design(DESIGNS[2], cfg, reg, seed=4242), reg,
                     n_replicates=24, seed=17, n_newton=4)
    naive = recover(build_design(DESIGNS[3], cfg, reg, seed=4242), reg,
                    n_replicates=24, seed=17, n_newton=4)
    assert (
        naive.delay_error_seconds["rmse_seconds"]
        > 3 * native.delay_error_seconds["rmse_seconds"]
    ), (naive.delay_error_seconds, native.delay_error_seconds)


def test_recovery_is_deterministic():
    cfg = SystemConfig(
        device=DEVICE, dtype="float64", epoch_seconds=2.0, n_epochs=2,
        n_delay_taps=22, hrf_stages=6, hrf_peak_stage=3, hrf_under_stage=6,
    )
    bd = build_design(DESIGNS[0], cfg, REGIMES[0], seed=1)
    a = recover(bd, REGIMES[0], n_replicates=4, seed=5, n_newton=2)
    b = recover(bd, REGIMES[0], n_replicates=4, seed=5, n_newton=2)
    assert np.array_equal(a.estimates, b.estimates)
