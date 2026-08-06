"""Calibration diagnostics must be right on problems whose answer is known."""

from __future__ import annotations

import math

import numpy as np
import pytest

from scwbd.infer.calibration import (
    bootstrap_ci,
    crps_ensemble,
    crps_gaussian,
    expected_coverage_curve,
    interval_coverage,
    log_score_gaussian,
    paired_bootstrap_difference,
    pit_histogram,
    pit_values,
    reliability_diagram,
    sharpness,
    subgroup_calibration,
)
from scwbd.infer.types import CoverageResult


def test_wilson_interval_and_coverage_bookkeeping():
    cr = CoverageResult("a", 0.95, 200, 190)
    assert abs(cr.empirical - 0.95) < 1e-12
    ci = cr.wilson_interval
    assert ci.lo < 0.95 < ci.hi
    assert cr.is_nominal()
    bad = CoverageResult("a", 0.95, 400, 280)      # 70%
    assert not bad.is_nominal()


def test_well_calibrated_gaussian_has_nominal_coverage_and_uniform_pit():
    rng = np.random.default_rng(0)
    n = 4000
    mu = rng.normal(size=n)
    sd = np.full(n, 1.3)
    y = mu + sd * rng.normal(size=n)
    cr = interval_coverage(y, mu - 1.959963985 * sd, mu + 1.959963985 * sd)
    assert cr.is_nominal()
    h = pit_histogram(pit_values(y, mu, sd))
    assert h["uniform_at_005"], h


def test_overconfident_forecast_is_detected():
    rng = np.random.default_rng(1)
    n = 4000
    mu = np.zeros(n)
    y = rng.normal(0, 1.0, n)
    sd = np.full(n, 0.5)          # half the true spread
    cr = interval_coverage(y, mu - 1.96 * sd, mu + 1.96 * sd)
    assert cr.empirical < 0.85
    assert not cr.is_nominal()
    h = pit_histogram(pit_values(y, mu, sd))
    assert not h["uniform_at_005"]


def test_crps_gaussian_matches_ensemble_crps():
    rng = np.random.default_rng(2)
    n = 400
    mu = rng.normal(size=n)
    sd = np.full(n, 0.8)
    y = mu + sd * rng.normal(size=n)
    ens = mu[:, None] + sd[:, None] * rng.normal(size=(n, 4000))
    a = crps_gaussian(y, mu, sd)["crps"]
    b = crps_ensemble(y, ens)["crps"]
    assert abs(a - b) / a < 0.03


def test_log_score_prefers_the_true_variance():
    rng = np.random.default_rng(3)
    y = rng.normal(0, 1.0, 5000)
    mu = np.zeros_like(y)
    good = log_score_gaussian(y, mu, np.ones_like(y))["log_score"]
    for wrong in (0.5, 2.0):
        assert log_score_gaussian(y, mu, np.full_like(y, wrong))["log_score"] > good


def test_reliability_diagram_on_a_calibrated_classifier():
    rng = np.random.default_rng(4)
    p = rng.uniform(size=20000)
    y = (rng.uniform(size=p.size) < p).astype(float)
    rd = reliability_diagram(p, y, n_bins=10)
    assert rd.expected_calibration_error < 0.02
    biased = reliability_diagram(np.clip(p + 0.2, 0, 1), y, n_bins=10)
    assert biased.expected_calibration_error > 0.1


def test_sharpness_and_coverage_are_reported_together():
    s = sharpness(np.full(100, 2.0))
    assert abs(s["mean_interval_width"] - 2 * 1.959963985 * 2.0) < 1e-6


def test_expected_coverage_curve_is_monotone():
    rng = np.random.default_rng(5)
    n = 3000
    mu = np.zeros(n)
    sd = np.ones(n)
    y = rng.normal(size=n)
    out = expected_coverage_curve(y, mu, sd)
    emp = [r["empirical"] for r in out["coverage"]]
    assert all(b >= a - 1e-9 for a, b in zip(emp, emp[1:]))
    for r in out["coverage"]:
        assert r["nominal_inside_wilson95"], r


def test_subgroup_calibration_finds_the_bad_stratum():
    rng = np.random.default_rng(6)
    n = 3000
    g = rng.integers(0, 3, n)
    mu = np.zeros(n)
    sd = np.ones(n)
    y = rng.normal(size=n)
    y[g == 2] *= 3.0                       # one stratum is badly under-dispersed
    out = subgroup_calibration(y, mu, sd, g)
    assert out["worst_coverage_group"] == "2"
    assert out["per_group"]["2"]["coverage"]["empirical"] < 0.8
    assert out["coverage_spread"] > 0.1
    # aggregate coverage alone would have looked much less alarming
    assert out["overall"]["empirical"] > out["worst_coverage"]


def test_bootstrap_intervals():
    rng = np.random.default_rng(7)
    v = rng.normal(1.0, 2.0, 500)
    ci = bootstrap_ci(v, seed=1)
    assert ci.lo < v.mean() < ci.hi
    assert ci.hi - ci.lo < 1.0
    a = rng.normal(1.0, 1.0, 400)
    b = a - 0.5
    d = paired_bootstrap_difference(a, b, seed=2)
    assert d["excludes_zero"] and abs(d["difference"] - 0.5) < 0.05
    with pytest.raises(ValueError):
        paired_bootstrap_difference(a, b[:10])
