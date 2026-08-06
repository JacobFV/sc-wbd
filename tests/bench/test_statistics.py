"""The measurement machinery must detect what it claims to detect."""

from __future__ import annotations

import numpy as np
import pytest

from scwbd.bench.statistics import (
    bootstrap_ci,
    calibration,
    data_efficiency_curve,
    decision_regret,
    gaussian_log_score,
    paired_bootstrap,
    plot_calibration,
    plot_metric_intervals,
    selection_optimism,
    smoothing_check,
    stratified_bias,
    systematic_error,
)


def test_bootstrap_interval_covers_the_truth():
    rng = np.random.default_rng(0)
    covered = 0
    for i in range(60):
        x = rng.normal(1.0, 1.0, size=200)
        _, iv = bootstrap_ci(x, n_boot=300, seed=i)
        covered += int(iv.lo <= 1.0 <= iv.hi)
    assert covered >= 50  # nominal 95%, allow Monte-Carlo slack


def test_paired_bootstrap_calls_a_null_indistinguishable():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, size=500)
    b = a + rng.normal(0, 0.01, size=500)
    d = paired_bootstrap(a, b, n_boot=500, seed=0)
    assert d.indistinguishable
    assert not d.significant_win


def test_paired_bootstrap_finds_a_real_win():
    rng = np.random.default_rng(2)
    b = rng.normal(0, 1, size=500)
    a = b + 0.5
    d = paired_bootstrap(a, b, n_boot=500, seed=0)
    assert d.significant_win


def test_calibration_flags_overconfidence():
    rng = np.random.default_rng(3)
    y = rng.normal(0, 1, size=2000)
    honest = calibration(y, np.zeros_like(y), np.ones_like(y), seed=0, n_boot=100)
    overconf = calibration(y, np.zeros_like(y), 0.3 * np.ones_like(y), seed=0, n_boot=100)
    assert honest.overconfidence < 0.03
    assert overconf.overconfidence > 0.2
    assert overconf.coverage_error > honest.coverage_error


def test_calibration_reports_sharpness_so_vagueness_is_visible():
    rng = np.random.default_rng(4)
    y = rng.normal(0, 1, size=1000)
    vague = calibration(y, np.zeros_like(y), 5.0 * np.ones_like(y), seed=0, n_boot=100)
    assert vague.overconfidence == 0.0        # never overconfident...
    assert vague.sharpness == pytest.approx(5.0)   # ...because it says nothing


def test_selection_optimism_is_positive_when_selecting_among_noise():
    rng = np.random.default_rng(5)
    # ten identical models: any apparent winner is pure selection noise
    scores = rng.normal(0, 1, size=(10, 8))
    so = selection_optimism(scores, n_boot=300, seed=0)
    assert so.optimism > 0
    assert so.corrected_score < so.naive_score


def test_stratified_bias_localises_a_site_effect():
    rng = np.random.default_rng(6)
    n = 600
    site = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
    y = rng.normal(0, 1, size=n)
    pred = y + np.where(site == "B", 1.5, 0.0)
    ba = stratified_bias(y, pred, {"site": site}, seed=0, n_boot=200)
    assert ba.worst.level == "B"
    assert ba.worst_abs_bias > 1.0
    assert ba.spread > 1.0
    mag, status, _ = systematic_error(ba)
    assert status == "design_estimable"
    assert mag > 1.0


def test_systematic_error_without_strata_is_only_prior_specified():
    from scwbd.bench.statistics import BiasAnalysis

    mag, status, detail = systematic_error(BiasAnalysis())
    assert status == "prior_specified_sensitivity"
    assert "warning" in detail


def test_smoothing_check_fires_on_a_deliberately_oversmoothed_model():
    rng = np.random.default_rng(7)
    y = rng.normal(0, 1, size=(400, 3))
    good = y + rng.normal(0, 0.4, size=y.shape)
    flat = np.zeros_like(y) + y.mean(axis=0)          # stable, and useless
    effect = lambda a: float(np.std(a.reshape(a.shape[0], -1), axis=0).mean())
    v = smoothing_check(arm_name="flat", reference_name="good", y_true=y,
                        pred_arm=flat, pred_reference=good, effect=effect,
                        seed=0, n_boot=200)
    assert v.lower_variance          # its predictions are maximally stable
    assert v.attenuated
    assert v.smoothed_away
    assert "REJECT-PREFERENCE" in v.verdict


def test_smoothing_check_does_not_fire_on_an_honest_model():
    rng = np.random.default_rng(8)
    y = rng.normal(0, 1, size=(400, 3))
    good = y + rng.normal(0, 0.2, size=y.shape)
    noisy = y + rng.normal(0, 0.9, size=y.shape)
    effect = lambda a: float(np.std(a.reshape(a.shape[0], -1), axis=0).mean())
    v = smoothing_check(arm_name="good", reference_name="noisy", y_true=y,
                        pred_arm=good, pred_reference=noisy, effect=effect,
                        seed=0, n_boot=200)
    assert not v.smoothed_away
    assert v.effect_retention > 0.8


def test_decision_regret_is_zero_for_the_oracle():
    rng = np.random.default_rng(9)
    u = rng.normal(0, 1, size=(200, 4))
    r_oracle, _ = decision_regret(u, u.argmax(axis=1), seed=0)
    r_bad, _ = decision_regret(u, u.argmin(axis=1), seed=0)
    assert r_oracle == pytest.approx(0.0)
    assert r_bad > 1.0


def test_data_efficiency_curve_reports_intervals_and_n_to_target():
    curve = data_efficiency_curve([10, 100, 1000],
                                  [[-3.0, -3.2], [-2.0, -2.1], [-1.0, -1.05]],
                                  target=-1.5, seed=0)
    assert curve["n_to_target"] == 1000
    assert len(curve["ci_lo"]) == 3


def test_plot_helpers_refuse_a_point_estimate_without_an_interval(tmp_path):
    from scwbd.bench.report import Interval

    with pytest.raises(ValueError):
        plot_metric_intervals(["a"], [1.0], [None], path=str(tmp_path / "x.png"))
    p = plot_metric_intervals(["a", "b"], [1.0, -0.5],
                              [Interval(0.5, 1.5), Interval(-1.0, 0.0)],
                              path=str(tmp_path / "ok.png"))
    assert p.endswith("ok.png")


def test_plot_calibration_writes_a_file(tmp_path):
    rng = np.random.default_rng(10)
    y = rng.normal(0, 1, size=300)
    rep = calibration(y, np.zeros_like(y), np.ones_like(y), seed=0, n_boot=50)
    p = plot_calibration({"model": rep}, path=str(tmp_path / "cal.png"))
    assert p.endswith("cal.png")


def test_log_score_punishes_overconfidence():
    y = np.array([0.0, 1.0, -1.0])
    honest = gaussian_log_score(y, np.zeros_like(y), np.ones_like(y)).mean()
    overconf = gaussian_log_score(y, np.zeros_like(y), 0.1 * np.ones_like(y)).mean()
    assert honest > overconf
