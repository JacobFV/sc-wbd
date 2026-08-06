"""Behaviour and report: the DDM likelihood, and the report/state distinction.

The drift-diffusion density is validated against two independent analytic facts
that it does not itself compute: the gambler's-ruin choice probability and the
closed-form mean decision time.  The reporting stage is validated by showing
that the reported response differs from the executed response in a way the
ledger names.
"""

from __future__ import annotations

import math

import pytest
import torch

from scwbd.observe.base import RefusalR08
from scwbd.observe.behavior import (
    BehaviorObservationOperator,
    MotorStage,
    PerceptionStage,
    PolicyStage,
    ReportingBias,
    chronometric,
    ddm_choice_probability,
    ddm_mean_decision_time,
    drift_diffusion_pdf,
    psychometric,
)

torch.set_default_dtype(torch.float64)


@pytest.mark.parametrize(
    "v,a,w", [(0.8, 1.2, 0.5), (-1.5, 1.0, 0.4), (0.0, 1.0, 0.6), (2.0, 2.0, 0.5)]
)
def test_ddm_density_integrates_to_the_analytic_choice_probability(v, a, w):
    t = torch.linspace(1e-4, 60.0, 200_000, dtype=torch.float64)
    for side in ("lower", "upper"):
        d = drift_diffusion_pdf(t, drift=v, boundary=a, start_rel=w, boundary_hit=side)
        num = float(torch.trapz(d, t))
        ana = ddm_choice_probability(drift=v, boundary=a, start_rel=w, boundary_hit=side)
        assert num == pytest.approx(ana, abs=1e-5), (
            f"{side}: numeric {num:.6f} vs analytic {ana:.6f}"
        )


@pytest.mark.parametrize("v,a,w", [(0.8, 1.2, 0.5), (-1.5, 1.0, 0.4), (2.0, 2.0, 0.5)])
def test_ddm_density_reproduces_the_analytic_mean_decision_time(v, a, w):
    t = torch.linspace(1e-4, 60.0, 200_000, dtype=torch.float64)
    d = drift_diffusion_pdf(t, drift=v, boundary=a, start_rel=w, boundary_hit="lower")
    d = d + drift_diffusion_pdf(t, drift=v, boundary=a, start_rel=w, boundary_hit="upper")
    num = float(torch.trapz(d * t, t))
    assert num == pytest.approx(
        ddm_mean_decision_time(drift=v, boundary=a, start_rel=w), abs=1e-5
    )


def test_ddm_density_is_non_negative_and_respects_non_decision_time():
    t = torch.linspace(0.0, 3.0, 3000, dtype=torch.float64)
    d = drift_diffusion_pdf(t, drift=1.0, boundary=1.0, start_rel=0.5, non_decision=0.35)
    assert float(d.min()) >= 0.0
    assert float(d[t <= 0.35].max()) == 0.0


def test_both_series_expansions_agree_in_the_overlap():
    from scwbd.observe.behavior import _f0_large_time, _f0_small_time

    tau = torch.linspace(0.30, 0.55, 40, dtype=torch.float64)
    w = torch.full_like(tau, 0.45)
    s = _f0_small_time(tau, w)
    l = _f0_large_time(tau, w)
    rel = float(((s - l).abs() / l.abs().clamp_min(1e-30)).max())
    assert rel < 1e-8, f"small/large-time expansions disagree by {rel:.3e} at the switch"


def test_chronometric_matches_the_ddm_mean_for_an_unbiased_start():
    s = torch.tensor([0.2, 0.5, 1.0, 2.0], dtype=torch.float64)
    got = chronometric(s, drift_gain=1.0, boundary=1.0, non_decision=0.0)
    for i, v in enumerate(s):
        assert float(got[i]) == pytest.approx(
            ddm_mean_decision_time(drift=float(v), boundary=1.0, start_rel=0.5), rel=1e-9
        )


def test_psychometric_has_lapse_guess_and_a_criterion():
    s = torch.linspace(-5, 5, 101, dtype=torch.float64)
    p = psychometric(s, threshold=0.0, slope=1.0, lapse=0.03, guess=0.02)
    assert float(p.max()) == pytest.approx(0.97, abs=1e-3)
    assert float(p.min()) == pytest.approx(0.02, abs=1e-3)
    # criterion shifts the curve without changing its slope
    shifted = psychometric(s, threshold=0.0, slope=1.0, lapse=0.03, guess=0.02, criterion=1.0)
    assert float(shifted[50]) < float(p[50])
    # the grid step is 0.1, so a criterion of 1.0 shifts the curve by 10 samples
    d_p = float((p[51] - p[49]))
    d_s = float((shifted[51 + 10] - shifted[49 + 10]))
    assert d_s == pytest.approx(d_p, rel=1e-6)


# --------------------------------------------------------------------------
# report is not latent state
# --------------------------------------------------------------------------


def _operator(reporting: ReportingBias) -> BehaviorObservationOperator:
    return BehaviorObservationOperator(
        perception=PerceptionStage(internal_noise_sd=0.2),
        policy=PolicyStage(drift_gain=2.0, boundary=1.0),
        motor=MotorStage(non_decision_s=0.3, motor_slip_rate=0.01),
        reporting=reporting,
        dtype=torch.float64,
    )


def test_reporting_stage_is_mandatory():
    with pytest.raises(RefusalR08):
        BehaviorObservationOperator(reporting=None)


def test_report_and_execution_are_returned_separately():
    op = _operator(
        ReportingBias(
            response_bias=0.4,
            demand_characteristic=0.5,
            external_bound="post-experiment debrief plus a matched no-instruction "
            "control block",
        )
    )
    ev = torch.linspace(-0.6, 0.6, 300, dtype=torch.float64)
    r = op.observe(ev, seed=1)
    assert "executed_choice" in r.components and "reported_choice" in r.components
    flips = float(r.components["report_flipped"].sum())
    assert flips > 0, "a strong demand characteristic produced no report/execution gap"
    assert r.ledger.validity_domain["report_flip_rate"] > 0.0
    assert "model of the participant's REPORT" in r.ledger.validity_domain["claim_boundary"]


def test_reporting_bias_status_follows_the_design():
    swept = ReportingBias().bias_term()
    assert swept.status == "prior_specified_sensitivity"
    assert swept.half_width > 0.0

    measured = ReportingBias(
        response_bias=0.2, estimator="asymmetric payoff blocks with matched d-prime"
    ).bias_term()
    assert measured.status == "design_estimable"
    assert measured.interval == (0.2, 0.2)

    bounded = ReportingBias(external_bound="forced-report control condition").bias_term()
    assert bounded.status == "externally_bounded"


def test_response_bias_shifts_the_choice_proportion():
    ev = torch.zeros(2000, dtype=torch.float64)
    neutral = _operator(ReportingBias(external_bound="control block")).observe(ev, seed=2)
    biased = _operator(
        ReportingBias(
            response_bias=1.5, demand_characteristic=0.9,
            external_bound="control block",
        )
    ).observe(ev, seed=2)
    p_n = float(neutral.prediction[0].mean())
    p_b = float(biased.prediction[0].mean())
    assert abs(p_b - p_n) > 0.02, (
        f"reporting bias moved the response proportion only from {p_n:.3f} to {p_b:.3f}"
    )


def test_log_likelihood_is_finite_and_prefers_the_generating_drift():
    op = _operator(ReportingBias(external_bound="control block"))
    ev = torch.full((400,), 0.5, dtype=torch.float64)
    r = op.observe(ev, seed=5)
    choice, rt = r.prediction[0], r.prediction[1]

    ll_true = float(op.log_likelihood(choice, rt, ev).sum())
    ll_wrong = float(op.log_likelihood(choice, rt, -ev).sum())
    assert math.isfinite(ll_true)
    assert ll_true > ll_wrong, "the likelihood does not identify the drift sign"


def test_reaction_times_are_positive_and_include_non_decision_time():
    op = _operator(ReportingBias(external_bound="control block"))
    ev = torch.linspace(-1, 1, 300, dtype=torch.float64)
    r = op.observe(ev, seed=7)
    rt = r.prediction[1]
    assert float(rt.min()) > 0.0
    assert float(rt.min()) > 0.10, "some RT is faster than any plausible motor delay"
    assert float(rt.median()) > op.motor.non_decision_s


def test_behaviour_read_is_deterministic_given_a_seed():
    op = _operator(ReportingBias(external_bound="control block"))
    ev = torch.linspace(-1, 1, 200, dtype=torch.float64)
    a = op.observe(ev, seed=3)
    b = op.observe(ev, seed=3)
    assert torch.equal(a.prediction, b.prediction)


def test_perception_policy_motor_report_are_four_separable_stages():
    op = _operator(ReportingBias(external_bound="control block"))
    assert isinstance(op.perception, PerceptionStage)
    assert isinstance(op.policy, PolicyStage)
    assert isinstance(op.motor, MotorStage)
    assert isinstance(op.reporting, ReportingBias)
    psi = op.nuisance_priors
    assert {"sensory_gain", "drift_gain", "non_decision_s", "reporting_response_bias"} <= set(psi)
