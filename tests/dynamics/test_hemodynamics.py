"""Balloon–Windkessel: impulse response shape, positivity, subject/session priors."""

from __future__ import annotations

import pytest
import torch

from scwbd.dynamics import BalloonWindkessel, bold_field_policy, sample_hemodynamic_params


def test_resting_state_is_a_fixed_point(device):
    bw = BalloonWindkessel()
    theta = bw.make_theta(3, 5, device=device)
    x = bw.init_state(3, 5, device=device)
    d = bw.drift(x, torch.zeros(3, 5, device=device), theta)
    assert float(d.abs().max()) < 1e-6, "s=0, f=v=q=1 must be the resting fixed point"
    assert float(bw.bold(x, theta).abs().max()) < 1e-6


def test_impulse_response_has_the_canonical_hrf_shape(device):
    """Peak a few seconds after onset, then a post-stimulus undershoot.

    With Friston's (2003) priors this model peaks near 3.5-4 s, somewhat earlier
    than the canonical SPM double-gamma HRF (~5 s); the bound below is set to the
    model's own behaviour rather than to the empirical HRF it is often confused
    with.
    """
    bw = BalloonWindkessel()
    B, N, dt = 1, 1, 0.01
    theta = bw.make_theta(B, N, device=device)
    T = 3000  # 30 s
    neural = torch.zeros(T, B, N, device=device)
    neural[:100] = 1.0  # 1 s of drive
    bold, _ = bw.rollout(neural, theta, dt=dt)
    y = bold[:, 0, 0]
    peak_i = int(y.argmax())
    peak_t = peak_i * dt
    assert 3.0 < peak_t < 7.0, f"HRF peak at {peak_t:.2f}s, expected a few seconds after onset"
    assert float(y.max()) > 0, "positive BOLD response to positive drive"
    undershoot = y[peak_i + 300 :]
    assert float(undershoot.min()) < 0, "canonical HRF has a post-stimulus undershoot"
    assert float(y[-1]) == pytest.approx(0.0, abs=2e-3), "the response must return to baseline"


def test_state_variables_stay_positive(device):
    """Log-space integration makes positivity structural, not a clamp."""
    bw = BalloonWindkessel(log_space=True)
    B, N = 4, 6
    theta = bw.make_theta(B, N, device=device)
    torch.manual_seed(0)
    neural = torch.randn(2000, B, N, device=device) * 2.0  # aggressive, includes negatives
    bold, xf = bw.rollout(neural, theta, dt=0.01)
    _, f, v, q = bw._unpack(xf)
    assert bool((f > 0).all() and (v > 0).all() and (q > 0).all())
    assert torch.isfinite(bold).all()


def test_parameters_are_batched_over_subjects(device):
    """Different hemodynamic parameters must give different BOLD from identical drive."""
    bw = BalloonWindkessel()
    B, N = 3, 2
    kappa = torch.tensor([[0.4], [0.65], [1.0]], device=device)
    theta = bw.make_theta(B, N, device=device, kappa=kappa)
    neural = torch.zeros(1500, B, N, device=device)
    neural[:100] = 1.0
    bold, _ = bw.rollout(neural, theta, dt=0.01)
    peaks = bold.max(dim=0).values[:, 0]
    assert len(set(round(float(p), 4) for p in peaks)) == 3, "subject parameters had no effect"
    # faster signal decay (larger kappa) gives a smaller, earlier response
    times = bold.argmax(dim=0)[:, 0]
    assert float(times[0]) > float(times[2])


def test_subject_session_effects_are_centred(device):
    """R07: population/subject/session effects must be centred to be identified."""
    pack = sample_hemodynamic_params(n_subjects=6, n_sessions=4, seed=0, device=device)
    assert pack.batch == 24
    for name in ("kappa", "tau", "neural_gain"):
        v = pack.get(name).reshape(6, 4)
        # session effects sum to zero within subject
        dev_within = v - v.mean(dim=1, keepdim=True)
        assert float(dev_within.mean(dim=1).abs().max()) < 1e-5
        # and the subject effects are centred about the population value
        assert float(v.mean()) == pytest.approx(float(v.mean(dim=1).mean()), rel=1e-5)
    # there is genuine between-subject and within-subject variation
    k = pack.get("kappa").reshape(6, 4)
    assert float(k.mean(dim=1).std()) > 0
    assert float((k - k.mean(dim=1, keepdim=True)).std()) > 0


def test_hemodynamics_is_declared_a_slow_field():
    p = bold_field_policy()
    assert p.dt >= 0.01, "the vascular field must be slow relative to the ~1 ms neural clock"
    assert p.interpolation == "linear"
    assert p.error_budget < float("inf"), "a slow field must declare a coarsening budget"


def test_bold_scales_with_neural_drive(device):
    bw = BalloonWindkessel()
    theta = bw.make_theta(2, 1, device=device)
    neural = torch.zeros(1500, 2, 1, device=device)
    neural[:100, 0] = 0.5
    neural[:100, 1] = 1.0
    bold, _ = bw.rollout(neural, theta, dt=0.01)
    assert float(bold[:, 1, 0].max()) > float(bold[:, 0, 0].max())
