"""The Kalman filter must equal the analytic solution on a linear--Gaussian
problem, on native clocks, with missing windows and unequal supports."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from scwbd.infer.filters import (
    LinearGaussianSSM,
    ObservationChannel,
    ensemble_kalman_filter,
    extended_kalman_filter,
    kalman_filter,
    multiepoch_kalman_filter,
    particle_filter,
    rts_smoother,
    simulate_lgssm,
    unscented_kalman_filter,
)
from scwbd.infer.linear_gaussian import structured_left_mul

torch.set_default_dtype(torch.float64)


def _random_multirate(seed=0, n=3, T=9):
    g = torch.Generator().manual_seed(seed)
    F = torch.tensor([[0.80, 0.10, 0.00],
                      [0.05, 0.70, 0.20],
                      [0.00, 0.10, 0.85]]).unsqueeze(0)
    A = torch.randn(1, n, n, generator=g) * 0.3
    Q = (A @ A.transpose(-1, -2) + 0.2 * torch.eye(n)).contiguous()
    B = torch.randn(1, n, n, generator=g) * 0.4
    P0 = (B @ B.transpose(-1, -2) + 0.5 * torch.eye(n)).contiguous()
    m0 = torch.randn(1, n, generator=g)
    # unequal supports: a 2-channel fast head and a 1-channel slow head
    H1 = torch.randn(1, 2, n, generator=g)
    H2 = torch.randn(1, 1, n, generator=g)
    R1 = 0.3 * torch.eye(2).unsqueeze(0)
    R2 = 0.7 * torch.eye(1).unsqueeze(0)
    inp = torch.randn(1, T, n, generator=g) * 0.2
    ssm = LinearGaussianSSM(
        F, Q, m0, P0,
        [ObservationChannel("fast", H1, R1, torch.arange(T)),
         ObservationChannel("slow", H2, R2,
                            torch.tensor([k for k in (2, 5, 8) if k < T]))],
        T, inp,
    )
    n_slow = len([k for k in (2, 5, 8) if k < T])
    y1 = torch.randn(1, T, 2, generator=g)
    y2 = torch.randn(1, n_slow, 1, generator=g)
    return ssm, {"fast": y1, "slow": y2}


def _joint_gaussian(ssm, data, masks=None):
    """Brute-force joint mean/covariance of every scheduled observation."""
    n, T = ssm.n, ssm.n_steps
    F, Q, m0, P0 = ssm.F[0], ssm.Q[0], ssm.m0[0], ssm.P0[0]
    inp = ssm.inputs[0] if ssm.inputs is not None else torch.zeros(T, n)
    mu_z = torch.zeros(T, n)
    Sig = torch.zeros(T, T, n, n)
    mu_z[0] = m0
    Sig[0, 0] = P0
    for k in range(T - 1):
        mu_z[k + 1] = F @ mu_z[k] + inp[k]
        Sig[k + 1, k + 1] = F @ Sig[k, k] @ F.T + Q
        for j in range(k + 1):
            Sig[k + 1, j] = F @ Sig[k, j]
            Sig[j, k + 1] = Sig[k + 1, j].T
    rows = []
    for k in range(T):
        for ch in ssm.channels:
            st = ch.steps.tolist()
            if k in st:
                j = st.index(k)
                if masks and ch.name in masks and float(masks[ch.name][0, j]) == 0:
                    continue
                rows.append((k, ch.H[0], ch.R[0], data[ch.name][0, j]))
    d = sum(r[1].shape[0] for r in rows)
    mu = torch.zeros(d); yv = torch.zeros(d); C = torch.zeros(d, d)
    offs, o = [], 0
    for (k, H, R, y) in rows:
        p = H.shape[0]
        mu[o:o + p] = H @ mu_z[k]
        yv[o:o + p] = y
        offs.append((o, p, k, H, R))
        o += p
    for (o1, p1, k1, H1, R1) in offs:
        for (o2, p2, k2, H2, R2) in offs:
            C[o1:o1 + p1, o2:o2 + p2] = H1 @ Sig[k1, k2] @ H2.T
            if o1 == o2:
                C[o1:o1 + p1, o2:o2 + p2] += R1
    L = torch.linalg.cholesky(C)
    r = yv - mu
    a = torch.cholesky_solve(r.unsqueeze(-1), L)
    ll = -0.5 * (
        float(r @ a.squeeze(-1))
        + 2 * float(torch.log(torch.diagonal(L)).sum())
        + d * math.log(2 * math.pi)
    )
    Czy = torch.zeros(T * n, d)
    for k in range(T):
        for (o1, p1, k1, H1, _R) in offs:
            Czy[k * n:(k + 1) * n, o1:o1 + p1] = Sig[k, k1] @ H1.T
    post = mu_z.reshape(-1) + (Czy @ a).squeeze(-1)
    return ll, post.reshape(T, n)


def test_kalman_matches_analytic_multirate():
    ssm, data = _random_multirate()
    fr = kalman_filter(ssm, data, store="all")
    ll, _ = _joint_gaussian(ssm, data)
    assert abs(float(fr.log_likelihood[0]) - ll) < 1e-9


def test_rts_smoother_matches_analytic_conditional_mean():
    ssm, data = _random_multirate(seed=3)
    fr = kalman_filter(ssm, data, store="all")
    sr = rts_smoother(ssm, fr)
    _, post = _joint_gaussian(ssm, data)
    assert float((sr.smoothed_mean[0] - post).abs().max()) < 1e-9


def test_missing_windows_equal_dropping_the_observation():
    """A masked sample must contribute exactly nothing -- not an imputed zero."""
    ssm, data = _random_multirate(seed=5)
    masks = {"fast": torch.ones(1, ssm.n_steps),
             "slow": torch.ones(1, ssm.channel("slow").n_obs)}
    masks["fast"][0, 2:5] = 0.0          # a contiguous missing window
    masks["slow"][0, 1] = 0.0
    fr = kalman_filter(ssm, data, masks)
    ll, _ = _joint_gaussian(ssm, data, masks)
    assert abs(float(fr.log_likelihood[0]) - ll) < 1e-9
    # and corrupting the masked samples changes nothing
    d2 = {k: v.clone() for k, v in data.items()}
    d2["fast"][0, 2:5] = 1e6
    d2["slow"][0, 1] = -1e6
    fr2 = kalman_filter(ssm, d2, masks)
    assert abs(float(fr2.log_likelihood[0]) - float(fr.log_likelihood[0])) < 1e-9


def test_channels_stay_on_native_clocks():
    ssm, data = _random_multirate(seed=11)
    used = kalman_filter(ssm, data).n_observations_used
    assert int(used["fast"][0]) == ssm.n_steps
    assert int(used["slow"][0]) == ssm.channel("slow").n_obs
    # the slow channel has a different support size: never resampled
    assert ssm.channel("fast").p != ssm.channel("slow").p


def test_multiepoch_equals_per_epoch(tiny_setup):
    from scwbd.infer.linear_gaussian import make_model

    cfg, proto, u0 = tiny_setup
    mdl = make_model(u0, cfg, proto)
    per = []
    datas = []
    for e in range(cfg.n_epochs):
        d, _ = simulate_lgssm(mdl.ssm(epoch=e), seed=100 + e, batch=1)
        datas.append(d)
        per.append(float(kalman_filter(mdl.ssm(epoch=e), d).log_likelihood[0]))
    data = {k: torch.stack([d[k][0] for d in datas], 0).unsqueeze(0) for k in datas[0]}
    out = multiepoch_kalman_filter(mdl.multiepoch_ssm(), data, n_epochs=cfg.n_epochs)
    got = out["log_likelihood"][0].tolist()
    assert np.allclose(got, per, rtol=0, atol=1e-6)


def test_structured_left_mul_equals_dense(tiny_setup):
    from scwbd.infer.linear_gaussian import make_model

    cfg, proto, u0 = tiny_setup
    mdl = make_model(u0, cfg, proto)
    X = torch.randn(1, cfg.n_state, 5, dtype=mdl.F.dtype, device=mdl.F.device)
    fast = structured_left_mul(mdl.F, cfg)(X)
    assert float((fast - mdl.F @ X).abs().max()) < 1e-12


def test_multiepoch_rejects_per_epoch_masks(tiny_setup):
    from scwbd.infer.linear_gaussian import make_model

    cfg, proto, u0 = tiny_setup
    mdl = make_model(u0, cfg, proto)
    ssm = mdl.multiepoch_ssm()
    data = {
        "eeg": torch.zeros(1, cfg.n_epochs, len(cfg.eeg_steps()), 4, dtype=mdl.F.dtype,
                           device=mdl.F.device),
        "bold": torch.zeros(1, cfg.n_epochs, len(cfg.bold_steps()), 3, dtype=mdl.F.dtype,
                            device=mdl.F.device),
    }
    bad = {"eeg": torch.ones(1, cfg.n_epochs, len(cfg.eeg_steps()))}
    with pytest.raises(ValueError, match="shared across epochs"):
        multiepoch_kalman_filter(ssm, data, bad, n_epochs=cfg.n_epochs)


def test_nonlinear_filters_reduce_to_kalman():
    """EKF, UKF, EnKF and the particle filter must all recover the exact answer
    when the model is in fact linear and Gaussian."""
    ssm, data = _random_multirate(seed=17, T=7)
    ref = kalman_filter(ssm, data, store="all")
    F, Q, m0, P0 = ssm.F, ssm.Q, ssm.m0, ssm.P0
    inp = ssm.inputs
    f = lambda x, k: (F[0] @ x.unsqueeze(-1)).squeeze(-1) + inp[0, k]  # noqa: E731
    h = {c.name: (lambda x, k, H=c.H: (H[0] @ x.unsqueeze(-1)).squeeze(-1))
         for c in ssm.channels}
    R = {c.name: c.R for c in ssm.channels}
    steps = {c.name: c.steps for c in ssm.channels}

    ek = extended_kalman_filter(f, h, Q, R, m0, P0, data, steps, ssm.n_steps)
    assert abs(float(ek.log_likelihood[0]) - float(ref.log_likelihood[0])) < 1e-8
    uk = unscented_kalman_filter(f, h, Q, R, m0, P0, data, steps, ssm.n_steps,
                                 alpha=1.0, kappa=1.0)
    assert abs(float(uk.log_likelihood[0]) - float(ref.log_likelihood[0])) < 1e-6

    en = ensemble_kalman_filter(f, h, Q, R, m0, P0, data, steps, ssm.n_steps,
                               n_ensemble=4000, seed=1)
    err = (en.ensemble_mean[0, -1] - ref.filtered_mean[0, -1]).abs().max()
    scale = ref.filtered_cov[0, -1].diagonal().sqrt().max()
    assert float(err / scale) < 0.15

    logpdf = {}
    for c in ssm.channels:
        def lp(X, y, k, H=c.H, Rm=c.R):
            v = y.unsqueeze(1) - torch.einsum("pn,ben->bep", H[0], X)
            Li = torch.linalg.inv(Rm[0])
            q = torch.einsum("bep,pq,beq->be", v, Li, v)
            return -0.5 * (q + torch.logdet(Rm[0]) + H.shape[1] * math.log(2 * math.pi))
        logpdf[c.name] = lp
    pf = particle_filter(f, logpdf, m0, P0, Q, data, steps, ssm.n_steps,
                         n_particles=20000, seed=2)
    rel = abs(float(pf.log_likelihood[0]) - float(ref.log_likelihood[0]))
    assert rel < 1.0, f"particle filter log-evidence off by {rel}"


def test_determinism_of_simulation(tiny_setup):
    from scwbd.infer.linear_gaussian import make_model

    cfg, proto, u0 = tiny_setup
    ssm = make_model(u0, cfg, proto).ssm(epoch=0)
    a, _ = simulate_lgssm(ssm, seed=99, batch=2)
    b, _ = simulate_lgssm(ssm, seed=99, batch=2)
    c, _ = simulate_lgssm(ssm, seed=100, batch=2)
    for k in a:
        assert torch.equal(a[k], b[k])
        assert not torch.equal(a[k], c[k])
