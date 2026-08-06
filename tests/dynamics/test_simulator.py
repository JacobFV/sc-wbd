"""Simulator composition, determinism and interface guards."""

from __future__ import annotations

import pytest
import torch

from scwbd.dynamics import (
    DelayedConnectome,
    EdgeSet,
    Kuramoto,
    LocalField,
    SimConfig,
    StuartLandau,
    WholeBrainSimulator,
    WilsonCowan,
)
from scwbd.dynamics.simulator import fc_correlation, functional_connectivity


def _sim(device, N=12, backend=None):
    edges = EdgeSet.random(N, density=0.3, seed=0, device=device)
    be = backend or WilsonCowan()
    con = DelayedConnectome(edges, mode=be.coupling_kind, n_channels=be.n_coupling_channels)
    return WholeBrainSimulator(be.to(device), con), be, N


def test_rollout_is_bitwise_reproducible_in_deterministic_mode(device):
    """Determinism is a test, not an aspiration (ARCHITECTURE.md §3).

    The sparse scatter accumulates with atomics, so the *default* fast path is
    reproducible only to fp32 tolerance; ``SimConfig(deterministic=True)`` makes
    it bitwise.  Both properties are asserted, because both are contracts: the
    fast path may not drift beyond tolerance, and the deterministic path may not
    drift at all.
    """
    sim, be, N = _sim(device)
    theta = be.sample_theta(4, N, seed=0, device=device)
    theta.set("velocity", torch.full((4, 1), 5.0, device=device))
    cfg = SimConfig(dt=1e-3, n_steps=100, seed=7, deterministic=True)
    a = sim.run(theta, cfg).activity()
    b = sim.run(theta, cfg).activity()
    assert torch.equal(a, b), "deterministic mode must be bitwise reproducible"
    c = sim.run(theta, SimConfig(dt=1e-3, n_steps=100, seed=8, deterministic=True)).activity()
    assert not torch.equal(a, c), "a different seed must give a different trajectory"


def test_fast_path_is_reproducible_to_tolerance(device):
    sim, be, N = _sim(device)
    theta = be.sample_theta(4, N, seed=0, device=device)
    theta.set("velocity", torch.full((4, 1), 5.0, device=device))
    cfg = SimConfig(dt=1e-3, n_steps=100, seed=7)
    a = sim.run(theta, cfg).activity()
    b = sim.run(theta, cfg).activity()
    assert torch.allclose(a, b, atol=1e-4, rtol=1e-3)


def test_noise_free_mode_depends_only_on_the_initial_state(device):
    """With ``stochastic=False`` the seed enters only through ``init_state``."""
    sim, be, N = _sim(device)
    theta = be.sample_theta(2, N, seed=1, device=device)
    theta.set("velocity", torch.full((2, 1), 5.0, device=device))
    x0 = be.init_state(2, N, seed=123, device=device)
    a = sim.run(
        theta, SimConfig(dt=1e-3, n_steps=50, seed=1, stochastic=False, deterministic=True), x0=x0
    ).activity()
    b = sim.run(
        theta, SimConfig(dt=1e-3, n_steps=50, seed=99, stochastic=False, deterministic=True), x0=x0
    ).activity()
    assert torch.equal(a, b), "noise-free rollouts from the same x0 must coincide"


def test_recording_shapes_and_state_storage(device):
    sim, be, N = _sim(device)
    theta = be.sample_theta(3, N, seed=2, device=device)
    theta.set("velocity", torch.full((3, 1), 5.0, device=device))
    res = sim.run(
        theta,
        SimConfig(dt=1e-3, n_steps=100, seed=3, record_every=10, record=("activity", "E"), store_state=True),
    )
    assert res.activity().shape == (10, 3, N)
    assert res.observables["E"].shape == (10, 3, N)
    assert res.states.shape == (10, 3, N, be.state_dim)
    assert res.x_final.shape == (3, N, be.state_dim)
    assert "SimResult" in res.summary()


def test_warmup_is_not_recorded(device):
    sim, be, N = _sim(device)
    theta = be.sample_theta(2, N, seed=4, device=device)
    theta.set("velocity", torch.full((2, 1), 5.0, device=device))
    res = sim.run(theta, SimConfig(dt=1e-3, n_steps=50, seed=5, warmup_steps=100))
    assert res.activity().shape[0] == 50


def test_coupling_mode_mismatch_is_refused(device):
    edges = EdgeSet.random(8, density=0.3, seed=0, device=device)
    with pytest.raises(ValueError, match="coupling mode mismatch"):
        WholeBrainSimulator(Kuramoto().to(device), DelayedConnectome(edges, mode="additive", n_channels=1))


def test_coupling_channel_mismatch_is_refused(device):
    edges = EdgeSet.random(8, density=0.3, seed=0, device=device)
    with pytest.raises(ValueError, match="coupling channel mismatch"):
        WholeBrainSimulator(StuartLandau().to(device), DelayedConnectome(edges, n_channels=1))


def test_local_field_participates_in_the_factorization(device):
    """F = F_local + F_long + R_theta: each term must be separately visible."""
    edges = EdgeSet.random(16, density=0.3, seed=0, device=device)
    pos = torch.randn(16, 3, device=device) * 25
    be = WilsonCowan().to(device)
    sim = WholeBrainSimulator(be, DelayedConnectome(edges), local=LocalField.from_positions(pos, k=4))
    theta = be.sample_theta(2, 16, seed=0, device=device)
    theta.set("velocity", torch.full((2, 1), 5.0, device=device))
    theta.set("kappa_local", 0.5)
    x, buf, ds = sim.prepare(theta, SimConfig(dt=1e-3, n_steps=1, seed=0))
    parts = sim.field(x, buf, theta, 1e-3, return_parts=True)
    assert set(parts) >= {"total", "f_regional", "f_local", "f_mech", "residual", "coupling"}
    assert float(parts["f_local"].abs().max()) > 0
    assert torch.allclose(parts["total"], parts["f_regional"] + parts["f_local"], atol=1e-6)


def test_functional_connectivity_matches_numpy_corrcoef(device):
    torch.manual_seed(0)
    x = torch.randn(500, 2, 5, device=device)
    fc = functional_connectivity(x)
    ref = torch.corrcoef(x[:, 0].T)
    assert torch.allclose(fc[0], ref, atol=1e-4)
    assert torch.allclose(fc.diagonal(dim1=-2, dim2=-1), torch.ones(2, 5, device=device), atol=1e-4)


def test_fc_correlation_is_one_for_identical_matrices(device):
    torch.manual_seed(0)
    x = torch.randn(300, 3, 6, device=device)
    fc = functional_connectivity(x)
    c = fc_correlation(fc, fc)
    assert torch.allclose(c, torch.ones(3, device=device), atol=1e-4)


def test_describe_reports_the_composition(device):
    sim, be, N = _sim(device)
    d = sim.describe()
    assert d["backend"]["name"] == "wilson_cowan"
    assert d["connectome"]["n_edges"] > 0
    assert d["has_residual"] is False and d["has_guard"] is False
