"""Delayed connectome coupling: correctness against analytic solutions and controls."""

from __future__ import annotations

import math

import pytest
import torch

from scwbd.dynamics import (
    DelayBuffer,
    DelayedConnectome,
    EdgePenalty,
    EdgeSet,
    LinearGaussian,
    LocalField,
    SimConfig,
    WholeBrainSimulator,
    randomized_control,
)
from scwbd.dynamics.coupling import EDGE_CLASS_CODES


# ---------------------------------------------------------------------------
# History buffer
# ---------------------------------------------------------------------------


def test_delay_buffer_integer_delay_exact(device):
    buf = DelayBuffer(2, 3, 1, max_delay_steps=8, device=device)
    for k in range(12):
        buf.push(torch.full((2, 3, 1), float(k), device=device))
    idx = torch.tensor([0, 1, 2], device=device)
    for d in (0, 1, 5, 8):
        v = buf.read(idx, torch.tensor([float(d)] * 3, device=device))
        assert torch.allclose(v, torch.full_like(v, 11.0 - d)), f"delay {d}"


def test_delay_buffer_fractional_delay_is_exact_on_a_ramp(device):
    """Linear interpolation is exact for a linear signal — hand-computed.

    After pushing v_k = k for k = 0..11 the head holds 11.  A delay of 2.5 steps
    must return 0.5*(9) + 0.5*(8) = 8.5.
    """
    buf = DelayBuffer(1, 1, 1, max_delay_steps=8, device=device)
    for k in range(12):
        buf.push(torch.full((1, 1, 1), float(k), device=device))
    idx = torch.tensor([0], device=device)
    v = buf.read(idx, torch.tensor([2.5], device=device))
    assert float(v) == pytest.approx(8.5, abs=1e-5)
    v = buf.read(idx, torch.tensor([0.25], device=device))
    assert float(v) == pytest.approx(10.75, abs=1e-5)


def test_delay_buffer_per_parameter_set_delays(device):
    """Conduction velocity is a batched parameter, so delays differ across the batch."""
    buf = DelayBuffer(3, 2, 1, max_delay_steps=8, device=device)
    for k in range(10):
        buf.push(torch.full((3, 2, 1), float(k), device=device))
    idx = torch.tensor([0, 1], device=device)
    d = torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], device=device)
    v = buf.read(idx, d)
    assert torch.allclose(v.squeeze(-1), 9.0 - d)


# ---------------------------------------------------------------------------
# Analytic delayed oscillator
# ---------------------------------------------------------------------------


def test_delayed_coupling_matches_analytic_dde(device):
    """``x'(t) = -a x(t - tau)`` with constant history, by the method of steps.

        t in [0, tau]:    x = x0 (1 - a t)
        t in [tau, 2tau]: x = x0 (1 - a t + a^2 (t - tau)^2 / 2)
    """
    a, tau, x0v = 3.0, 0.05, 1.0
    dt = 1e-4
    # one region with a self-edge whose tract length gives exactly tau
    velocity = 5.0  # m/s
    distance_mm = velocity * 1000.0 * tau
    edges = EdgeSet(
        src=torch.zeros(1, dtype=torch.long, device=device),
        dst=torch.zeros(1, dtype=torch.long, device=device),
        weight=torch.ones(1, device=device),
        distance_mm=torch.full((1,), distance_mm, device=device),
        evidence=torch.full((1,), EDGE_CLASS_CODES["hard"], dtype=torch.long, device=device),
        n_regions=1,
    )
    backend = LinearGaussian(state_dim=1)
    con = DelayedConnectome(edges, mode="additive", n_channels=1)
    sim = WholeBrainSimulator(backend, con)
    theta = backend.make_theta(
        1, 1, device=device, tau=1e9, self_gain=0.0, G=-a, I=0.0, sigma=0.0, velocity=velocity
    )
    x0 = torch.full((1, 1, 1), x0v, device=device)
    n = int(round(2 * tau / dt))
    res = sim.run(theta, SimConfig(dt=dt, n_steps=n, seed=0, stochastic=False, method="euler"), x0=x0)
    traj = res.activity()[:, 0, 0]

    def analytic(t):
        if t <= tau:
            return x0v * (1 - a * t)
        return x0v * (1 - a * t + a * a * (t - tau) ** 2 / 2)

    for frac in (0.25, 0.5, 0.9, 1.5, 1.99):
        t = frac * tau
        k = int(round(t / dt)) - 1
        got = float(traj[k])
        want = analytic((k + 1) * dt)
        assert got == pytest.approx(want, abs=2e-3), f"t={t:.4f}: got {got:.6f} want {want:.6f}"


def test_zero_delay_reduces_to_instantaneous_coupling(device):
    """With infinite conduction velocity the delayed operator equals a plain matmul."""
    N = 12
    edges = EdgeSet.random(N, density=0.3, seed=1, device=device)
    con = DelayedConnectome(edges, n_channels=1)
    buf = DelayBuffer(2, N, 1, 2, device=device)
    x = torch.randn(2, N, 1, device=device)
    buf.push(x)
    theta = LinearGaussian().make_theta(2, N, device=device, velocity=1e9)
    got = con(buf, theta, dt=1e-3)
    W = edges.to_dense(allow_dense=True)  # legitimate: this is the declared control
    want = torch.einsum("ij,bj->bi", W, x[..., 0]).unsqueeze(-1)
    assert torch.allclose(got, want, atol=1e-4)


# ---------------------------------------------------------------------------
# Batched == looped
# ---------------------------------------------------------------------------


def test_batched_equals_looped_coupling(device):
    """The batch axis is a parameter axis, not an approximation."""
    N, B = 16, 5
    edges = EdgeSet.random(N, density=0.25, seed=2, device=device)
    con = DelayedConnectome(edges, n_channels=1)
    buf = DelayBuffer(B, N, 1, 10, device=device)
    torch.manual_seed(0)
    for _ in range(12):
        buf.push(torch.randn(B, N, 1, device=device))
    vel = torch.linspace(2.0, 9.0, B, device=device).reshape(B, 1)
    theta = LinearGaussian().make_theta(B, N, device=device, velocity=vel)
    batched = con(buf, theta, dt=1e-3)
    for b in range(B):
        sub = DelayBuffer(1, N, 1, 10, device=device)
        sub.buf.copy_(buf.buf[b : b + 1])
        sub.head = buf.head
        th_b = LinearGaussian().make_theta(1, N, device=device, velocity=vel[b : b + 1])
        one = con(sub, th_b, dt=1e-3)
        assert torch.allclose(batched[b], one[0], atol=1e-6), f"batch element {b} differs"


def test_batched_equals_looped_full_rollout(device):
    from scwbd.dynamics import WilsonCowan

    N, B = 10, 4
    edges = EdgeSet.random(N, density=0.3, seed=5, device=device)
    be = WilsonCowan()
    con = DelayedConnectome(edges, n_channels=1)
    sim = WholeBrainSimulator(be, con)
    theta = be.sample_theta(B, N, seed=4, device=device)
    theta.set("velocity", torch.linspace(3.0, 8.0, B, device=device).reshape(B, 1))
    cfg = SimConfig(dt=5e-4, n_steps=100, seed=9, stochastic=False)
    full = sim.run(theta, cfg).activity()
    for b in range(B):
        th_b = be.make_theta(1, N, device=device)
        for k, v in theta.values.items():
            th_b.set(k, v[b : b + 1] if isinstance(v, torch.Tensor) and v.shape[0] == B else v)
        x0 = be.init_state(B, N, seed=cfg.seed, device=device)[b : b + 1]
        one = sim.run(th_b, cfg, x0=x0).activity()
        assert torch.allclose(full[:, b], one[:, 0], atol=1e-5), f"batch element {b}"


# ---------------------------------------------------------------------------
# Evidence classes and controls
# ---------------------------------------------------------------------------


def test_dense_materialisation_refused_by_default(device):
    edges = EdgeSet.random(8, density=0.3, seed=0, device=device)
    with pytest.raises(RuntimeError, match="G2 control"):
        edges.to_dense()
    W = edges.to_dense(allow_dense=True)
    assert W.shape == (8, 8)


def test_evidence_class_masking_partitions_the_graph(device):
    edges = EdgeSet.random(40, density=0.2, seed=3, device=device)
    counts = edges.class_counts()
    hard = edges.mask_classes(["hard"])
    hs = edges.mask_classes(["hard", "soft"])
    assert hard.n_edges == counts["hard"]
    assert hs.n_edges == counts["hard"] + counts["soft"]
    assert hs.n_edges + counts["proposed"] == edges.n_edges


def test_hard_edges_are_a_fixed_mask(device):
    """Hard-supported pathways keep the anatomical weight; only soft/proposed learn."""
    edges = EdgeSet.random(20, density=0.25, seed=6, device=device)
    con = DelayedConnectome(edges, learn_soft_weights=True)
    hard = edges.evidence == EDGE_CLASS_CODES["hard"]
    assert float(con.learnable_mask[hard].sum()) == 0.0
    assert float(con.learnable_mask[~hard].min()) == 1.0
    with torch.no_grad():
        con.log_dev.fill_(1.0)  # try to move every edge
    w = con.effective_weights()
    base = edges.weight
    assert torch.allclose(w[0][hard], base[hard]), "a hard edge moved"
    assert not torch.allclose(w[0][~hard], base[~hard]), "soft/proposed edges failed to move"


def test_proposed_edges_require_a_declared_model_comparison(device):
    edges = EdgeSet.random(30, density=0.2, seed=7, device=device)
    con = DelayedConnectome(edges, learn_soft_weights=True)
    pen = EdgePenalty()
    with pytest.raises(RuntimeError, match="declared model comparison"):
        pen(con)
    out = pen(con, in_model_comparison=True)
    assert float(out["proposed"].detach()) > 0
    assert float(out["total"].detach()) >= float(out["soft"].detach())


def test_missing_tract_lengths_are_not_imputed(device):
    """Rule 1: missing data is never imputed as zero."""
    W = torch.rand(6, 6, device=device)
    edges = EdgeSet.from_dense(W, None, evidence="soft", threshold=0.5, device=device)
    con = DelayedConnectome(edges)
    theta = LinearGaussian().make_theta(1, 6, device=device, velocity=5.0)
    with pytest.raises(ValueError, match="NaN"):
        con.delay_steps(theta, 1e-3)


def test_delays_require_an_explicit_velocity(device):
    edges = EdgeSet.random(6, density=0.4, seed=0, device=device)
    con = DelayedConnectome(edges)
    theta = LinearGaussian().make_theta(1, 6, device=device)
    with pytest.raises(KeyError, match="velocity"):
        con.delay_steps(theta, 1e-3)


def test_randomized_control_preserves_edge_count(device):
    edges = EdgeSet.random(30, density=0.2, seed=1, device=device)
    ctrl = randomized_control(edges, seed=2)
    assert ctrl.n_edges == edges.n_edges
    assert torch.allclose(torch.sort(ctrl.weight).values, torch.sort(edges.weight).values)
    assert "randomized_control" in ctrl.provenance


# ---------------------------------------------------------------------------
# Local field
# ---------------------------------------------------------------------------


def test_laplacian_annihilates_constants_and_conserves_the_mean(device):
    pos = torch.randn(24, 3, device=device) * 30.0
    lf = LocalField.from_positions(pos, k=5, sigma_mm=20.0, include_kernel=False)
    ones = torch.ones(2, 24, 1, device=device)
    assert float(lf.laplacian_apply(ones).abs().max()) < 1e-5
    x = torch.randn(2, 24, 1, device=device)
    theta = LinearGaussian().make_theta(2, 24, device=device, kappa_local=0.5)
    dx = lf(x, theta)
    # symmetric weights -> the diffusion term conserves the sum
    assert float(dx.sum(dim=1).abs().max()) < 1e-4


def test_local_field_smooths_high_frequency_content(device):
    pos = torch.stack([torch.arange(32, device=device, dtype=torch.float32), torch.zeros(32, device=device)], dim=-1)
    lf = LocalField.from_positions(pos, k=2, sigma_mm=1.5, include_kernel=False)
    theta = LinearGaussian().make_theta(1, 32, device=device, kappa_local=0.2)
    alternating = torch.tensor([(-1.0) ** i for i in range(32)], device=device).reshape(1, 32, 1)
    smooth = torch.sin(torch.linspace(0, math.pi, 32, device=device)).reshape(1, 32, 1)
    d_alt = float(lf(alternating, theta).abs().mean())
    d_smooth = float(lf(smooth, theta).abs().mean())
    assert d_alt > 10 * d_smooth, "graph diffusion must damp high spatial frequencies hardest"
