"""Backend interface conformance and per-backend physiological signatures.

Conformance is parametrized over *every* registered backend, which is what makes
the backends interchangeable in fact and not just in intention.
"""

from __future__ import annotations

import math

import pytest
import torch

from scwbd.dynamics import (
    DelayedConnectome,
    EdgeSet,
    JansenRit,
    Kuramoto,
    LearnedNeuralOperator,
    LinearGaussian,
    ReducedWongWang,
    ReducedWongWangSingle,
    SimConfig,
    StuartLandau,
    WholeBrainSimulator,
    WilsonCowan,
    assert_equal_capacity,
    get_backend,
    list_backends,
    match_capacity,
    tune_fic,
)
from scwbd.dynamics.backends import kuramoto_order_parameter, ou_stationary_covariance
from scwbd.dynamics.simulator import functional_connectivity

BACKEND_FACTORIES = {
    "wilson_cowan": WilsonCowan,
    "jansen_rit": JansenRit,
    "wong_wang": ReducedWongWang,
    "wong_wang_single": ReducedWongWangSingle,
    "stuart_landau": StuartLandau,
    "kuramoto": Kuramoto,
    "linear_gaussian": LinearGaussian,
    "learned_operator": lambda: LearnedNeuralOperator(state_dim=2),
}


def test_registry_covers_every_required_backend():
    required = {
        "wilson_cowan", "jansen_rit", "wong_wang", "stuart_landau",
        "kuramoto", "linear_gaussian", "learned_operator",
    }
    assert required <= set(list_backends())
    for name in required:
        assert get_backend(name) is not None


@pytest.mark.parametrize("name", sorted(BACKEND_FACTORIES))
def test_backend_interface_conformance(name, device):
    be = BACKEND_FACTORIES[name]().to(device)
    B, N = 6, 9
    theta = be.sample_theta(B, N, seed=0, device=device)
    x = be.init_state(B, N, seed=1, device=device, theta=theta)
    assert x.shape == (B, N, be.state_dim)
    assert x.dtype == torch.float32
    c = be.zero_coupling(x)
    assert c.shape == (B, N, be.n_coupling_channels)
    d = be.drift(x, c, theta)
    assert d.shape == x.shape and torch.isfinite(d).all()
    g = be.diffusion(x, theta)
    assert g.shape == x.shape and torch.isfinite(g).all()
    obs = be.observables(x)
    assert "activity" in obs and obs["activity"].shape == (B, N)
    cv = be.coupling_variable(x, theta)
    assert cv.shape == (B, N, be.n_coupling_channels)
    info = be.describe()
    assert info["mechanistic_status"] in {"mechanistic", "effective", "functional", "surrogate"}
    assert info["falsifier"], "every backend must state what would disable it (§4 claim gate)"


@pytest.mark.parametrize("name", sorted(BACKEND_FACTORIES))
def test_init_state_is_seed_deterministic(name, device):
    be = BACKEND_FACTORIES[name]().to(device)
    a = be.init_state(4, 5, seed=42, device=device)
    b = be.init_state(4, 5, seed=42, device=device)
    c = be.init_state(4, 5, seed=43, device=device)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


@pytest.mark.parametrize("name", sorted(BACKEND_FACTORIES))
def test_drift_is_batched_not_looped(name, device):
    """Each batch element must be an independent parameter set."""
    be = BACKEND_FACTORIES[name]().to(device)
    B, N = 5, 7
    theta = be.sample_theta(B, N, seed=3, device=device)
    x = be.init_state(B, N, seed=4, device=device)
    c = be.zero_coupling(x)
    full = be.drift(x, c, theta)
    for b in range(B):
        th_b = be.make_theta(1, N, device=device)
        for k, v in theta.values.items():
            th_b.set(k, v[b : b + 1] if isinstance(v, torch.Tensor) and v.shape[0] == B else v)
        one = be.drift(x[b : b + 1], c[b : b + 1], th_b)
        assert torch.allclose(full[b], one[0], atol=1e-5), f"{name} batch element {b}"


@pytest.mark.parametrize("name", sorted(BACKEND_FACTORIES))
def test_drift_is_differentiable(name, device):
    """Gradients flow through both the state and the coupling input.

    Kuramoto's drift is genuinely independent of its own phase (all the state
    dependence is in the pairwise coupling term), so the state gradient is
    correctly zero there; the coupling gradient must still be non-zero.
    """
    be = BACKEND_FACTORIES[name]().to(device)
    theta = be.sample_theta(2, 4, seed=0, device=device)
    x = be.init_state(2, 4, seed=0, device=device).requires_grad_(True)
    c = be.zero_coupling(x).requires_grad_(True)
    d = be.drift(x, c, theta)
    gx, gc = torch.autograd.grad(d.sum(), [x, c], allow_unused=True)
    assert gc is not None and torch.isfinite(gc).all() and float(gc.abs().sum()) > 0
    if name != "kuramoto":
        assert gx is not None and torch.isfinite(gx).all() and float(gx.abs().sum()) > 0


# ---------------------------------------------------------------------------
# Backend-specific signatures
# ---------------------------------------------------------------------------


def test_linear_gaussian_stationary_covariance_matches_simulation(device):
    """The T1 reference: simulated covariance must match the Lyapunov solution."""
    N, B = 6, 1
    torch.manual_seed(0)
    W = (torch.rand(N, N, device=device) < 0.4).float() * 0.15
    W.fill_diagonal_(0.0)
    be = LinearGaussian(state_dim=1)
    edges = EdgeSet.from_dense(W, torch.ones(N, N, device=device), evidence="hard", device=device)
    con = DelayedConnectome(edges, n_channels=1)
    theta = be.make_theta(B, N, device=device, tau=0.05, G=1.0, sigma=0.2, velocity=1e6)
    analytic = be.stationary_covariance(W, theta)[0]
    sim = WholeBrainSimulator(be, con)
    res = sim.run(theta, SimConfig(dt=1e-3, n_steps=60000, seed=5, warmup_steps=2000, record_every=5))
    a = res.activity()[:, 0]
    emp = torch.cov(a.T)
    rel = float((emp - analytic).abs().max() / analytic.abs().max())
    assert rel < 0.15, f"empirical vs Lyapunov covariance mismatch: {rel:.3f}"


def test_lyapunov_solver_refuses_unstable_systems(device):
    A = torch.tensor([[[0.5, 0.0], [0.0, -1.0]]], device=device)
    Q = torch.eye(2, device=device).unsqueeze(0)
    with pytest.raises(ValueError, match="Hurwitz"):
        ou_stationary_covariance(A, Q)


def test_wong_wang_fic_reaches_the_target_working_point(device):
    """FIC must drive the excitatory input current to ``b_E/a_E - 0.026 nA``."""
    N, B = 8, 3
    torch.manual_seed(1)
    W = (torch.rand(N, N, device=device) < 0.5).float() * 0.2
    W.fill_diagonal_(0.0)
    be = ReducedWongWang().to(device)
    theta = be.make_theta(B, N, device=device, G=torch.tensor([[0.5], [1.5], [2.5]], device=device))
    x0 = be.init_state(B, N, seed=0, device=device)

    def coupling(s):  # instantaneous coupling, the declared FIC approximation
        return torch.einsum("ij,bj->bi", W, s[..., 0]).unsqueeze(-1)

    res = tune_fic(be, theta, coupling, x0=x0, dt=1e-3, n_steps=600, n_rounds=25, lr=1.0, tol=0.01)
    err = (res.final_offset - res.target).abs().max()
    assert float(err) < 0.02, f"FIC did not converge: max offset error {float(err):.4f} nA"
    assert res.J_i.shape == (B, N)
    # stronger global coupling requires stronger feedback inhibition
    assert float(res.J_i[2].mean()) > float(res.J_i[0].mean())


def test_kuramoto_synchronises_with_coupling_strength(device):
    N, B = 24, 3
    edges = EdgeSet.random(N, density=1.0, seed=0, device=device)
    be = Kuramoto().to(device)
    con = DelayedConnectome(edges, mode="phase_difference", n_channels=1, normalize="max")
    sim = WholeBrainSimulator(be, con)
    K = torch.tensor([[0.0], [2.0], [40.0]], device=device)
    theta = be.make_theta(B, N, device=device, K=K, f=10.0, sigma=0.5, velocity=1e6)
    res = sim.run(theta, SimConfig(dt=1e-3, n_steps=2000, seed=1, warmup_steps=500, record_every=5))
    phase = res.observables["activity"]  # sin(phase); recover phase from the state
    ph = torch.asin(phase.clamp(-1, 1))
    R = kuramoto_order_parameter(ph).mean(dim=0)
    assert float(R[2]) > float(R[1]) > float(R[0]) - 0.05, f"order parameter not monotone in K: {R}"
    assert float(R[2]) > 0.7


def test_stuart_landau_bifurcation(device):
    """Below the Hopf bifurcation the amplitude is noise-driven; above it, a limit cycle."""
    be = StuartLandau().to(device)
    sim = WholeBrainSimulator(be, None)
    a = torch.tensor([[-0.5], [0.5]], device=device)
    theta = be.make_theta(2, 1, device=device, a=a, f=10.0, sigma=0.01, G=0.0)
    res = sim.run(theta, SimConfig(dt=1e-3, n_steps=4000, seed=0, warmup_steps=2000, store_state=True))
    amp = (res.states[..., 0] ** 2 + res.states[..., 1] ** 2).sqrt().mean(dim=0)[:, 0]
    assert float(amp[0]) < 0.15, "subcritical region should sit near the fixed point"
    assert float(amp[1]) == pytest.approx(math.sqrt(0.5), rel=0.1), "supercritical amplitude ~ sqrt(a)"


def test_jansen_rit_alpha_peak_and_inhibitory_rate_dependence(device):
    """The point of paying for six PSP states is a realistic EEG spectrum.

    Two claims, both relative to the PSP kinetics rather than to a fitted filter:
    (i) at literature parameters the dominant peak is in the alpha band, and
    (ii) speeding up the inhibitory PSP (larger ``b``) moves the peak up — the
    rhythm's frequency is set by the inhibitory time constant, which is exactly
    the commitment a spectral surrogate does not make.
    """
    be = JansenRit().to(device)
    sim = WholeBrainSimulator(be, None)
    b = torch.tensor([[50.0], [80.0]], device=device)
    theta = be.make_theta(2, 1, device=device, p_mean=220.0, sigma=5.0, b=b)
    dt = 5e-4
    res = sim.run(theta, SimConfig(dt=dt, n_steps=8000, seed=0, warmup_steps=2000))
    x = res.activity()[:, :, 0]
    x = x - x.mean(dim=0, keepdim=True)
    psd = torch.fft.rfft(x, dim=0).abs().pow(2)
    freqs = torch.fft.rfftfreq(x.shape[0], d=dt).to(device)
    band = (freqs > 2.0) & (freqs < 45.0)
    peaks = [float(freqs[band][psd[band, i].argmax()]) for i in range(2)]
    assert 8.0 < peaks[0] < 13.0, f"Jansen-Rit peak at {peaks[0]:.2f} Hz, expected the alpha band"
    assert peaks[1] > peaks[0], f"faster inhibition must raise the peak frequency: {peaks}"


def test_wilson_cowan_inhibition_reduces_excitatory_rate(device):
    be = WilsonCowan().to(device)
    sim = WholeBrainSimulator(be, None)
    c_ei = torch.tensor([[6.0], [12.0], [20.0]], device=device)
    theta = be.make_theta(3, 1, device=device, c_ei=c_ei, sigma=0.0, P=1.25)
    res = sim.run(theta, SimConfig(dt=1e-4, n_steps=3000, seed=0, stochastic=False))
    E = res.activity()[-500:].mean(dim=0)[:, 0]
    assert float(E[0]) > float(E[1]) > float(E[2]), f"E rate not monotone in inhibition: {E}"


# ---------------------------------------------------------------------------
# The equal-capacity control
# ---------------------------------------------------------------------------


def test_surrogate_is_capacity_matchable():
    target = 12000
    m = match_capacity(target, state_dim=2, n_coupling_channels=1)
    assert abs(m.capacity() - target) <= 0.10 * target


def test_capacity_mismatch_is_refused():
    a = LearnedNeuralOperator(state_dim=2, width=8)
    b = LearnedNeuralOperator(state_dim=2, width=128)
    with pytest.raises(ValueError, match="capacity mismatch"):
        assert_equal_capacity(a, b)
    assert_equal_capacity(a, LearnedNeuralOperator(state_dim=2, width=8))


def test_mechanistic_backends_hold_no_learnable_parameters():
    """Mechanistic parameters live in the batched ParamPack, not in nn.Parameters."""
    for name in ("wilson_cowan", "jansen_rit", "wong_wang", "stuart_landau", "kuramoto", "linear_gaussian"):
        assert BACKEND_FACTORIES[name]().capacity() == 0, name
    assert LearnedNeuralOperator(state_dim=2).capacity() > 0


def test_surrogate_may_compute_in_bfloat16_but_returns_fp32(device):
    """bf16 is permitted inside a learned operator, never in the solver (§3)."""
    be = LearnedNeuralOperator(state_dim=2, compute_dtype=torch.bfloat16).to(device)
    theta = be.sample_theta(2, 3, seed=0, device=device)
    x = be.init_state(2, 3, seed=0, device=device)
    d = be.drift(x, be.zero_coupling(x), theta)
    assert d.dtype == torch.float32
