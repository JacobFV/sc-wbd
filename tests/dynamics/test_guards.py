"""The two runtime refusals: R05 (mechanism dominance) and R06 (semigroup).

Both are tested by *constructing the violation* and proving the guard fires —
a guard that has never fired is not evidence of anything.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from scwbd.dynamics import (
    AdaptiveStepper,
    DelayedConnectome,
    EdgeSet,
    LinearGaussian,
    MechanismDominanceGuard,
    MechanismRefusal,
    ResidualOperator,
    SemigroupCertificate,
    SemigroupGuard,
    SemigroupRefusal,
    SimConfig,
    WholeBrainSimulator,
    WilsonCowan,
    certify_semigroup,
    semigroup_residual,
)
from scwbd.dynamics.residual import ValiditySet
from scwbd.dynamics.types import NumericalBudget


# ---------------------------------------------------------------------------
# R06 — semigroup residual
# ---------------------------------------------------------------------------


def exact_flow(x, d):
    """A genuine semigroup: the exact flow of ``x' = -x``."""
    return x * math.exp(-d)


def violating_flow(x, d):
    """``Phi^d(x) = (1 + d) x`` — first-order accurate, *not* a semigroup.

    ``Phi^{d1+d2} - Phi^{d2} o Phi^{d1} = -d1 d2 x``: the composition and the
    single coarse step describe different dynamics, which is exactly what R06
    exists to detect.
    """
    return x * (1.0 + d)


def test_semigroup_residual_is_zero_for_a_true_semigroup(device):
    x = torch.randn(16, 4, device=device)
    eps = semigroup_residual(exact_flow, x, 0.1, 0.2)
    assert float(eps.max()) < 1e-6


def test_semigroup_residual_detects_the_violation(device):
    x = torch.randn(16, 4, device=device)
    eps = semigroup_residual(violating_flow, x, 0.1, 0.1)
    # relative residual = d1 d2 / (d1 + d2 + d1 d2) = 0.01 / 0.21
    assert float(eps.mean()) == pytest.approx(0.01 / 0.21, rel=1e-4)


def test_certificate_passes_for_exact_flow_and_fails_for_the_violator(device):
    x = torch.randn(32, 3, device=device)
    steps = [0.05, 0.1, 0.2]
    good = certify_semigroup(exact_flow, x, steps, tolerance=1e-3, name="exact")
    bad = certify_semigroup(violating_flow, x, steps, tolerance=1e-3, name="violator")
    assert good.passed and good.worst < 1e-5  # fp32 round-off floor
    assert not bad.passed
    assert len(bad.failing_pairs()) == len(bad.pairs)
    assert bad.as_dict()["code"] == "R06"


def test_adaptive_stepping_refused_without_a_certificate(device):
    """A learned propagator may not be stepped adaptively on trust (R06)."""
    guard = SemigroupGuard(None, learned=True, owner="learned_propagator")
    stepper = AdaptiveStepper(guard=guard)
    x0 = torch.ones(4, 2, device=device)
    with pytest.raises(SemigroupRefusal) as ei:
        stepper.run(lambda x, t: -x, x0, t_end=0.1)
    assert ei.value.violation.code == "R06"
    assert guard.violations


def test_adaptive_stepping_refused_when_residual_above_tolerance(device):
    x = torch.randn(16, 2, device=device)
    cert = certify_semigroup(violating_flow, x, [0.05, 0.1], tolerance=1e-3)
    guard = SemigroupGuard(cert, learned=True, owner="violator")
    with pytest.raises(SemigroupRefusal, match="above tolerance"):
        AdaptiveStepper(guard=guard).run(lambda xx, t: -xx, torch.ones(4, 2, device=device), t_end=0.1)


def test_adaptive_stepping_allowed_below_tolerance_and_budgeted(device):
    """Below tolerance the residual distribution enters the numerical budget."""
    x = torch.randn(16, 2, device=device)
    cert = certify_semigroup(exact_flow, x, [0.05, 0.1], tolerance=1e-3, name="exact")
    budget = NumericalBudget()
    guard = SemigroupGuard(cert, learned=True, owner="exact", budget=budget)
    res = AdaptiveStepper(rtol=1e-5, atol=1e-8, guard=guard).run(
        lambda xx, t: -xx, torch.ones(4, 2, device=device), t_end=0.5, dt_init=1e-3
    )
    assert res.n_steps > 0 and res.dt_max_used > res.dt_min_used  # it really adapted
    assert float(res.x.mean()) == pytest.approx(math.exp(-0.5), rel=1e-3)
    assert "semigroup_residual" in budget.entries


def test_coarse_for_fine_substitution_checks_every_prefix(device):
    x = torch.randn(8, 2, device=device)
    cert = certify_semigroup(violating_flow, x, [0.01, 0.02, 0.03], tolerance=1e-6)
    guard = SemigroupGuard(cert, learned=True, owner="violator")
    with pytest.raises(SemigroupRefusal, match="not certified"):
        guard.check_substitution(0.03, [0.01, 0.02])


def test_mechanistic_propagator_is_not_gated(device):
    """A non-learned propagator has an analytic error theory; R06 targets learned ones."""
    guard = SemigroupGuard(None, learned=False, owner="heun")
    guard.check_adaptive()  # must not raise
    res = AdaptiveStepper(guard=guard).run(lambda x, t: -x, torch.ones(2, 1, device=device), t_end=0.2)
    assert res.n_steps > 0


def test_learned_backend_declares_itself_learned():
    from scwbd.dynamics import LearnedNeuralOperator, WilsonCowan

    assert LearnedNeuralOperator(state_dim=2).learned is True
    assert WilsonCowan().learned is False


def test_learned_propagator_violates_semigroup_in_practice(device):
    """A trained-style step network with a step-size input is not a semigroup.

    This is the realistic version of the synthetic violator: a network that maps
    ``(x, d) -> x'`` has no structural reason to satisfy
    ``Phi^{d1+d2} = Phi^{d2} o Phi^{d1}``, and the certificate shows it.
    """
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, 2)).to(device)

    def prop(x, d):
        dd = torch.full((*x.shape[:-1], 1), float(d), device=x.device, dtype=x.dtype)
        return x + net(torch.cat([x, dd], dim=-1)) * d

    x = torch.randn(64, 2, device=device)
    cert = certify_semigroup(prop, x, [0.05, 0.1, 0.2], tolerance=1e-3, name="learned_step_net")
    assert not cert.passed, "an untrained step network should not accidentally be a semigroup"
    assert cert.worst > 1e-3


# ---------------------------------------------------------------------------
# R05 — mechanism dominance
# ---------------------------------------------------------------------------


def test_dominance_guard_fires_when_residual_dominates(device):
    guard = MechanismDominanceGuard(rho_max=0.25, on_violation="reclassify", owner="test_field")
    f_mech = torch.ones(8, 5, 2, device=device)
    r_big = 0.9 * torch.ones_like(f_mech)
    guard.observe(f_mech, r_big)
    rep = guard.finalize()
    assert rep.violated and rep.n_violations == 8
    assert rep.status == "surrogate"
    assert rep.ratio_max == pytest.approx(0.9, rel=1e-3)
    assert rep.as_dict()["code"] == "R05"


def test_dominance_guard_silent_when_mechanism_dominates(device):
    guard = MechanismDominanceGuard(rho_max=0.25)
    f_mech = torch.ones(8, 5, 2, device=device)
    guard.observe(f_mech, 0.05 * torch.ones_like(f_mech))
    rep = guard.finalize()
    assert not rep.violated
    assert rep.status == "mechanistic"
    assert rep.ratio_quantiles["q50"] == pytest.approx(0.05, rel=1e-3)


def test_dominance_guard_raise_mode(device):
    guard = MechanismDominanceGuard(rho_max=0.1, on_violation="raise", owner="strict")
    f = torch.ones(4, 3, 1, device=device)
    with pytest.raises(MechanismRefusal) as ei:
        guard.observe(f, 0.5 * torch.ones_like(f))
    assert ei.value.violation.code == "R05"


def test_dominance_guard_clamp_mode_bounds_the_ratio(device):
    guard = MechanismDominanceGuard(rho_max=0.2, on_violation="clamp")
    f = torch.ones(4, 3, 1, device=device)
    r = 1.5 * torch.ones_like(f)
    guard.observe(f, r)
    r_clamped = guard.apply(f, r)
    assert float(guard.ratio(f, r_clamped).max()) == pytest.approx(0.2, rel=1e-4)
    # the clamp is recorded: a silent clamp would itself be the R05 failure mode
    assert guard.finalize().violated


def test_validity_set_separates_in_domain_from_ood(device):
    from scwbd.dynamics import ParamPack

    vs = ValiditySet(param_bounds={"G": (0.0, 1.0)}, name="G_in_unit_interval")
    guard = MechanismDominanceGuard(rho_max=0.25, validity=vs, on_violation="reclassify")
    f = torch.ones(4, 2, 1, device=device)
    theta = ParamPack(
        {"G": torch.tensor([[0.5], [0.5], [5.0], [5.0]], device=device)}, batch=4, n_regions=2, device=device
    )
    guard.observe(f, 0.9 * torch.ones_like(f), theta)
    rep = guard.finalize()
    assert rep.n_in_validity == 2, "only the in-domain parameter sets certify the mechanism"
    assert rep.n_violations == 2
    assert rep.ood_ratio_max == pytest.approx(0.9, rel=1e-3)


def test_assert_mechanistic_fails_closed(device):
    guard = MechanismDominanceGuard(rho_max=0.1)
    f = torch.ones(2, 2, 1, device=device)
    guard.observe(f, 0.5 * torch.ones_like(f))
    with pytest.raises(MechanismRefusal, match="reclassified"):
        guard.assert_mechanistic()


def test_guard_fires_inside_a_real_rollout(device):
    """End-to-end: an oversized residual in the simulator reclassifies the module."""
    N, B = 8, 4
    edges = EdgeSet.random(N, density=0.3, seed=0, device=device)
    be = WilsonCowan()
    residual = ResidualOperator(state_dim=2, n_coupling_channels=1, scale=50.0).to(device)
    guard = MechanismDominanceGuard(rho_max=0.25, on_violation="reclassify", owner="wilson_cowan+residual")
    sim = WholeBrainSimulator(be, DelayedConnectome(edges), residual=residual, guard=guard)
    theta = be.sample_theta(B, N, seed=1, device=device)
    theta.set("velocity", torch.full((B, 1), 5.0, device=device))
    res = sim.run(theta, SimConfig(dt=1e-3, n_steps=50, seed=2))
    assert res.guard_report is not None
    assert res.guard_report["status"] == "surrogate"
    assert res.guard_report["n_violations"] > 0


def test_small_residual_leaves_the_module_mechanistic(device):
    N, B = 8, 4
    edges = EdgeSet.random(N, density=0.3, seed=0, device=device)
    be = WilsonCowan()
    residual = ResidualOperator(state_dim=2, n_coupling_channels=1, scale=1e-4).to(device)
    guard = MechanismDominanceGuard(rho_max=0.25, on_violation="reclassify")
    sim = WholeBrainSimulator(be, DelayedConnectome(edges), residual=residual, guard=guard)
    theta = be.sample_theta(B, N, seed=1, device=device)
    theta.set("velocity", torch.full((B, 1), 5.0, device=device))
    res = sim.run(theta, SimConfig(dt=1e-3, n_steps=50, seed=2))
    assert res.guard_report["status"] == "mechanistic"
    assert res.guard_report["n_violations"] == 0
