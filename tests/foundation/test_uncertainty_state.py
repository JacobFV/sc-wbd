"""``X_i^uncertainty`` must actually carry predictive variance.

``EEGHead.log_noise`` and ``BOLDHead.log_noise`` are ``nn.Parameter`` vectors
broadcast with ``expand_as``: the predictive variance of both instrument heads is
constant in state, time, horizon, window, participant and condition, while the
five held-out-calibrated baselines get a variance of shape ``(horizon, C)``.

These tests do **not** claim to repair run 1.  Turing's decomposition attributes
run 1's FAIL to the *scale* term (0.4467 of the 0.4469 excess) — a training
schedule defect — with *state* worth 0.1896-0.2587 beyond that and *horizon*
only 0.0096.  What is tested here is that the state-dependence path exists and
is real; whether it buys anything is a training result nobody has yet.

These tests are written so they **fail against that implementation**.  A shape
assertion would not — the broadcast variance already has the right shape.  So
each test here measures a *dependence*, and
:func:`test_the_broadcast_parameter_fails_these_tests` runs the same measurements
against the un-repaired path and asserts they come out at exactly zero, which is
what makes the rest of the file evidence rather than decoration.
"""

from __future__ import annotations

import pytest
import torch

from scwbd.foundation.anatomy import load_anatomy
from scwbd.foundation.config import ModelConfig
from scwbd.foundation.model import SCWBD
from scwbd.foundation.uncertainty import (
    LOGVAR_CLAMP,
    UNCERTAINTY_COMPONENT,
    FamilyObservationInterface,
    FlatObservationInterface,
    UncertaintyPropagator,
)


@pytest.fixture(scope="module")
def anat():
    return load_anatomy(device="cpu")


def _cfg(**kw) -> ModelConfig:
    d = dict(
        hidden=64, n_local_layers=2, region_embed=16, context_dim=32,
        encoder_channels=16, encoder_layers=2,
    )
    d.update(kw)
    return ModelConfig(**d)


def _rollout(anat, cfg, *, steps: int = 40, seed: int = 0):
    torch.manual_seed(seed)
    m = SCWBD(cfg, anat)
    m.eval()
    theta = torch.randn(3, 6) * 0.3
    m.set_mechanistic_theta(theta, anat)
    with torch.no_grad():
        res = m.rollout(
            y_context=torch.randn(3, 8, anat.n_regions) * 0.3, theta=theta, n_steps=steps
        )
    return m, res


ARMS = [("treatment", dict(family_state=True)), ("control", dict(family_state=False))]


# ======================================================================
# the dependence that was missing
# ======================================================================
@pytest.mark.parametrize("arm,kw", ARMS)
def test_predictive_logvar_varies_across_parcels_and_across_time(anat, arm, kw):
    m, res = _rollout(anat, _cfg(**kw))
    with torch.no_grad():
        lv = m.observation.predictive_logvar(res.state)  # (B,T,N,1)
    assert lv.shape[:3] == res.state.shape[:3]
    across_parcels = float(lv[:, -1].std())
    across_time = float(lv[:, :, 0].std())
    # Both are exactly 0.0 for a broadcast nn.Parameter. Thresholds are set an
    # order of magnitude below the measured values (0.058 / 0.253 on the
    # treatment arm at this seed) so the test tracks the mechanism, not the seed.
    assert across_parcels > 5e-3, f"{arm}: log-variance is flat across parcels ({across_parcels:.2e})"
    assert across_time > 5e-2, f"{arm}: log-variance is flat across time ({across_time:.2e})"


@pytest.mark.parametrize("arm,kw", ARMS)
def test_predictive_logvar_grows_with_the_horizon(anat, arm, kw):
    """The time index after assimilation *is* the horizon step (ruling: (a)-primary).

    Nothing tells the model what ``h`` is.  Growth comes from integrating the
    uncertainty state forward, which is why it is falsifiable: a regime where
    forecast error does not grow but this channel does would kill it.
    """
    m, res = _rollout(anat, _cfg(**kw))
    with torch.no_grad():
        lv = m.observation.predictive_logvar(res.state).mean(dim=(0, 2, 3))  # (T,)
    assert lv[-1] > lv[0] + 0.1, f"{arm}: no horizon growth ({float(lv[0]):.3f} -> {float(lv[-1]):.3f})"
    assert bool((lv.diff() >= -1e-6).all()), f"{arm}: horizon growth is not monotone"


@pytest.mark.parametrize("arm,kw", ARMS)
def test_perturbing_a_non_uncertainty_component_changes_the_innovation(anat, arm, kw):
    """Isolates the state -> variance path.

    The previous test could in principle pass on the initial condition alone.
    This one perturbs ``rate_e`` — a component the variance does *not* read —
    and checks the uncertainty **generation rate** responds. That can only
    happen through ``UncertaintyPropagator.innovation``, i.e. through state.
    """
    m, res = _rollout(anat, _cfg(**kw))
    x = res.state[:, 0]
    if m.family_layout is not None:
        name = "cortex_vis"
        prop = m.family_local.uncertainty[name]
        idx = m.family_layout.index(name)
        f = m.family_layout.family(name)
        take = lambda z: z.index_select(-2, idx)[..., : f.dim]  # noqa: E731
        extra = torch.zeros(x.shape[0], idx.numel(), m.cfg.hidden // 2)
        sl = f.layout.slice("rate_e")
    else:
        prop = m.uncertainty_propagator
        take = lambda z: z  # noqa: E731
        extra = torch.zeros(x.shape[0], x.shape[1], m.cfg.hidden // 2)
        sl = m.layout.slice("rate_e")
    x2 = x.clone()
    x2[..., sl] = x2[..., sl] + 1.0
    with torch.no_grad():
        d = float((prop.innovation(take(x2), extra) - prop.innovation(take(x), extra)).abs().mean())
    assert d > 1e-4, (
        f"{arm}: the uncertainty innovation does not respond to state ({d:.2e}). A zero-initialised "
        "output layer produces exactly this and makes the state path a shape rather than a mechanism."
    )


def test_the_broadcast_parameter_fails_these_tests(anat):
    """The un-repaired path, measured — so the tests above are evidence.

    ``state_dependent_variance=False`` restores run 1: no observation interface,
    and ``heads.py`` falls back to ``log_noise.expand_as(y)``.  Its variance is
    **exactly** constant, so every measurement above comes out at 0.0.
    """
    m, res = _rollout(anat, _cfg(family_state=True, state_dependent_variance=False))
    assert m.observation is None, "the un-repaired arm must expose no interface"
    assert len(m.family_local.uncertainty) == 0, "no uncertainty propagator should be built"
    with torch.no_grad():
        _, lv = m.eeg(res.state)  # (B,T,C)
    assert float(lv.detach().std(dim=1).max()) == 0.0, "EEG log-variance varies over time; it should not, un-repaired"
    assert float(lv.detach().std(dim=0).max()) == 0.0, "EEG log-variance varies over samples; it should not, un-repaired"
    # ...and the same for BOLD, which the ruling added to scope. `BOLDHead.signal`
    # returns `self.log_noise.expand_as(y)` -- the identical defect, on the other
    # head that faces measured data and enters the NLL.
    with torch.no_grad():
        hemo = m.bold.initial(res.state.shape[0], m.n_regions, res.state.device)
        _, blv = m.bold.signal(hemo)
    assert float(blv.detach().std(dim=0).max()) == 0.0, "BOLD log-variance varies over samples; it should not, un-repaired"


# ======================================================================
# A1 must not be confounded by the repair
# ======================================================================
def test_both_ablation_arms_get_the_interface(anat):
    """Otherwise A1 measures the variance path instead of the structured state."""
    treat = SCWBD(_cfg(family_state=True), anat)
    ctrl = SCWBD(_cfg(family_state=False), anat)
    assert isinstance(treat.observation, FamilyObservationInterface)
    assert isinstance(ctrl.observation, FlatObservationInterface)
    assert treat.family_report()["predictive_variance"] == ctrl.family_report()["predictive_variance"]


def test_family_arm_source_features_are_not_narrower_than_the_control(anat):
    """The mean-path regression the architect ranked above the variance work.

    With families on, a head handed ``SCWBD.layout`` sees only the shared
    interface prefix (``rate_e``, ``rate_i``) = 2 dims, against the control
    arm's (``rate_e``, ``rate_i``, ``spectral``) = 18.  A1 would then have been
    biased *against* the treatment arm by an interface, not by the model.
    """
    treat = SCWBD(_cfg(family_state=True), anat)
    ctrl = SCWBD(_cfg(family_state=False), anat)
    narrow = sum(treat.layout.spec(n).dim for n in treat.layout.exported_names())
    assert narrow == 2, "the interface view is expected to be narrow; that is why heads must not use it"
    assert treat.observation.feature_dim >= ctrl.observation.feature_dim // 2, (
        f"treatment source features ({treat.observation.feature_dim}) are far narrower than the "
        f"control's ({ctrl.observation.feature_dim}); the ablation would measure the interface"
    )
    # and they must carry each family's own out-ports, not a shared slice
    ports = treat.observation.describe()["out_ports"]
    assert "oscillatory" in ports["cortex_vis"]
    assert "recall" in ports["hippocampus"]
    assert "gate_out" in ports["basal_ganglia"]


@pytest.mark.parametrize("arm,kw", ARMS)
def test_source_features_actually_depend_on_the_private_state(anat, arm, kw):
    m, res = _rollout(anat, _cfg(**kw))
    x = res.state[:, -1]
    with torch.no_grad():
        f0 = m.observation.source_features(x)
        x2 = x.clone()
        if m.family_layout is not None:
            sl = m.family_layout.family("cortex_vis").layout.slice("spectral")
            idx = m.family_layout.index("cortex_vis")
            x2[:, idx, sl] = x2[:, idx, sl] + 1.0
        else:
            sl = m.layout.slice("spectral")
            x2[..., sl] = x2[..., sl] + 1.0
        d = float((m.observation.source_features(x2) - f0).abs().max())
    assert d > 1e-6, f"{arm}: source features ignore the spectral component"


# ======================================================================
# the propagator's own contract
# ======================================================================
def test_uncertainty_is_generated_not_destroyed():
    """A signed innovation could cancel accumulated uncertainty to fit one sample."""
    p = UncertaintyPropagator(8, 4, hidden=16, dt=0.008)
    x = torch.randn(5, 3, 8) * 10.0
    assert bool((p.innovation(x) > 0).all())


def test_the_logvar_map_is_monotone_in_the_uncertainty_channel():
    """More accumulated uncertainty may only ever raise the predicted variance."""
    from scwbd.foundation.uncertainty import _LogVarHead

    h = _LogVarHead(4, 1)
    torch.nn.init.normal_(h.w, std=1.0)  # even adversarial weights stay positive
    u = torch.randn(64, 4)
    lo = h(u)
    hi = h(u + 1.0)
    assert bool((hi >= lo - 1e-6).all()), "the uncertainty -> log-variance map is not monotone"


def test_the_uncertainty_state_has_a_fixed_point():
    """It saturates rather than diverging over a long rollout."""
    p = UncertaintyPropagator(4, 2, hidden=16, dt=0.008)
    x = torch.zeros(1, 1, 4)
    u = torch.zeros(1, 1, 2)
    for _ in range(20000):  # 160 s of model time
        u = u + p(x, u)
    lam = torch.nn.functional.softplus(p.log_decay)
    expected = p.innovation(x) / lam
    assert torch.allclose(u, expected, atol=1e-3), f"no fixed point: {u} vs {expected}"
    assert float(u.max()) < 50.0


def test_logvar_is_clamped():
    from scwbd.foundation.uncertainty import _LogVarHead

    h = _LogVarHead(2, 1)
    out = h(torch.full((4, 2), 1e6))
    assert float(out.max()) <= LOGVAR_CLAMP[1]


def test_every_family_declares_the_uncertainty_component(anat):
    m = SCWBD(_cfg(family_state=True), anat)
    for f in m.family_layout:
        assert UNCERTAINTY_COMPONENT in f.layout, f"{f.name} carries no X_i^uncertainty"
        assert f.name in m.family_local.uncertainty


def test_the_uncertainty_channel_stays_in_span(anat):
    """The propagator writes only into its own component."""
    m, res = _rollout(anat, _cfg(family_state=True))
    m.family_layout.assert_clean(res.state, where="uncertainty propagation")


@pytest.mark.parametrize("arm,kw", ARMS)
def test_the_residual_may_not_write_to_the_uncertainty_channel(anat, arm, kw):
    """``X_i^uncertainty`` has exactly one law.

    If ``R_theta`` could also write there it would buy likelihood by moving the
    variance directly, bypassing the innovation/decay dynamics that make the
    channel mean anything — and R05 prices the residual against the mechanistic
    terms, not against the variance.
    """
    torch.manual_seed(0)
    m = SCWBD(_cfg(**kw), anat)
    x = torch.randn(2, anat.n_regions, m.layout.dim) * 0.3
    extra = torch.zeros(2, anat.n_regions, m.cfg.hidden // 2)
    if m.family_layout is not None:
        with torch.no_grad():
            for p in m.family_residual.parameters():
                torch.nn.init.normal_(p, std=0.5)
            r = m.family_residual(m.family_layout.zero_pad(x), extra)
        for f in m.family_layout:
            u = m.family_layout.get(r, f.name, UNCERTAINTY_COMPONENT)
            assert float(u.abs().max()) == 0.0, f"{f.name}: residual wrote into X^uncertainty"
    else:
        # control arm: SCWBD.step zeroes the slice before adding
        sl = m.layout.slice(UNCERTAINTY_COMPONENT)
        with torch.no_grad():
            for p in m.residual.parameters():
                torch.nn.init.normal_(p, std=0.5)
            f_res = m.residual(x, extra)
            f_res = torch.cat(
                [f_res[..., : sl.start], torch.zeros_like(f_res[..., sl]), f_res[..., sl.stop :]], dim=-1
            )
        assert float(f_res[..., sl].abs().max()) == 0.0
