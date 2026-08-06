"""Thalamus, basal ganglia, cerebellum, and the neuromodulator semantics refusal."""

from __future__ import annotations

import pytest
import torch

from scwbd.dynamics import (
    BasalGangliaGate,
    Cerebellum,
    NeuromodulatorBank,
    NeuromodulatoryField,
    ReceptorSpec,
    SemanticCollapseError,
    ThalamicRelay,
)


# ---------------------------------------------------------------------------
# Thalamus
# ---------------------------------------------------------------------------


def test_hyperpolarisation_de_inactivates_the_t_current(device):
    """Burst readiness is a *state*, not a mode switch someone sets."""
    th = ThalamicRelay(4, device=device)
    x = th.init_state(2, device=device)
    dt = 1e-3
    hyper = torch.full((2, 4), -1.0, device=device)
    depol = torch.full((2, 4), 1.0, device=device)
    xh, xd = x.clone(), x.clone()
    for _ in range(500):
        xh = xh + dt * th.drift(xh, hyper)
        xd = xd + dt * th.drift(xd, depol)
    assert float(th.output(xh)["burst_readiness"].mean()) > 0.9
    assert float(th.output(xd)["burst_readiness"].mean()) < 0.1


def test_burst_state_amplifies_the_same_input(device):
    th = ThalamicRelay(3, burst_gain=3.0, device=device)
    dt = 1e-3
    primed = th.init_state(1, device=device)
    tonic = th.init_state(1, device=device)
    for _ in range(400):  # prime one with hyperpolarisation, keep the other depolarised
        primed = primed + dt * th.drift(primed, torch.full((1, 3), -1.0, device=device))
        tonic = tonic + dt * th.drift(tonic, torch.full((1, 3), 0.5, device=device))
    drive = torch.full((1, 3), 1.0, device=device)
    for _ in range(15):
        primed = primed + dt * th.drift(primed, drive)
        tonic = tonic + dt * th.drift(tonic, drive)
    assert float(th.output(primed)["burst"].mean()) > float(th.output(tonic)["burst"].mean())


def test_trn_inhibition_reduces_relay(device):
    th = ThalamicRelay(3, device=device)
    x = th.init_state(1, device=device)
    drive = torch.full((1, 3), 1.0, device=device)
    free = x + 1e-3 * th.drift(x, drive)
    inhibited = x + 1e-3 * th.drift(x, drive, trn=torch.full((1, 3), 0.8, device=device))
    assert float(inhibited[..., 0].sum()) < float(free[..., 0].sum())


def test_thalamic_routing_is_an_explicit_operator(device):
    th = ThalamicRelay(3, device=device)
    relay = torch.tensor([[1.0, 0.0, 0.0]], device=device)
    gain = torch.zeros(1, 3, 2, device=device)
    gain[0, 0, 1] = 1.0  # channel 0 routed to target 1 only
    out = th.route(relay, gain)
    assert torch.allclose(out, torch.tensor([[0.0, 1.0]], device=device))


# ---------------------------------------------------------------------------
# Basal ganglia
# ---------------------------------------------------------------------------


def test_dopamine_gain_opens_the_gate_through_d1_and_d2(device):
    bg = BasalGangliaGate(4, device=device)
    x = bg.init_state(3, device=device)
    cortex = torch.full((3, 4), 0.8, device=device)
    da = torch.tensor([[-0.5], [0.0], [0.8]], device=device)
    _, out = bg(x, cortex, dopamine_gain=da)
    gates = out["gate"].mean(dim=-1)
    assert float(gates[2]) > float(gates[1]) > float(gates[0]), f"gate not monotone in DA: {gates}"
    assert float(out["d1"][2].mean()) > float(out["d1"][0].mean())
    assert float(out["d2"][2].mean()) < float(out["d2"][0].mean()), "D2 must respond with opposite sign"


def test_hyperdirect_pathway_suppresses_the_gate(device):
    """Broad cortical drive recruits STN and closes the gate — a global stop."""
    weak = BasalGangliaGate(4, w_hyper=0.0, device=device)
    strong = BasalGangliaGate(4, w_hyper=3.0, device=device)
    cortex = torch.full((1, 4), 1.0, device=device)
    _, a = weak(weak.init_state(1, device=device), cortex, dopamine_gain=0.5)
    _, b = strong(strong.init_state(1, device=device), cortex, dopamine_gain=0.5)
    assert float(b["stn"].mean()) > float(a["stn"].mean())
    assert float(b["gate"].mean()) < float(a["gate"].mean())


def test_basal_ganglia_gate_is_channel_specific(device):
    bg = BasalGangliaGate(3, device=device)
    cortex = torch.tensor([[1.0, 0.1, 0.0]], device=device)
    _, out = bg(bg.init_state(1, device=device), cortex, dopamine_gain=0.5)
    g = out["gate"][0]
    assert float(g[0]) > float(g[1]) > float(g[2]), f"gating must be per channel: {g}"


def test_basal_ganglia_consumes_no_reward_signal(device):
    """Dopamine here is a receptor-typed gain, not a reward (thesis §5).

    Checked on the interface, not the prose: the gate takes cortical drive and a
    dopaminergic *gain*, and there is no reward, value or RPE input anywhere in
    its signature or its outputs.
    """
    import inspect

    params = set(inspect.signature(BasalGangliaGate.forward).parameters)
    assert "dopamine_gain" in params
    forbidden = {"reward", "value", "rpe", "reward_prediction_error", "utility"}
    assert not (forbidden & params)
    bg = BasalGangliaGate(2, device=device)
    _, out = bg(bg.init_state(1, device=device), torch.ones(1, 2, device=device))
    assert not (forbidden & set(out))


# ---------------------------------------------------------------------------
# Cerebellum
# ---------------------------------------------------------------------------


def test_cerebellum_learns_a_forward_model(device):
    torch.manual_seed(0)
    cb = Cerebellum(4, 2, n_granule=256, lr=0.05, error_delay=2, device=device)
    W = torch.randn(4, 2, device=device)
    errs = []
    for step in range(300):
        x = torch.randn(16, 4, device=device)
        target = torch.tanh(x @ W)
        e = cb.learn(x, target)
        if step > 5:
            errs.append(float(e.abs().mean()))
    assert errs[-1] < 0.5 * errs[0], f"cerebellar forward model did not learn: {errs[0]:.4f} -> {errs[-1]:.4f}"
    x = torch.randn(8, 4, device=device)
    pred = cb.predict(x)
    assert pred.shape == (8, 2) and torch.isfinite(pred).all()


def test_cerebellar_error_is_delayed(device):
    """The climbing-fibre error arrives late; the first steps produce no update."""
    cb = Cerebellum(3, 1, n_granule=32, error_delay=3, device=device)
    x = torch.randn(4, 3, device=device)
    for _ in range(3):
        e = cb.learn(x, torch.ones(4, 1, device=device))
        assert float(e.abs().max()) == 0.0
    e = cb.learn(x, torch.ones(4, 1, device=device))
    assert float(e.abs().max()) > 0.0
    assert cb.n_updates == 1


def test_cerebellum_states_its_epistemic_status():
    assert "mechanistic label" in Cerebellum.falsifier


# ---------------------------------------------------------------------------
# Neuromodulation — the §5 refusal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,label",
    [
        ("dopamine", "reward"),
        ("dopamine", "reward_prediction_error"),
        ("acetylcholine", "attention"),
        ("norepinephrine", "arousal"),
        ("serotonin", "punishment"),
    ],
)
def test_psychological_labels_on_neuromodulators_are_refused(name, label, device):
    r = ReceptorSpec("R1", ("all",), gain_effect=0.5)
    with pytest.raises(SemanticCollapseError, match="receptor-"):
        NeuromodulatoryField(name, 4, [r], semantics=label, device=device)


def test_receptor_typed_declaration_is_accepted(device):
    """The permitted description: receptor, target, timescale, state dependence."""
    field = NeuromodulatoryField(
        "dopamine",
        6,
        [
            ReceptorSpec("D1", ("0", "1"), gain_effect=+0.8, plasticity_effect=+1.0, tau=0.5),
            ReceptorSpec("D2", ("2", "3"), gain_effect=-0.5, plasticity_effect=+0.2, tau=2.0),
        ],
        device=device,
    )
    d = field.describe()
    assert d["n_receptors"] == 2
    assert {r["receptor"] for r in d["receptors"]} == {"D1", "D2"}
    assert d["receptors"][0]["tau_s"] != d["receptors"][1]["tau_s"]


def test_a_modulator_without_receptors_is_refused(device):
    with pytest.raises(ValueError, match="at least one ReceptorSpec"):
        NeuromodulatoryField("dopamine", 4, [], device=device)


def test_receptor_effects_are_target_and_sign_specific(device):
    field = NeuromodulatoryField(
        "dopamine",
        4,
        [
            ReceptorSpec("D1", ("0",), gain_effect=+1.0, tau=0.05),
            ReceptorSpec("D2", ("1",), gain_effect=-0.5, tau=0.05),
        ],
        device=device,
    )
    drive = torch.full((1, 4), 1.0, device=device)
    for _ in range(200):
        out = field.update(drive, 0.01)
    g = out["gain"][0]
    assert float(g[0]) > 1.0, "D1 target gain should increase"
    assert float(g[1]) < 1.0, "D2 target gain should decrease"
    assert float(g[2]) == pytest.approx(1.0), "non-target regions are unaffected"


def test_receptor_timescales_separate(device):
    fast = NeuromodulatoryField(
        "acetylcholine", 2, [ReceptorSpec("nAChR", ("all",), gain_effect=1.0, tau=0.05)], device=device
    )
    slow = NeuromodulatoryField(
        "acetylcholine", 2, [ReceptorSpec("M1", ("all",), gain_effect=1.0, tau=5.0)], device=device
    )
    drive = torch.ones(1, 2, device=device)
    for _ in range(20):
        f = fast.update(drive, 0.01)
        s = slow.update(drive, 0.01)
    assert float(f["gain"].mean()) > float(s["gain"].mean()), "timescales must actually differ"


def test_state_dependent_receptor_requires_activity(device):
    field = NeuromodulatoryField(
        "norepinephrine",
        3,
        [ReceptorSpec("alpha2A", ("all",), gain_effect=1.0, tau=0.05, state_dependence="activity")],
        device=device,
    )
    drive = torch.ones(1, 3, device=device)
    quiet = torch.zeros(1, 3, device=device)
    active = torch.ones(1, 3, device=device)
    for _ in range(100):
        gq = field.update(drive, 0.01, activity=quiet)["gain"].clone()
    field2 = NeuromodulatoryField(
        "norepinephrine",
        3,
        [ReceptorSpec("alpha2A", ("all",), gain_effect=1.0, tau=0.05, state_dependence="activity")],
        device=device,
    )
    for _ in range(100):
        ga = field2.update(drive, 0.01, activity=active)["gain"].clone()
    assert float(ga.mean()) > float(gq.mean())


def test_modulator_bank_composes_multiplicatively(device):
    a = NeuromodulatoryField("dopamine", 2, [ReceptorSpec("D1", ("all",), gain_effect=1.0, tau=0.01)], device=device)
    b = NeuromodulatoryField(
        "acetylcholine", 2, [ReceptorSpec("M1", ("all",), gain_effect=1.0, tau=0.01)], device=device
    )
    bank = NeuromodulatorBank([a, b])
    drive = {"dopamine": torch.ones(1, 2, device=device), "acetylcholine": torch.ones(1, 2, device=device)}
    for _ in range(200):
        bank.update(drive, 0.01)
    total = bank.total_gain()
    assert float(total.mean()) == pytest.approx(float(a.gain.mean()) * float(b.gain.mean()), rel=1e-4)
    assert float(total.mean()) > float(a.gain.mean()), "interaction is multiplicative, not additive"
