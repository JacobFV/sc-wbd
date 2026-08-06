"""theta = (state, gain, synapse, structure): four clocks, and constrained rewriting."""

from __future__ import annotations

import pytest
import torch

from scwbd.dynamics import EdgeSet, GainController, StructuralRewriter, SynapticPlasticity, theta_field_policies
from scwbd.dynamics.coupling import EDGE_CLASS_CODES
from scwbd.dynamics.plasticity import THETA_CLOCKS, EditProposal, ResidualEvidence


def test_parameter_classes_have_strictly_slower_clocks():
    c = THETA_CLOCKS
    assert c["state"] < c["gain"] < c["synapse"] < c["structure"]
    policies = {p.name: p for p in theta_field_policies()}
    assert policies["theta_gain"].dt < policies["theta_synapse"].dt < policies["theta_structure"].dt
    assert policies["theta_structure"].kind == "event_driven", (
        "structural edits are event-driven proposals, never a scheduled gradient step"
    )


# ---------------------------------------------------------------------------
# theta^gain
# ---------------------------------------------------------------------------


def test_gain_controller_drives_activity_to_target(device):
    gc = GainController(6, batch=2, target=0.3, tau_gain=1.0, tau_avg=0.2, device=device)
    activity = torch.full((2, 6), 0.9, device=device)
    for _ in range(400):
        g = gc.update(activity * gc.gain / gc.gain.mean(), 0.01)
    assert float(gc.gain.mean()) < 1.0, "persistent over-activity must reduce gain"
    gc2 = GainController(6, batch=2, target=0.3, tau_gain=1.0, tau_avg=0.2, device=device)
    for _ in range(400):
        gc2.update(torch.full((2, 6), 0.05, device=device), 0.01)
    assert float(gc2.gain.mean()) > 1.0, "persistent under-activity must raise gain"


def test_gain_controller_respects_bounds(device):
    gc = GainController(4, target=0.5, tau_gain=0.01, g_min=0.5, g_max=2.0, device=device)
    for _ in range(1000):
        gc.update(torch.full((1, 4), 100.0, device=device), 0.01)
    assert float(gc.gain.min()) >= 0.5


# ---------------------------------------------------------------------------
# theta^synapse
# ---------------------------------------------------------------------------


def test_synaptic_plasticity_never_touches_hard_edges(device):
    edges = EdgeSet.random(20, density=0.3, seed=0, device=device)
    sp = SynapticPlasticity(edges, rule="hebb", eta=1.0, decay=0.0, batch=1)
    hard = edges.evidence == EDGE_CLASS_CODES["hard"]
    a = torch.rand(1, 20, device=device) + 0.5
    for _ in range(50):
        sp.update(a, 0.01)
    dev = sp.deviation()[0]
    assert float(dev[hard].abs().max()) == 0.0, "a hard-supported pathway's efficacy moved"
    assert float(dev[~hard].abs().max()) > 0.0, "soft/proposed efficacies failed to learn"


def test_synaptic_weights_stay_bounded_and_shrink_to_the_prior(device):
    edges = EdgeSet.random(16, density=0.3, seed=1, device=device)
    sp = SynapticPlasticity(edges, rule="hebb", eta=5.0, decay=0.0, w_max=2.0, batch=1)
    a = torch.ones(1, 16, device=device)
    for _ in range(200):
        sp.update(a, 0.05)
    assert float(sp.w.max()) <= 2.0
    # with no activity, shrinkage pulls the deviation back towards the prior
    sp2 = SynapticPlasticity(edges, rule="hebb", eta=0.0, decay=1.0, batch=1)
    sp2.w += 0.5
    before = float(sp2.deviation().abs().mean())
    for _ in range(50):
        sp2.update(torch.zeros(1, 16, device=device), 0.05)
    assert float(sp2.deviation().abs().mean()) < before


def test_bcm_rule_has_a_sliding_threshold(device):
    edges = EdgeSet.random(12, density=0.4, seed=2, device=device)
    sp = SynapticPlasticity(edges, rule="bcm", eta=0.5, decay=0.0, tau_theta=1.0, batch=1)
    high = torch.full((1, 12), 2.0, device=device)
    for _ in range(200):
        sp.update(high, 0.02)
    assert float(sp.theta_m.mean()) > 1.0, "the BCM threshold must slide with recent activity"


# ---------------------------------------------------------------------------
# theta^structure — constrained graph rewriting
# ---------------------------------------------------------------------------


def _evidence(device, n=8, magnitude=3.0, windows=10, uncertainty=1.0, where=(5, 2)):
    ev = None
    for _ in range(windows):
        r = torch.zeros(n, n, device=device)
        r[where] = magnitude
        ev = StructuralRewriter.accumulate_evidence(r, torch.full((n, n), uncertainty, device=device), ev)
    return ev


def test_no_proposal_when_the_residual_is_within_uncertainty(device):
    """The uncertainty gate is the whole point: a fit improvement is not anatomy."""
    n = 8
    edges = EdgeSet.random(n, density=0.2, seed=0, device=device)
    dist = torch.full((n, n), 40.0, device=device)
    ev = _evidence(device, n=n, magnitude=0.5, uncertainty=1.0)  # residual below uncertainty
    rw = StructuralRewriter()
    assert rw.propose(ev, edges, dist) == []


def test_persistent_residual_beyond_uncertainty_yields_a_proposal(device):
    n = 8
    edges = EdgeSet.random(n, density=0.05, seed=3, device=device)
    dist = torch.full((n, n), 40.0, device=device)
    ev = _evidence(device, n=n, magnitude=3.0, uncertainty=1.0, where=(5, 2))
    rw = StructuralRewriter(evidence_threshold=1.0, persistence_threshold=0.6)
    props = rw.propose(ev, edges, dist)
    assert props, "a persistent unexplained residual must produce a candidate edit"
    assert (props[0].dst, props[0].src) == (5, 2)
    assert props[0].persistence == pytest.approx(1.0)
    assert props[0].kind == "add"


def test_existing_and_over_long_edges_are_never_proposed(device):
    n = 8
    src = torch.tensor([2], device=device)
    dst = torch.tensor([5], device=device)
    edges = EdgeSet(
        src, dst, torch.ones(1, device=device), torch.full((1,), 40.0, device=device),
        torch.full((1,), EDGE_CLASS_CODES["hard"], dtype=torch.long, device=device), n,
    )
    dist = torch.full((n, n), 40.0, device=device)
    ev = _evidence(device, n=n, magnitude=3.0, where=(5, 2))
    assert StructuralRewriter().propose(ev, edges, dist) == [], "an existing edge was re-proposed"
    far = torch.full((n, n), 500.0, device=device)
    ev2 = _evidence(device, n=n, magnitude=3.0, where=(6, 1))
    assert StructuralRewriter(max_distance_mm=120.0).propose(ev2, edges, far) == []


def test_edit_is_rejected_when_it_degrades_prior_competencies(device):
    p = EditProposal("add", 2, 5, 40.0, 3.0, 1.0)
    rw = StructuralRewriter()
    dec = rw.evaluate(
        [p],
        replay=lambda _p: +0.5,  # replay loss got worse
        anatomical_logprob=lambda _p: -1.0,
        stability=lambda _p: 1.0,
    )[0]
    assert not dec.accepted
    assert any("prior competencies" in r for r in dec.reasons)


@pytest.mark.parametrize(
    "kwargs,expect",
    [
        (dict(stability=lambda p: -0.1), "destabilises"),
        (dict(anatomical_logprob=lambda p: -50.0), "implausible"),
    ],
)
def test_edit_rejection_reasons(kwargs, expect, device):
    p = EditProposal("add", 2, 5, 40.0, 3.0, 1.0)
    args = dict(replay=lambda _p: -0.5, anatomical_logprob=lambda _p: -1.0, stability=lambda _p: 1.0)
    args.update(kwargs)
    dec = StructuralRewriter().evaluate([p], **args)[0]
    assert not dec.accepted and any(expect in r for r in dec.reasons)


def test_energetic_cost_scales_with_wiring_length(device):
    long_edit = EditProposal("add", 0, 1, 100.0, 3.0, 1.0)
    rw = StructuralRewriter(energetic_cost_per_mm=0.02, energetic_budget=1.0)
    dec = rw.evaluate(
        [long_edit], replay=lambda p: -0.5, anatomical_logprob=lambda p: -1.0, stability=lambda p: 1.0
    )[0]
    assert not dec.accepted and any("energetic cost" in r for r in dec.reasons)
    assert dec.energetic_cost == pytest.approx(2.0)


def test_accepted_edit_enters_as_a_proposed_edge(device):
    """An edge admitted because a residual persisted is model evidence, not anatomy."""
    n = 8
    edges = EdgeSet.random(n, density=0.1, seed=4, device=device)
    before = edges.class_counts()
    p = EditProposal("add", 2, 5, 40.0, 3.0, 1.0)
    rw = StructuralRewriter()
    decs = rw.evaluate(
        [p], replay=lambda _p: -0.5, anatomical_logprob=lambda _p: -1.0, stability=lambda _p: 1.0
    )
    assert decs[0].accepted
    new = StructuralRewriter.apply(edges, decs)
    after = new.class_counts()
    assert new.n_edges == edges.n_edges + 1
    assert after["proposed"] == before["proposed"] + 1
    assert after["hard"] == before["hard"] and after["soft"] == before["soft"]
    assert "structural_rewrite" in new.provenance
    rep = rw.report()
    assert rep["n_accepted"] == 1 and rep["decisions"][0]["src"] == 2


def test_rewriter_report_records_every_criterion(device):
    p = EditProposal("add", 1, 3, 30.0, 2.0, 0.8)
    rw = StructuralRewriter()
    dec = rw.evaluate(
        [p], replay=lambda _p: -0.1, anatomical_logprob=lambda _p: -2.0, stability=lambda _p: 0.5
    )[0]
    d = dec.as_dict()
    for key in ("competency_delta", "energetic_cost", "anatomical_plausibility", "stability_margin"):
        assert key in d, f"{key} must be reported separately, never folded into one score"
