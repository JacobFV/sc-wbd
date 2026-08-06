"""Validation: reduced Wong–Wang FC structure onsets at the bifurcation in G.

The empirical result this reproduces (Deco et al. 2013, J Neurosci 33:11239;
Deco & Kringelbach 2014) is that a whole-brain reduced Wong–Wang model fits
empirical resting-state functional connectivity best when the global coupling
``G`` sits near the bifurcation of the system — not at weak coupling, and not
deep in the strongly-coupled regime.

What is validated here, honestly stated: with a synthetic connectome and no
empirical FC available in the dynamics module, this test validates the
*mechanism* behind that finding —

1. a bifurcation in ``G`` exists and is located from the mean-activity curve;
2. simulated FC carries **no** anatomical structure well below it
   (FC-SC correlation ~ 0): weak coupling cannot imprint the connectome;
3. FC-SC correlation **onsets sharply at the bifurcation** and stays elevated
   across the supracritical band;
4. it **degrades again** under strong over-coupling, as global synchronisation
   washes the topology out.

**Why the peak location is not asserted.**  An earlier version of (3) asserted
that ``argmax(FC-SC)`` lies near ``G_c``.  It does not survive a change of
floating-point backend: the supracritical FC-SC curve is a *plateau*, ~0.21–0.25
on CUDA and ~0.24–0.29 on CPU across G ∈ [0.8, 3.0], and the argmax over a flat
plateau is noise — it lands on G=2.0 on CUDA and G=4.0 on CPU from the same
seed.  The science does not actually claim a sharp optimum at ``G_c``; it claims
structure appears at the transition and is lost under over-coupling.  Both of
those are stable to ~13% margins on both backends, so those are what is
asserted, and :func:`test_the_supracritical_plateau_is_flat` pins the flatness
that makes the peak location unusable.  Widening the old window until it passed
would have turned a real result into a decorative one.

The full empirical claim — simulated vs *measured* FC across a G sweep on a real
connectome — is a G2 bench gate (agent J) with real data, not a unit test.  This
test is the model-side half of it, and it is written so that a failure is a
result about the model rather than a flaky assertion.

FC is computed on **simulated BOLD** through the Balloon–Windkessel model, as in
the literature, not on the raw synaptic gating variable.
"""

from __future__ import annotations

import pytest
import torch

from scwbd.dynamics import (
    BalloonWindkessel,
    DelayedConnectome,
    EdgeSet,
    ReducedWongWangSingle,
    SimConfig,
    WholeBrainSimulator,
)
from scwbd.dynamics.simulator import functional_connectivity


@pytest.fixture(scope="module")
def sweep(device):
    """One batched G-sweep: 14 coupling strengths integrated in parallel."""
    N = 40
    torch.manual_seed(0)
    pos = torch.randn(N, 3, device=device) * 30.0
    D = torch.cdist(pos, pos)
    W = torch.exp(-D / 40.0) * (torch.rand(N, N, device=device) < 0.4)
    W = (W + W.T) / 2
    W.fill_diagonal_(0.0)
    W = W / W.sum(dim=1, keepdim=True).mean()

    edges = EdgeSet.from_dense(W, D, evidence="hard", threshold=1e-6, device=device)
    be = ReducedWongWangSingle().to(device)
    sim = WholeBrainSimulator(be, DelayedConnectome(edges, n_channels=1))

    Gs = torch.tensor(
        [0.0, 0.2, 0.4, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=device
    ).reshape(-1, 1)
    theta = be.make_theta(Gs.shape[0], N, device=device, G=Gs, sigma=0.01, velocity=1e6)
    res = sim.run(
        theta, SimConfig(dt=1e-3, n_steps=45000, seed=0, warmup_steps=15000, record_every=10)
    )
    a = res.activity()  # (T, B, N), 10 ms sampling

    bw = BalloonWindkessel()
    bold, _ = bw.rollout(a, bw.make_theta(Gs.shape[0], N, device=device), dt=0.01)
    bold = bold[1500:][::20]  # drop the hemodynamic transient; TR = 0.2 s

    fc = functional_connectivity(bold)
    mask = ~torch.eye(N, dtype=torch.bool, device=device)
    off = fc[:, mask]
    sc = W[mask]
    fc_sc = torch.stack(
        [torch.corrcoef(torch.stack([off[i], sc]))[0, 1] for i in range(Gs.shape[0])]
    )
    mean_S = a.mean(dim=(0, 2))
    return {
        "G": Gs.flatten(),
        "mean_S": mean_S,
        "fc_sc": fc_sc,
        "fc_mean": off.mean(dim=1),
        "W": W,
    }


def test_bifurcation_in_global_coupling_exists(sweep):
    """The mean-activity curve has a sharp transition — the working-point bifurcation."""
    G, S = sweep["G"], sweep["mean_S"]
    dS = (S[1:] - S[:-1]) / (G[1:] - G[:-1])
    i = int(dS.argmax())
    Gc = float(G[i + 1])
    assert 0.5 < Gc < 1.5, f"bifurcation located at G={Gc:.2f}, outside the expected window"
    # the transition is sharp: the largest slope dominates the rest
    assert float(dS[i]) > 5 * float(dS[dS != dS.max()].abs().max())
    # below the transition the network stays in the low-activity branch
    assert float(S[0]) < 0.1 and float(S[-1]) > 0.5


def test_weak_coupling_imprints_no_anatomy(sweep):
    """Well below the bifurcation, simulated FC carries no connectome structure."""
    G, fc_sc = sweep["G"], sweep["fc_sc"]
    weak = fc_sc[G < 0.5]
    assert float(weak.abs().max()) < 0.12, f"unexpected FC-SC structure at weak coupling: {weak}"


def _bifurcation_G(sweep) -> float:
    G, S = sweep["G"], sweep["mean_S"]
    dS = (S[1:] - S[:-1]) / (G[1:] - G[:-1])
    return float(G[int(dS.argmax()) + 1])


def _bands(sweep):
    """Sub-critical, supracritical-plateau and over-coupled slices of FC-SC.

    The plateau stops at G=3: beyond that the over-coupling regime begins, and
    mixing the two would let a degrading tail mask the plateau level.
    """
    G, fc_sc = sweep["G"], sweep["fc_sc"]
    Gc = _bifurcation_G(sweep)
    # "Well below" means well below: the last couple of points before the
    # transition are already inside the critical neighbourhood (FC-SC is visibly
    # rising at G=0.70 while mean activity is still on the low branch), so
    # including them would compare the onset against itself.
    sub = fc_sc[G < 0.8 * Gc]
    plateau = fc_sc[(G >= Gc) & (G <= 3.0)]
    return Gc, sub, plateau


def test_fc_sc_structure_onsets_at_the_bifurcation(sweep):
    """Anatomical structure appears in FC *at* the transition, not before it.

    Asserted on the onset step and the plateau level — both stable across
    floating-point backends — rather than on ``argmax``, which is not (see the
    module docstring).
    """
    G, fc_sc = sweep["G"], sweep["fc_sc"]
    Gc, sub, plateau = _bands(sweep)
    at_Gc = float(fc_sc[G == Gc][0])

    # below the transition, essentially nothing
    assert float(sub.abs().max()) < 0.15, f"unexpected sub-critical FC-SC structure: {sub}"
    # at the transition, structure is present
    assert at_Gc > 0.18, f"no FC-SC structure at the bifurcation G_c={Gc:.2f}: r={at_Gc:.3f}"
    # and the onset is a clear step above the sub-critical band, not a drift
    assert at_Gc > float(sub.abs().max()) + 0.08, (
        f"FC-SC at G_c ({at_Gc:.3f}) is not clearly above the sub-critical band "
        f"({float(sub.abs().max()):.3f})"
    )
    # the whole supracritical band stays elevated: this is a plateau, not a spike
    assert float(plateau.min()) > 0.15, f"supracritical FC-SC not sustained: {plateau}"


def test_the_supracritical_plateau_is_flat(sweep):
    """Pin the flatness that makes the peak *location* an unusable statistic.

    This is the positive form of the caveat in the module docstring: if the
    plateau ever became sharply peaked, asserting only onset and degradation
    would be leaving a real result untested, and this test would fail to remind
    us.
    """
    _, _, plateau = _bands(sweep)
    spread = float(plateau.max() - plateau.min())
    level = float(plateau.mean())
    assert spread < 0.4 * level, (
        f"supracritical FC-SC is not flat (spread {spread:.3f} vs level {level:.3f}); "
        "if it has become genuinely peaked, assert the peak location again"
    )


def test_over_coupling_degrades_the_fc_match(sweep):
    """Strong over-coupling synchronises the network and washes the topology out.

    Compared against the plateau *mean* rather than its max: the max is one
    sample of a flat, noisy plateau, so a ratio against it inherits that noise.
    The measured drop is ~13% of the plateau level on both backends.
    """
    _, _, plateau = _bands(sweep)
    level = float(plateau.mean())
    strong = float(sweep["fc_sc"][-1])  # G = 6, well past the bifurcation
    assert strong < 0.95 * level, (
        f"FC-SC at the strongest coupling ({strong:.3f}) should fall below the "
        f"supracritical plateau level ({level:.3f})"
    )
    assert float(sweep["fc_mean"][-1]) > 0.9, "the strongly-coupled regime should be globally synchronised"


def test_fc_mean_is_monotone_a_control_for_the_structure_claim(sweep):
    """Control: mean FC rises monotonically with G.

    Without this control, the rise in FC-*structure* at ``G_c`` could be
    dismissed as an artefact of "more coupling = more correlation".  It is not:
    overall correlation keeps climbing to ~0.999 while the *anatomical* structure
    onsets, plateaus and then falls away.  This control is what makes the
    degradation at G=6 evidence rather than a plot.
    """
    fc_mean = sweep["fc_mean"]
    supra = fc_mean[6:]  # from the bifurcation upwards
    assert bool((supra[1:] >= supra[:-1] - 0.02).all()), f"mean FC not monotone above G_c: {supra}"
