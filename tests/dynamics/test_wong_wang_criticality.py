"""Validation: reduced Wong–Wang FC structure peaks near the bifurcation in G.

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
3. FC-SC correlation rises sharply at the bifurcation and attains its maximum
   in a neighbourhood of it;
4. it **degrades again** under strong over-coupling, as global synchronisation
   washes the topology out.

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


def test_fc_sc_correlation_peaks_near_the_bifurcation(sweep):
    """The known result: the FC match is best near the critical point, not at the extremes."""
    G, S, fc_sc = sweep["G"], sweep["mean_S"], sweep["fc_sc"]
    dS = (S[1:] - S[:-1]) / (G[1:] - G[:-1])
    Gc = float(G[int(dS.argmax()) + 1])
    best_G = float(G[int(fc_sc.argmax())])
    assert 0.7 * Gc <= best_G <= 4.0 * Gc, (
        f"FC-SC correlation peaks at G={best_G:.2f}, far from the bifurcation G_c={Gc:.2f}"
    )
    assert float(fc_sc.max()) > 0.2, "no anatomical structure appears in FC at any coupling"


def test_over_coupling_degrades_the_fc_match(sweep):
    """Strong over-coupling synchronises the network and washes the topology out."""
    G, fc_sc = sweep["G"], sweep["fc_sc"]
    peak = float(fc_sc.max())
    strong = float(fc_sc[-1])  # G = 6, well past the bifurcation
    assert strong < 0.9 * peak, (
        f"FC-SC at the strongest coupling ({strong:.3f}) should fall below the peak ({peak:.3f})"
    )
    assert float(sweep["fc_mean"][-1]) > 0.9, "the strongly-coupled regime should be globally synchronised"


def test_fc_mean_is_monotone_a_control_for_the_peak(sweep):
    """Control: mean FC rises monotonically with G.

    Without this control, a peak in FC-*structure* could be dismissed as an
    artefact of "more coupling = more correlation".  It is not: overall
    correlation keeps rising while the *anatomical* structure peaks and falls.
    """
    fc_mean = sweep["fc_mean"]
    supra = fc_mean[6:]  # from the bifurcation upwards
    assert bool((supra[1:] >= supra[:-1] - 0.02).all()), f"mean FC not monotone above G_c: {supra}"
