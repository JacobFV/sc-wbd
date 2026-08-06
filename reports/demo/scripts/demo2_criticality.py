"""Demo 2 - regenerate the reduced Wong-Wang criticality sweep from scwbd.dynamics.

Faithful reproduction of the module-scoped `sweep` fixture in
tests/dynamics/test_wong_wang_criticality.py (same seed, same grid, same
N=40 synthetic connectome, FC on simulated BOLD through Balloon-Windkessel).

Run with cwd = main repo. Writes scratchpad/part_demo2.json.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
import time

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

OUT = pathlib.Path(sys.argv[1])
DEVICE = torch.device(sys.argv[2] if len(sys.argv) > 2 else "cuda")
REPO = pathlib.Path(__file__).resolve()

if DEVICE.type == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.20, DEVICE.index or 0)

t_start = time.time()

# ---- the fixture, verbatim -------------------------------------------------
N = 40
torch.manual_seed(0)
pos = torch.randn(N, 3, device=DEVICE) * 30.0
D = torch.cdist(pos, pos)
W = torch.exp(-D / 40.0) * (torch.rand(N, N, device=DEVICE) < 0.4)
W = (W + W.T) / 2
W.fill_diagonal_(0.0)
W = W / W.sum(dim=1, keepdim=True).mean()

edges = EdgeSet.from_dense(W, D, evidence="hard", threshold=1e-6, device=DEVICE)
be = ReducedWongWangSingle().to(DEVICE)
sim = WholeBrainSimulator(be, DelayedConnectome(edges, n_channels=1))

Gs = torch.tensor(
    [0.0, 0.2, 0.4, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=DEVICE
).reshape(-1, 1)
theta = be.make_theta(Gs.shape[0], N, device=DEVICE, G=Gs, sigma=0.01, velocity=1e6)
res = sim.run(
    theta, SimConfig(dt=1e-3, n_steps=45000, seed=0, warmup_steps=15000, record_every=10)
)
a = res.activity()  # (T, B, N), 10 ms sampling

bw = BalloonWindkessel()
bold, _ = bw.rollout(a, bw.make_theta(Gs.shape[0], N, device=DEVICE), dt=0.01)
bold = bold[1500:][::20]  # drop the hemodynamic transient; TR = 0.2 s

fc = functional_connectivity(bold)
mask = ~torch.eye(N, dtype=torch.bool, device=DEVICE)
off = fc[:, mask]
sc = W[mask]
fc_sc = torch.stack(
    [torch.corrcoef(torch.stack([off[i], sc]))[0, 1] for i in range(Gs.shape[0])]
)
mean_S = a.mean(dim=(0, 2))
fc_mean = off.mean(dim=1)
# ---- end fixture -----------------------------------------------------------

wall = time.time() - t_start

G = Gs.flatten()
dS = (S1 := mean_S)[1:] - mean_S[:-1]
dS = dS / (G[1:] - G[:-1])
i_bif = int(dS.argmax())
Gc = float(G[i_bif + 1])

g = [float(v) for v in G]
fs = [float(v) for v in fc_sc]
fm = [float(v) for v in fc_mean]
ms = [float(v) for v in mean_S]

i_peak = int(max(range(len(fs)), key=lambda k: fs[k]))
i_gc = g.index(Gc)

# the flatness that makes the peak location unusable, as the test itself states
plateau = [fs[k] for k in range(len(g)) if g[k] >= Gc and g[k] <= 3.0]

block = {
    "title": "Criticality: FC-SC correlation onsets at the bifurcation of global coupling G",
    "status": "regenerated",
    "what_it_is": (
        "Reduced Wong-Wang mean-field on a 40-region SYNTHETIC distance-dependent "
        "connectome. FC is computed on simulated BOLD through Balloon-Windkessel "
        "(TR 0.2 s), not on the raw gating variable. Reproduces the mechanism behind "
        "Deco et al. 2013 (J Neurosci 33:11239); it is NOT a fit to measured FC."
    ),
    "model_used": "scwbd.dynamics.ReducedWongWangSingle - NOT the trained SC-WBD-001-beta checkpoint",
    "series": {
        "G": {"units": "dimensionless (global coupling)", "values": g},
        "fc_sc_correlation": {
            "units": "Pearson r between off-diagonal simulated FC and synthetic SC",
            "values": fs,
        },
        "fc_mean": {
            "units": "mean off-diagonal simulated FC (Pearson r)",
            "values": fm,
            "role": "control - rules out 'more coupling = more correlation'",
        },
        "mean_activity_S": {
            "units": "dimensionless (mean synaptic gating variable)",
            "values": ms,
        },
        "dS_dG": {
            "units": "dimensionless per unit G",
            "values": [float(v) for v in dS],
            "note": "finite difference; G_c is the RIGHT endpoint of the steepest interval",
        },
    },
    "derived": {
        "bifurcation_G_c": Gc,
        "fc_sc_at_G_c": fs[i_gc],
        "fc_sc_max": fs[i_peak],
        "fc_sc_argmax_G": g[i_peak],
        "fc_sc_at_G_max_grid": fs[-1],
        "G_max_grid": g[-1],
        "fc_mean_final": fm[-1],
        "fc_mean_monotone_from_index_6_tol_0.02": all(
            fm[k + 1] >= fm[k] - 0.02 for k in range(6, len(fm) - 1)
        ),
        "weak_coupling_max_abs_fc_sc_G_lt_0.5": max(
            abs(fs[k]) for k in range(len(g)) if g[k] < 0.5
        ),
        "supracritical_plateau_G_c_to_3": {
            "values": plateau,
            "min": min(plateau),
            "max": max(plateau),
            "spread": max(plateau) - min(plateau),
        },
    },
    "caveat_from_the_test_itself": (
        "The location of the FC-SC peak is NOT a claim. The supracritical curve is a "
        "flat plateau and its argmax is not stable across floating-point backends: the "
        "test docstring records ~0.21-0.25 on CUDA and ~0.24-0.29 on CPU for G in "
        "[0.8, 3.0], with argmax landing on G=2.0 on CUDA and G=4.0 on CPU from the "
        "same seed. What is asserted is that structure APPEARS at the transition and is "
        "LOST under over-coupling. Also: the synthetic connectome W is drawn on-device, "
        "so CPU and CUDA runs generate DIFFERENT connectomes and therefore different "
        "fc_sc values."
    ),
    "assertions_reproduced": {
        "bifurcation_in_window_0.5_to_1.5": 0.5 < Gc < 1.5,
        "transition_is_sharp_slope_dominates_5x": float(dS[i_bif])
        > 5 * float(dS[dS != dS.max()].abs().max()),
        "low_branch_below_and_high_branch_above": float(mean_S[0]) < 0.1
        and float(mean_S[-1]) > 0.5,
        "weak_coupling_imprints_no_anatomy_lt_0.12": max(
            abs(fs[k]) for k in range(len(g)) if g[k] < 0.5
        )
        < 0.12,
        "fc_sc_peak_exceeds_0.2": max(fs) > 0.2,
        "over_coupling_degrades_fc_match": fs[-1] < 0.9 * max(fs),
        "fc_mean_final_exceeds_0.9": fm[-1] > 0.9,
    },
    "provenance": {
        "produced_by": "scratchpad/demo2_criticality.py - verbatim reproduction of the "
        "module-scoped `sweep` fixture in tests/dynamics/test_wong_wang_criticality.py",
        "source_test": "tests/dynamics/test_wong_wang_criticality.py",
        "source_test_sha256": hashlib.sha256(
            pathlib.Path("tests/dynamics/test_wong_wang_criticality.py").read_bytes()
        ).hexdigest(),
        "device": str(DEVICE),
        "device_name": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "seed": 0,
        "n_regions": N,
        "connectome": "synthetic distance-dependent, generated in-script from torch.manual_seed(0)",
        "sim_config": {
            "dt_s": 1e-3,
            "n_steps": 45000,
            "warmup_steps": 15000,
            "record_every": 10,
            "method": "heun",
            "sigma": 0.01,
            "velocity_m_per_s": 1e6,
        },
        "checkpoint": None,
        "loaded_key_count": None,
        "checkpoint_note": "no trained checkpoint is involved in this demo",
        "repo_git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "wall_seconds": round(wall, 2),
    },
}

OUT.write_text(json.dumps(block, indent=1))
print("wrote", OUT, f"({wall:.1f}s)")
print("G_c =", Gc, " fc_sc@G_c =", round(fs[i_gc], 4))
print("fc_sc max =", round(max(fs), 4), "at G =", g[i_peak])
print("fc_sc @G=6 =", round(fs[-1], 4), " fc_mean final =", round(fm[-1], 6))
print("plateau spread:", round(min(plateau), 4), "-", round(max(plateau), 4))
for k in range(len(g)):
    print(f"  G={g[k]:<5} fc_sc={fs[k]: .4f}  fc_mean={fm[k]: .6f}  S={ms[k]: .4f}")
print("assertions:", json.dumps(block["assertions_reproduced"]))
