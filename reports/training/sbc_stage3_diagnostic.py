"""Stage III SBC diagnostic -- pre-committed in sbc_stage3_precommitment.md.

NOT the preregistered SBC. That one runs on the Stage V checkpoint and is the
only one 🛡️ Popper adjudicates. This is a mid-run diagnostic whose result is
pre-committed to change nothing.

Posterior-only: builds the AmortizedPosterior alone (no SCWBD rollout, no
lead field, no anatomy) and runs it on CPU, so it does not contend with the
live training job for the unified memory pool.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from scwbd.foundation.checkpoint import load_checkpoint
from scwbd.foundation.config import load_config
from scwbd.foundation.posterior import AmortizedPosterior, posterior_report
from scwbd.foundation.simulate import THETA_NAMES, SimCorpus, ThetaPrior


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/scwbd_001_beta.yaml")
    ap.add_argument("--ckpt", default="checkpoints/scwbd-001-beta/stage_III_sliced.pt")
    ap.add_argument("--out", default="reports/training/sbc_stage3_diagnostic.json")
    ap.add_argument("--n-datasets", type=int, default=512, help="0 = use every val window")
    ap.add_argument("--n-samples", type=int, default=256)
    ap.add_argument(
        "--order",
        choices=("sequential", "shuffled"),
        default="sequential",
        help=(
            "'sequential' takes the FIRST n windows and is backend-biased -- the "
            "first 512 of 1888 contain zero samples from backends 0 and 1. Use "
            "'shuffled' (seeded) or --n-datasets 0 for a representative sample."
        ),
    )
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    cfg = load_config(args.config)
    d = cfg.data

    prior = ThetaPrior()
    posterior = AmortizedPosterior(
        cfg.posterior,
        len(THETA_NAMES),
        prior=prior,
        fs=d.fs_hz,
        nuisance_dim=cfg.posterior.nuisance_dim,
    )
    payload = load_checkpoint(args.ckpt, posterior=posterior, map_location="cpu", restore_rng=False)
    posterior.eval()

    # Same val split the trainer holds out: same index, window, fraction, seed.
    val = SimCorpus(
        Path(d.sim_index_fast),
        window=d.window + d.context,
        trajectory_subset="val",
        val_fraction=d.val_fraction,
        seed=d.seed,
    )
    want = len(val) if args.n_datasets <= 0 else min(args.n_datasets, len(val))
    if args.order == "shuffled":
        idx = torch.randperm(len(val), generator=torch.Generator().manual_seed(0))[:want].tolist()
    else:
        idx = list(range(want))
    subset = torch.utils.data.Subset(val, idx)
    loader = torch.utils.data.DataLoader(subset, batch_size=64, shuffle=False, num_workers=2)
    ys, ths, bks = [], [], []
    for b in loader:
        ys.append(b["activity"][:, : d.context])
        ths.append(b["theta"])
        bks.append(b["backend"])
    y = torch.cat(ys)
    th = torch.cat(ths)
    backends = torch.cat(bks)

    rep = posterior_report(posterior, y, th, param_names=THETA_NAMES, n_samples=args.n_samples)

    ranks = np.asarray(rep["sbc_ranks"], dtype=float)
    rep["label"] = "sbc_stage3_diagnostic"
    rep["is_preregistered_sbc"] = False
    rep["precommitment"] = "reports/training/sbc_stage3_precommitment.md"
    rep["not_the_verdict"] = (
        "Mid-run diagnostic. The preregistered SBC is the final one on the Stage V "
        "checkpoint; this result must never be substituted for or aggregated with it."
    )
    rep["checkpoint"] = {
        "path": args.ckpt,
        "step": payload.get("step"),
        "stage": payload.get("stage"),
        "git_sha": payload.get("git_sha"),
        "config_sha": payload.get("config_sha"),
    }
    rep["n_datasets"] = int(ranks.shape[0])
    rep["val_windows_available"] = len(val)
    rep["order"] = args.order
    rep["backend_counts"] = torch.bincount(backends, minlength=5).tolist()
    rep["diagnostic_sha"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()

    # Shape summary per parameter: mean rank (bias) and the U-statistic
    # (fraction of ranks in the outer 10% of the range; 0.10 under uniformity,
    # >0.10 U-shaped/over-confident, <0.10 inverted-U/under-confident).
    L = float(rep["sbc_n_bins"])
    shapes = []
    for j in range(ranks.shape[1]):
        r = ranks[:, j] / L
        edge = float(((r < 0.05) | (r > 0.95)).mean())
        shapes.append(
            {
                "param": rep["param_names"][j],
                "mean_rank": float(r.mean()),  # 0.5 under uniformity
                "edge_mass": edge,  # 0.10 under uniformity
                "ks_p": float(rep["sbc_ks_pvalue"][j]),
            }
        )
    rep["shape_summary"] = shapes

    Path(args.out).write_text(json.dumps(rep, indent=2))

    print(f"n_datasets={rep['n_datasets']}  n_bins={int(L)}  ckpt_step={payload.get('step')}")
    print(f"{'param':<22}{'mean_rank':>11}{'edge_mass':>11}{'KS p':>11}")
    for s in shapes:
        print(f"{s['param']:<22}{s['mean_rank']:>11.3f}{s['edge_mass']:>11.3f}{s['ks_p']:>11.4f}")
    print(f"\nmin KS p = {rep['sbc_ks_pvalue_min']:.4g}   coverage MAE = {rep['coverage_mae']:.4f}")
    print("uniform reference: mean_rank 0.500, edge_mass 0.100")


if __name__ == "__main__":
    main()
