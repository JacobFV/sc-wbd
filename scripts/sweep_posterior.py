"""Pick the posterior LR by measurement, and say whether ISSUE-012's fix works.

ISSUE-012's discharge condition is `posterior_r2` on `log_G` above **0.4** on
held-out simulated datasets, with `sbc_ks_pvalue_min > 0.01` and
`coverage_mae < 0.05` still holding. The floor is 0.4 because a ridge probe on
the same conditioning already reaches 0.439 and an MLP reaches 0.753: the
amortised machinery has to beat a two-layer probe to justify the 1,111,568
parameters it costs.

The issue also says the retrain is **one stage, no whole-model run needed**.
That is what this script is. It trains the posterior alone -- summary encoder
and flow, nothing else -- against the same objective, the same 35%-masked
context window and the same corpus that `sim_losses` uses, across a grid of
(cond_norm, lr). A full run is ~38 h; this is minutes per cell, and it answers
the only question that decides whether those 38 h are worth spending.

What it deliberately does NOT do:

  * touch the dynamics model. The posterior's gradient path in `sim_losses` is
    `posterior.loss(y_summary, theta)` and nothing else reaches it, so training
    it alone is the same objective, not an approximation of it.
  * report a winner on training data. Every number is on trajectories held out
    by `trajectory_subset`, which splits on the trajectory because a simulated
    replica is not an independent subject.
  * pick the best cell by `npe_loss`. That is the metric ISSUE-012 showed can
    fall 10.3 -> -5.6 while carrying 0.09 nats about theta. The selection metric
    is R^2 on held-out theta; `-log q` is recorded beside it as a diagnostic.

Usage:

    python scripts/sweep_posterior.py --steps 1500 --out reports/run4_posterior
    python scripts/sweep_posterior.py --quick        # 300 steps, smoke the plumbing
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scwbd.foundation.config import load_config
from scwbd.foundation.posterior import AmortizedPosterior, posterior_report
# THETA_NAMES from `foundation.simulate`, NOT `infer.linear_gaussian` -- both
# export that name and they are different parameter sets (ARCHITECTURE.md O-3).
# The linear-Gaussian one is (a21, a32, a13, tau) in a surrogate that imports
# nothing from `foundation`; using it here would score the flow against labels
# it was never trained on and report a clean zero.
from scwbd.foundation.simulate import THETA_NAMES, SimCorpus, ThetaPrior

REPO = Path(__file__).resolve().parents[1]

#: ISSUE-012's discharge floor for `log_G`, and the two calibration conditions
#: that must survive it. Recovering calibration by widening back to the prior
#: does not discharge the issue, so R2 and calibration are checked together.
DISCHARGE = {"log_G_r2": 0.4, "sbc_ks_pvalue_min": 0.01, "coverage_mae": 0.05}


def _slice_mask(B: int, N: int, device, *, p_observed: float = 0.65, generator=None) -> torch.Tensor:
    """The observed subgraph, identical to `FoundationTrainer._slice_mask`.

    Duplicated rather than imported because importing it would mean constructing
    a `FoundationTrainer`, which builds the 26.3M-parameter dynamics model and
    the anatomy prior for a mask. Kept honest by
    `tests/foundation/test_sweep_matches_the_trainer.py`.
    """
    u = torch.rand(B, N, device=device, generator=generator)
    keep = (u < p_observed).float()
    empty = keep.sum(1) == 0
    if empty.any():
        keep[empty, 0] = 1.0
    return keep


def _loader(cfg, subset: str, *, batch: int, seed: int, workers: int) -> torch.utils.data.DataLoader:
    ds = SimCorpus(
        REPO / cfg.data.sim_index_fast,
        window=cfg.data.window,
        trajectory_subset=subset,
        val_fraction=0.1,
        seed=seed,
    )
    if len(ds) == 0:
        raise SystemExit(
            f"SimCorpus({cfg.data.sim_index_fast!r}, subset={subset!r}) is EMPTY. An unmatched "
            "glob is an empty permission set, not an error -- this repo lost 88.8% of run 2's "
            "parameters to that. Refusing to report a sweep over no data."
        )
    # The VALIDATION loader is shuffled too, at a fixed seed.
    #
    # It was `shuffle=(subset == "train")`, so the eval set was the first 512
    # windows in file order -- which is the first shard or two, i.e. one or two
    # simulator backends. `evaluate.posterior_calibration` draws its 512
    # BACKEND-STRATIFIED, and the difference is not academic: reproducing run 3's
    # setting gave `posterior_r2` and `posterior_sd_over_prior_sd` that match its
    # published values (+0.001 vs -0.010; 1.031 vs 1.024) while
    # `sbc_ks_pvalue_min` came out 0.000 against a published 0.0976. Two of three
    # statistics agreeing pins the disagreement to the eval POPULATION rather
    # than to the posterior, and SBC ranks are far more sensitive to a
    # homogeneous sample than R^2 is.
    #
    # Shuffling is not full stratification -- it does not guarantee the backend
    # proportions production uses -- so the sweep's calibration column is still
    # only comparable to itself. That is stated in the report rather than
    # papered over.
    g = torch.Generator().manual_seed(seed if subset == "train" else seed + 977)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch,
        shuffle=True,
        generator=g,
        num_workers=workers,
        drop_last=True,
        persistent_workers=workers > 0,
    )


#: SBC bins used by `evaluate.posterior_calibration`. The sweep MUST match it.
#:
#: The first version of this script used 128 and applied the training slice mask
#: at evaluation. Both differ from production, and the difference is not
#: cosmetic: reproducing run 3's exact setting (layer_v1 at 4e-6) returned
#: `sbc_ks_pvalue_min` 0.000 against the 0.0976 in
#: `reports/training/evaluation_run3.json`. R^2 agreed to within 0.01 and
#: calibration did not, so the calibration column was measuring the harness.
#: A sweep whose numbers cannot be compared to the run they are meant to inform
#: is an instrument that reports confidently and knows nothing.
PRODUCTION_SBC_BINS = 256


def _context(act: torch.Tensor, cfg, device, *, mask: bool) -> torch.Tensor:
    """The context window the posterior is conditioned on.

    ``mask=True`` reproduces `sim_losses`, which shows the posterior only the
    observed subgraph. ``mask=False`` reproduces `evaluate.posterior_calibration`,
    which passes `b["activity"][:, :context]` unmasked.

    The two differ, ISSUE-012 measured that the difference does not move R^2
    (the sd ratios agree to within 0.005 across all four combinations of split
    and mask), and it says nothing about SBC ranks. Training uses the masked
    form because that is what training does; evaluation uses the unmasked form
    because that is what the published numbers are.
    """
    act = act.to(device, non_blocking=True)
    ctx = act[:, : cfg.data.context]
    if not mask:
        return ctx
    B, _, N = ctx.shape
    m = _slice_mask(B, N, device).unsqueeze(1).expand_as(ctx)
    return ctx * m


def run_cell(
    cfg,
    *,
    cond_norm: str,
    lr: float,
    steps: int,
    train_dl,
    val_dl,
    device: str,
    seed: int,
    weight_decay: float,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    pcfg = type(cfg.posterior)(**{**vars(cfg.posterior), "cond_norm": cond_norm})
    prior = ThetaPrior()
    post = AmortizedPosterior(
        pcfg, len(THETA_NAMES), prior=prior, fs=cfg.data.fs_hz, nuisance_dim=pcfg.nuisance_dim
    ).to(device)
    n_par = sum(p.numel() for p in post.parameters())
    opt = torch.optim.AdamW(post.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)

    post.train()
    t0 = time.time()
    hist: list[float] = []
    it = iter(train_dl)
    for step in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_dl)
            batch = next(it)
        y = _context(batch["activity"], cfg, device, mask=True)
        theta = batch["theta"].to(device, non_blocking=True)
        loss = post.loss(y, theta)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(post.parameters(), 1.0)
        opt.step()
        sched.step()
        hist.append(float(loss.detach()))
        if step % 200 == 0:
            print(f"      step {step:5d}  -log q {np.mean(hist[-50:]):8.3f}", flush=True)

    # -- held-out evaluation ------------------------------------------------
    post.eval()
    ys, ths = [], []
    with torch.no_grad():
        for batch in val_dl:
            ys.append(_context(batch["activity"], cfg, device, mask=False))
            ths.append(batch["theta"].to(device))
            if sum(t.shape[0] for t in ths) >= 512:
                break
    y_val = torch.cat(ys)[:512]
    th_val = torch.cat(ths)[:512]

    with torch.no_grad():
        rep = posterior_report(
            post, y_val, th_val, param_names=THETA_NAMES, n_samples=PRODUCTION_SBC_BINS
        )
        held = float(post.loss(y_val, th_val))

    r2 = dict(zip(rep["param_names"], rep["posterior_r2"]))
    return {
        "cond_norm": cond_norm,
        "lr": lr,
        "steps": steps,
        "weight_decay": weight_decay,
        "seed": seed,
        "n_posterior_parameters": n_par,
        "posterior_r2": r2,
        "log_G_r2": r2.get("log_G"),
        "held_out_neg_log_q": held,
        "train_neg_log_q_last50": float(np.mean(hist[-50:])),
        "sbc_ks_pvalue_min": rep["sbc_ks_pvalue_min"],
        "coverage_mae": rep["coverage_mae"],
        "posterior_sd_over_prior_sd": rep.get("posterior_sd_over_prior_sd"),
        "npe_rejected": int(post.npe_rejected),
        "n_val": int(y_val.shape[0]),
        "sbc_n_bins": PRODUCTION_SBC_BINS,
        "eval_context": "unmasked, as evaluate.posterior_calibration",
        "seconds": round(time.time() - t0, 1),
    }


def discharges(cell: dict[str, Any]) -> bool:
    """All three of ISSUE-012's conditions, together. R2 alone does not do it."""
    g = cell.get("log_G_r2")
    return (
        g is not None
        and g > DISCHARGE["log_G_r2"]
        and cell["sbc_ks_pvalue_min"] > DISCHARGE["sbc_ks_pvalue_min"]
        and cell["coverage_mae"] < DISCHARGE["coverage_mae"]
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/run4/scwbd-004.yaml")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--out", default="reports/run4_posterior")
    ap.add_argument("--quick", action="store_true", help="300 steps, one lr, both norms")
    ap.add_argument(
        "--lrs",
        default="4e-6,4e-5,2e-4,1e-3",
        help="absolute posterior LRs. 4e-6 is run 3's effective rate (0.02 x T4's 2.0e-4).",
    )
    ap.add_argument("--norms", default="layer_v1,dataset_std_v2")
    a = ap.parse_args()

    steps = 300 if a.quick else a.steps
    lrs = [float(x) for x in (["4e-6", "2e-4"] if a.quick else a.lrs.split(","))]
    norms = a.norms.split(",")

    cfg = load_config(a.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)

    print(f"sweep: {len(norms)} norms x {len(lrs)} lrs x {steps} steps on {device}", flush=True)
    print(f"config {a.config}  nuisance_dim={cfg.posterior.nuisance_dim}", flush=True)

    train_dl = _loader(cfg, "train", batch=a.batch, seed=a.seed, workers=a.workers)
    val_dl = _loader(cfg, "val", batch=a.batch, seed=a.seed, workers=0)
    print(f"corpus: {len(train_dl.dataset)} train / {len(val_dl.dataset)} val trajectories", flush=True)

    cells: list[dict[str, Any]] = []
    for norm in norms:
        for lr in lrs:
            print(f"\n  [{norm}  lr={lr:.1e}]", flush=True)
            cell = run_cell(
                cfg,
                cond_norm=norm,
                lr=lr,
                steps=steps,
                train_dl=train_dl,
                val_dl=val_dl,
                device=device,
                seed=a.seed,
                weight_decay=a.weight_decay,
            )
            cell["discharges_issue_012"] = discharges(cell)
            cells.append(cell)
            print(
                f"      -> log_G R2 {cell['log_G_r2']:+.3f}   -log q {cell['held_out_neg_log_q']:8.3f}"
                f"   ks_min {cell['sbc_ks_pvalue_min']:.3f}   cov_mae {cell['coverage_mae']:.3f}"
                f"   {'DISCHARGES' if cell['discharges_issue_012'] else ''}",
                flush=True,
            )
            (out / "sweep.json").write_text(json.dumps({"cells": cells}, indent=2))

    # -- the table -----------------------------------------------------------
    print(f"\n{'cond_norm':16s} {'lr':>9s} {'log_G R2':>9s} {'-log q':>9s} {'ks_min':>7s} {'cov_mae':>8s}")
    for c in cells:
        print(
            f"{c['cond_norm']:16s} {c['lr']:9.1e} {c['log_G_r2']:+9.3f} "
            f"{c['held_out_neg_log_q']:9.3f} {c['sbc_ks_pvalue_min']:7.3f} {c['coverage_mae']:8.3f}"
            + ("   DISCHARGES" if c["discharges_issue_012"] else "")
        )

    winners = [c for c in cells if c["discharges_issue_012"]]
    best = max(cells, key=lambda c: (c["log_G_r2"] if c["log_G_r2"] is not None else -9e9))
    summary = {
        "discharge_condition": DISCHARGE,
        "n_cells": len(cells),
        "n_discharging": len(winners),
        "best_by_log_G_r2": best,
        "recommended_lr_scale_at_T4": (best["lr"] / 2.0e-4) if best["log_G_r2"] else None,
        "cells": cells,
    }
    (out / "sweep.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out / 'sweep.json'}")

    if not winners:
        print(
            "\nNO CELL DISCHARGES ISSUE-012. That is a result, not a failure of the sweep: if a "
            "retrain under these conditions still returns the prior, the summary statistics are "
            "the wrong ones and this becomes an identifiability result about the observation "
            "operator -- which ISSUE-012 says is the more important finding. Do not widen the "
            "grid and re-run until something passes without saying so."
        )
        return 1
    print(
        f"\nbest: {best['cond_norm']} at lr {best['lr']:.1e} -> log_G R2 {best['log_G_r2']:+.3f}; "
        f"that is lr_scale {best['lr'] / 2.0e-4:.4g} against T4's 2.0e-4."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
