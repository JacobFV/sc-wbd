"""P0 — decompose SC-WBD-001-beta's run-1 NLL penalty into model and instrument parts.

Procedure is pre-registered in `reports/training/PREREG_p0_variance_decomposition.md`
(committed 4c5c1de, before this script was run). Rungs L0-L4 and the read-out rule
are fixed there; nothing here chooses them after the fact.

Every arm's CONDITIONAL MEAN is held exactly as that arm produced it. Only the
variance model changes across rungs, and it changes identically for every arm.

Run under the fleet memory budget:
    systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=4G -- \
      .venv/bin/python reports/training/p0_variance_decomposition.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

torch.cuda.set_per_process_memory_fraction(0.16)

from scwbd.foundation.baselines import (  # noqa: E402
    ARBaseline,
    DenseNeuralBaseline,
    PersistenceBaseline,
    PopulationGaussianBaseline,
    SubjectSpecificBaseline,
    VARBaseline,
    _paired_ci,
    _split_indices,
    _VAR_FLOOR,
    bootstrap_ci,
)
from scwbd.foundation.config import load_config  # noqa: E402
from scwbd.foundation.evaluate import _participant_stratified, split_fingerprint  # noqa: E402
from scwbd.foundation.simulate import THETA_NAMES  # noqa: E402

SEED = 0
N_BOOT = 2000
N_MEAN_SAMPLES = 256
OUT = Path("reports/training/p0_variance_decomposition.json")


# ---------------------------------------------------------------------------
# the ladder
# ---------------------------------------------------------------------------
def _nll_elements(resid_sq: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Gaussian NLL per element. `resid_sq` (N,H,C); `var` broadcastable to it."""
    return 0.5 * (math.log(2 * math.pi) + np.log(var) + resid_sq / var)


def ladder(resid_sq: np.ndarray, own_logvar: np.ndarray, calib_resid_sq: np.ndarray | None):
    """L0-L4 for one arm. Returns dict of rung -> per-window NLL array (N,).

    resid_sq       (N,H,C) squared residuals on TEST
    own_logvar     (N,H,C) the arm's own emitted log-variance
    calib_resid_sq (M,H,C) squared residuals on the held-out CALIBRATION split,
                   or None if unavailable for this arm
    """
    out: dict[str, np.ndarray] = {}
    # L0 -- as shipped
    out["L0"] = _nll_elements(resid_sq, np.exp(own_logvar)).mean(axis=(1, 2))
    # L1 -- one global scalar (oracle). Equals 0.5*log(2*pi*e*MSE) in the mean.
    v1 = np.maximum(resid_sq.mean(), _VAR_FLOOR)
    out["L1"] = _nll_elements(resid_sq, v1).mean(axis=(1, 2))
    # L2 -- per channel (oracle)
    v2 = np.maximum(resid_sq.mean(axis=(0, 1), keepdims=True), _VAR_FLOOR)
    out["L2"] = _nll_elements(resid_sq, v2).mean(axis=(1, 2))
    # L3 -- per (horizon, channel) (oracle)
    v3 = np.maximum(resid_sq.mean(axis=0, keepdims=True), _VAR_FLOOR)
    out["L3"] = _nll_elements(resid_sq, v3).mean(axis=(1, 2))
    # L4 -- per (horizon, channel), fitted on held-out calibration windows.
    # Same estimator as baselines._LinearForecaster._calibrate_variance:
    #   var = resid.pow(2).sum(0) / max(n-1, 1), clamped at _VAR_FLOOR
    if calib_resid_sq is not None and calib_resid_sq.shape[0] > 1:
        n = calib_resid_sq.shape[0]
        v4 = np.maximum(calib_resid_sq.sum(axis=0, keepdims=True) / max(n - 1, 1), _VAR_FLOOR)
        out["L4"] = _nll_elements(resid_sq, v4).mean(axis=(1, 2))
    # L5/L6 -- per-WINDOW oracles. L1-L4 all vary only over (horizon, channel);
    # a state-dependent variance head can in principle do better than any of
    # them, so without these the horizon term (L2-L3) would be reported as if it
    # bounded what state-dependence is worth. It does not. These bound it.
    v5 = np.maximum(resid_sq.mean(axis=(1, 2), keepdims=True), _VAR_FLOOR)  # (N,1,1)
    out["L5"] = _nll_elements(resid_sq, v5).mean(axis=(1, 2))
    v6 = np.maximum(resid_sq.mean(axis=1, keepdims=True), _VAR_FLOOR)  # (N,1,C)
    out["L6"] = _nll_elements(resid_sq, v6).mean(axis=(1, 2))
    return out


# ---------------------------------------------------------------------------
# SC-WBD residual collection -- mirrors evaluate._scwbd_scores exactly
# ---------------------------------------------------------------------------
def scwbd_residuals(trainer, loader, *, n_mean_samples: int = N_MEAN_SAMPLES):
    """Return (resid_sq (N,H,C), logvar (N,H,C), subjects [N])."""
    model, cfg = trainer.model, trainer.cfg
    model.eval()
    c = cfg.data.context
    rs: list[np.ndarray] = []
    lv_: list[np.ndarray] = []
    subs: list[str] = []
    with torch.no_grad():
        for batch in loader:
            eeg = batch["eeg"].to(trainer.device)
            ctx_e, tgt_e = eeg[:, :c], eeg[:, c:]
            src = trainer.sensor_to_parcel(ctx_e)
            src = src / src.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            y = tgt_e.float()
            th_bar = trainer.posterior.sample(ctx_e, n_mean_samples).mean(dim=1)
            th_bar = th_bar[:, : len(THETA_NAMES)]
            _ind = getattr(trainer, "individualizer", None)
            if _ind is not None:
                _pid = trainer.participant_index(list(batch.get("subject", [])))
                th_bar = _ind(participant=_pid, base=th_bar)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.model.use_bf16):
                roll = model.rollout(
                    y_context=src, theta=th_bar, n_steps=y.shape[1], enforce_r05=False
                )
                mu, lv = model.eeg(roll.state)
            m_k = mu.float()
            v_k = lv.float().clamp(-14, 14)
            rs.append(((y - m_k) ** 2).double().cpu().numpy())
            lv_.append(v_k.double().cpu().numpy())
            subs.extend(list(batch["subject"]))
    model.train()
    return np.concatenate(rs), np.concatenate(lv_), subs


def main() -> dict[str, Any]:
    from scwbd.foundation.checkpoint import load_checkpoint
    from scwbd.foundation.train import FoundationTrainer
    from scwbd.foundation.util import set_determinism

    set_determinism(SEED)
    cfg = load_config("configs/scwbd_001_beta.yaml")
    tr = FoundationTrainer(cfg, resume=False, quick=False)
    tr.build_data()

    ckpt = str(Path(cfg.train.out_dir) / "last.pt")
    _peek = torch.load(ckpt, map_location="cpu", weights_only=False)
    if _peek.get("individualizer") is not None and tr.individualizer is None:
        from scwbd.foundation.individual import Individualizer

        _np_ = max(len(tr._participant_ids()), 1)
        tr.individualizer = Individualizer(
            len(THETA_NAMES), n_groups=2, n_participants=_np_, n_sessions=max(_np_ * 4, 1)
        ).to(tr.device)
    del _peek
    payload = load_checkpoint(
        ckpt,
        model=tr.model,
        posterior=tr.posterior,
        individualizer=tr.individualizer,
        map_location=str(tr.device),
        strict=False,
    )
    rep = payload.get("load_report", {})
    miss = [*rep.get("missing", []), *rep.get("posterior_missing", [])]
    unexp = [*rep.get("unexpected", []), *rep.get("posterior_unexpected", [])]
    if miss or unexp:
        raise SystemExit(f"checkpoint did not load cleanly: {len(miss)} missing, {len(unexp)} unexpected")
    print(f"loaded {ckpt}", flush=True)

    # -- identical split to real_eeg_holdout ---------------------------------
    ds, split = tr.real_dataset, tr.real_split
    c = cfg.data.context
    te_idx = _participant_stratified(ds, split["test"], 40, fold="test")
    tr_idx = _participant_stratified(ds, split["train"], 30, fold="train")
    bs = max(8, cfg.data.batch // 4)
    mk = lambda idx: torch.utils.data.DataLoader(  # noqa: E731
        torch.utils.data.Subset(ds, idx), batch_size=bs, shuffle=False, num_workers=2
    )
    test_loader, train_loader = mk(te_idx), mk(tr_idx)

    def collect(loader):
        xs, ss = [], []
        for b in loader:
            xs.append(b["eeg"])
            ss.extend(list(b["subject"]))
        return torch.cat(xs), ss

    tr_x, tr_s = collect(train_loader)
    te_x, te_s = collect(test_loader)
    dev = tr.device
    tr_x, te_x = tr_x.to(dev), te_x.to(dev)
    ctx, tgt = te_x[:, :c], te_x[:, c:]
    horizon = int(tgt.shape[1])
    groups = np.asarray(te_s)

    # The calibration split for L4: the SAME slice the linear baselines hold out
    # at fit time, so every arm calibrates on identical windows.
    fit_idx, calib_idx = _split_indices(tr_x.shape[0], 0.25, SEED)
    calib_x = tr_x[calib_idx.to(tr_x.device)]
    calib_ctx, calib_tgt = calib_x[:, :c], calib_x[:, c:]
    calib_groups = np.asarray(tr_s)[calib_idx.cpu().numpy()]
    print(
        f"test {tuple(te_x.shape)} ({len(set(te_s))} participants) | "
        f"train {tuple(tr_x.shape)} | calib {tuple(calib_x.shape)}",
        flush=True,
    )

    arms: dict[str, dict[str, Any]] = {}

    # -- SC-WBD --------------------------------------------------------------
    set_determinism(SEED)
    scw_rs, scw_lv, scw_subs = scwbd_residuals(tr, test_loader)
    assert list(scw_subs) == list(te_s), "SC-WBD window order does not match the baselines'"
    set_determinism(SEED)
    calib_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, [tr_idx[i] for i in calib_idx.tolist()]),
        batch_size=bs,
        shuffle=False,
        num_workers=2,
    )
    scw_cal_rs, _, _ = scwbd_residuals(tr, calib_loader)
    arms["scwbd_001_beta"] = {
        "resid_sq": scw_rs,
        "logvar": scw_lv,
        "calib_resid_sq": scw_cal_rs,
        "n_parameters": sum(p.numel() for p in tr.model.parameters()),
    }
    print("scwbd residuals collected", flush=True)

    # -- baselines -----------------------------------------------------------
    models = {
        "persistence": PersistenceBaseline(),
        "ar16": ARBaseline(order=16),
        "var4": VARBaseline(order=4),
        "population_gaussian": PopulationGaussianBaseline(),
        "subject_specific_ar": SubjectSpecificBaseline(
            ARBaseline, base_kwargs={"order": 16}, name="subject_specific_ar"
        ),
        "dense_neural": DenseNeuralBaseline(
            target_parameters=arms["scwbd_001_beta"]["n_parameters"], steps=400, seed=SEED
        ),
    }
    for name, m in models.items():
        m.fit(tr_x, groups=np.asarray(tr_s))
        with torch.no_grad():
            mean, log_var = m.predict(ctx, horizon, groups=groups)
            rsq = (mean - tgt).pow(2).double().cpu().numpy()
            lv = log_var.double().cpu().numpy()
            cmean, _ = m.predict(calib_ctx, horizon, groups=calib_groups)
            crsq = (cmean - calib_tgt).pow(2).double().cpu().numpy()
        arms[name] = {
            "resid_sq": rsq,
            "logvar": lv,
            "calib_resid_sq": crsq,
            "n_parameters": int(m.n_parameters()),
            "describe": m.describe(),
        }
        print(f"  {name} scored", flush=True)

    # -- ladder + intervals --------------------------------------------------
    table: dict[str, Any] = {}
    pw: dict[str, dict[str, np.ndarray]] = {}
    for name, a in arms.items():
        rungs = ladder(a["resid_sq"], a["logvar"], a["calib_resid_sq"])
        pw[name] = rungs
        mse_pw = a["resid_sq"].mean(axis=(1, 2))
        pw[name]["MSE"] = mse_pw
        mse_pt, mse_lo, mse_hi = bootstrap_ci(mse_pw, groups, n_boot=N_BOOT, seed=SEED)
        row = {
            "mse": mse_pt,
            "mse_ci95": [mse_lo, mse_hi],
            "n_parameters": a["n_parameters"],
            "entropy_floor_check": 0.5 * math.log(2 * math.pi * math.e * mse_pt),
        }
        for k, v in rungs.items():
            pt, lo, hi = bootstrap_ci(v, groups, n_boot=N_BOOT, seed=SEED)
            row[k] = {"nll": pt, "ci95": [lo, hi]}
        row["model_attributable_L0_minus_L1"] = row["L0"]["nll"] - row["L1"]["nll"]
        row["instrument_attributable_L1_minus_L3"] = row["L1"]["nll"] - row["L3"]["nll"]
        if "L4" in row:
            row["total_removable_L0_minus_L4"] = row["L0"]["nll"] - row["L4"]["nll"]
        table[name] = row

    # -- paired participant-clustered intervals, SC-WBD vs each baseline -----
    paired: dict[str, Any] = {}
    for name in models:
        entry: dict[str, Any] = {}
        # MSE: the interval that was missing. Negative favours SC-WBD.
        entry["mse"] = _paired_ci(
            pw["scwbd_001_beta"]["MSE"] - pw[name]["MSE"], groups, n_boot=N_BOOT, alpha=0.05, seed=SEED
        )
        # NLL at each rung. Positive => SC-WBD worse.
        for rung in ("L0", "L1", "L3", "L4", "L5", "L6"):
            if rung in pw["scwbd_001_beta"] and rung in pw[name]:
                entry[rung] = _paired_ci(
                    pw["scwbd_001_beta"][rung] - pw[name][rung],
                    groups,
                    n_boot=N_BOOT,
                    alpha=0.05,
                    seed=SEED,
                )
        paired[name] = entry

    result = {
        "prereg": "reports/training/PREREG_p0_variance_decomposition.md @ 4c5c1de",
        "seed": SEED,
        "n_boot": N_BOOT,
        "n_test_windows": int(te_x.shape[0]),
        "n_test_participants": int(len(set(te_s))),
        "n_calib_windows": int(calib_x.shape[0]),
        "horizon": horizon,
        "n_channels": int(tgt.shape[2]),
        "real_split": split_fingerprint(ds, split),
        "rung_semantics": {
            "L0": "the arm's own emitted variance, as shipped",
            "L1": "one global scalar variance, fitted on TEST (ORACLE, optimistic)",
            "L2": "per-channel variance, fitted on TEST (ORACLE, optimistic)",
            "L3": "per-(horizon,channel) variance, fitted on TEST (ORACLE, optimistic)",
            "L5": "per-WINDOW global scalar variance, fitted on TEST (ORACLE)",
            "L6": "per-(window,channel) variance, fitted on TEST (ORACLE); together with L5 these bound what a STATE-dependent variance head could buy, which the (horizon,channel) rungs do not",
            "L4": (
                "per-(horizon,channel) variance fitted on held-out CALIBRATION windows; "
                "the only rung that is a score. In-sample for SC-WBD (it trained on "
                "these participants) and genuinely held out for the baselines, so L4 "
                "FLATTERS SC-WBD -- declared in the prereg before running."
            ),
        },
        "difference_convention": (
            "scwbd minus baseline. NLL/MSE lower is better, so POSITIVE means SC-WBD is worse."
        ),
        "interval_method": "cluster bootstrap over participants",
        "table": table,
        "paired_vs_scwbd": paired,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o)))
    print(json.dumps(table, indent=2, default=float))
    return result


if __name__ == "__main__":
    main()
