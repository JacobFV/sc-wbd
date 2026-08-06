"""Phase 4 -- honest evaluation of SC-WBD-001-beta.

What is measured, and against what:

* **Held-out real-EEG log-likelihood** at a *participant-level* holdout, in nats
  per channel per sample, with participant-clustered bootstrap intervals.  The
  likelihood is evaluated in **sensor space**, where the measurement lives.
* **Baselines**: persistence, per-channel AR, VAR, a population Gaussian, a
  subject-specific statistical model, and an equal-capacity dense neural
  forecaster with no connectome and no structured state.  If SC-WBD loses to one
  of these, that is the result and it is reported (ARCHITECTURE.md §4).
* **Amortized-posterior calibration**: SBC rank histograms, KS uniformity
  p-values and expected-coverage curves -- on simulator-conditioned evidence,
  which certifies self-consistency and *not* biological validity.
* **Per-source contribution and negative transfer**: remove each source family in
  turn (Appendix D, "Dataset-family breadth").
* **Backend comparison**: the mechanistic backends against the learned operator
  at matched inputs, so a mechanistic label has to be earned.
* **Ablations**: scalar-state, dense coupling, randomized / distance-matched /
  local-only connectomes, no-residual.

Nothing here decides whether a claim gate passes.  The numbers go to agent J.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .config import FoundationConfig, load_config
from .heads import gaussian_nll
from .simulate import THETA_NAMES, ThetaPrior
from .util import Timer, git_sha, set_determinism

__all__ = ["evaluate_model", "real_eeg_holdout", "posterior_calibration", "source_ablation", "main"]


# ======================================================================
# real EEG held-out likelihood
# ======================================================================
@torch.no_grad()
def _scwbd_scores(
    trainer, loader, *, max_batches: int | None = None, n_theta_samples: int = 32
) -> dict[str, Any]:
    """Per-window sensor-space NLL and MSE of the foundation model."""
    model, cfg = trainer.model, trainer.cfg
    model.eval()
    c = cfg.data.context
    nlls: list[np.ndarray] = []
    nlls_norm: list[np.ndarray] = []
    mses: list[np.ndarray] = []
    subs: list[str] = []
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        eeg = batch["eeg"].to(trainer.device)
        ctx_e, tgt_e = eeg[:, :c], eeg[:, c:]
        src = trainer.sensor_to_parcel(ctx_e)
        src = src / src.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        # MARGINALISE over the posterior rather than scoring one draw. The metric
        # names a predictive density, p(y | context) = E_theta[p(y | theta)], and a
        # single sample is a plug-in estimator of a different quantity. With a wide,
        # weakly-informative posterior (Stage III: z_sd 1.0-1.4, R^2 <= 0.21) one
        # draw is materially worse than the distribution it came from.
        y = tgt_e.float()
        n_elem = float(y.shape[1] * y.shape[2])
        th_all = trainer.posterior.sample(ctx_e, n_theta_samples)  # (B, K, P)
        joint_ll: list[Tensor] = []
        mu_sum = None
        for k in range(n_theta_samples):
            th = th_all[:, k][:, : len(THETA_NAMES)]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.model.use_bf16):
                roll = model.rollout(
                    y_context=src, theta=th, n_steps=tgt_e.shape[1], enforce_r05=False
                )
                mu, lv = model.eeg(roll.state)
            m_k = mu.float()
            v_k = lv.float().clamp(-14, 14)
            # RAW data units: the baselines score the raw target through
            # baselines._gaussian_nll, so rescaling by the target's own per-window
            # std would compare densities of two DIFFERENT random variables
            # (NLL_scaled = NLL_raw - log s; mean log s = 0.598 here, ~17x the
            # 0.035-nat spread between the non-trivial baselines).
            nll_el = 0.5 * (math.log(2 * math.pi) + v_k + (y - m_k) ** 2 * torch.exp(-v_k))
            joint_ll.append(-nll_el.sum(dim=(1, 2)))  # (B,) log p(y | theta_k)
            mu_sum = m_k if mu_sum is None else mu_sum + m_k
        # log (1/K) sum_k p(y | theta_k), then back to nats per channel per sample
        L = torch.stack(joint_ll, dim=1)  # (B, K)
        logp = torch.logsumexp(L, dim=1) - math.log(float(n_theta_samples))
        nlls.append((-logp / n_elem).cpu().numpy())
        m_bar = mu_sum / float(n_theta_samples)
        mses.append(((y - m_bar) ** 2).mean(dim=(1, 2)).cpu().numpy())
        # Secondary, separately labelled: amplitude-normalised score. Defensible as
        # *a* metric; NOT comparable to the baselines. Change of variables by a
        # constant per window, so it is the marginal score shifted by log s.
        s_ = tgt_e.std(dim=(1, 2), keepdim=True).clamp_min(1e-8).float()
        nlls_norm.append(
            ((-logp / n_elem) - torch.log(s_).reshape(-1)).cpu().numpy()
        )
        subs.extend(list(batch["subject"]))
    model.train()
    return {
        "nll_per_window": np.concatenate(nlls) if nlls else np.zeros(0),
        "nll_per_window_amplitude_normalised": (
            np.concatenate(nlls_norm) if nlls_norm else np.zeros(0)
        ),
        "mse_per_window": np.concatenate(mses) if mses else np.zeros(0),
        "subjects": subs,
        "units_note": (
            "nll_per_window and mse_per_window are in RAW data units, matching "
            "baselines._gaussian_nll. nll_per_window_amplitude_normalised divides by "
            "the target's own per-window std and is NOT comparable to the baselines."
        ),
    }


def real_eeg_holdout(
    trainer,
    *,
    max_batches: int | None = 40,
    n_boot: int = 2000,
    seed: int = 0,
    n_theta_samples: int = 32,
) -> dict[str, Any]:
    """SC-WBD vs every required baseline on a participant-level holdout.

    Windows within a participant are correlated, so the interval is a **cluster
    bootstrap over participants**.  A window-level interval here would be a
    silent overstatement of precision, which is the exact failure Appendix D's
    "Participant or family leakage" row is about.
    """
    from .baselines import (
        ARBaseline,
        DenseNeuralBaseline,
        PersistenceBaseline,
        PopulationGaussianBaseline,
        SubjectSpecificBaseline,
        VARBaseline,
        bootstrap_ci,
    )

    if getattr(trainer, "real_test", None) is None or len(trainer.real_test) == 0:
        return {"available": False, "reason": "no measured EEG holdout available"}
    cfg = trainer.cfg
    c = cfg.data.context
    bs = max(8, cfg.data.batch // 4)
    test_loader = torch.utils.data.DataLoader(trainer.real_test, batch_size=bs, shuffle=False, num_workers=2)
    train_loader = torch.utils.data.DataLoader(trainer.real_train, batch_size=bs, shuffle=False, num_workers=2)

    def collect(loader, limit):
        xs, ss = [], []
        for i, b in enumerate(loader):
            if limit is not None and i >= limit:
                break
            xs.append(b["eeg"])
            ss.extend(list(b["subject"]))
        return (torch.cat(xs) if xs else torch.zeros(0), ss)

    tr_x, tr_s = collect(train_loader, max_batches)
    te_x, te_s = collect(test_loader, max_batches)
    if te_x.numel() == 0:
        return {"available": False, "reason": "empty holdout"}
    dev = trainer.device
    tr_x, te_x = tr_x.to(dev), te_x.to(dev)
    ctx, tgt = te_x[:, :c], te_x[:, c:]

    scw = _scwbd_scores(
        trainer, test_loader, max_batches=max_batches, n_theta_samples=n_theta_samples
    )
    n_model_params = sum(p.numel() for p in trainer.model.parameters())

    models: dict[str, Any] = {
        "persistence": PersistenceBaseline(),
        "ar16": ARBaseline(order=16),
        "var4": VARBaseline(order=4),
        "population_gaussian": PopulationGaussianBaseline(),
        "subject_specific_ar": SubjectSpecificBaseline(order=16),
        "dense_neural": DenseNeuralBaseline(target_parameters=n_model_params, steps=400, seed=seed),
    }
    rows: dict[str, Any] = {}
    for name, m in models.items():
        t0 = time.time()
        try:
            m.fit(tr_x, groups=tr_s)
            r = m.score(ctx, tgt, groups=te_s) if _accepts_groups(m.score) else m.score(ctx, tgt)
            per = np.asarray(r.get("nll_per_window", r.get("per_window_nll", [])), dtype=float)
            pt, lo, hi = bootstrap_ci(per, np.asarray(te_s), n_boot=n_boot, seed=seed) if per.size else (r["nll_per_sample"], float("nan"), float("nan"))
            rows[name] = {
                "nll_per_sample": float(r["nll_per_sample"]),
                "nll_ci95": [float(lo), float(hi)],
                "mse": float(r.get("mse", float("nan"))),
                "n_parameters": int(_nparams(m)),
                "fit_seconds": round(time.time() - t0, 2),
                "describe": m.describe() if hasattr(m, "describe") else {},
            }
        except Exception as exc:  # noqa: BLE001 - a baseline that cannot run is reported, not hidden
            rows[name] = {"error": f"{type(exc).__name__}: {exc}"}

    pt, lo, hi = bootstrap_ci(scw["nll_per_window"], np.asarray(scw["subjects"]), n_boot=n_boot, seed=seed)
    rows["scwbd_001_beta"] = {
        "nll_per_sample": float(np.mean(scw["nll_per_window"])),
        "nll_ci95": [float(lo), float(hi)],
        "mse": float(np.mean(scw["mse_per_window"])),
        "n_parameters": int(n_model_params),
        "describe": {"name": "SC-WBD-001-beta", "structured_state": True, "connectome_masked": True},
    }
    ref = rows["scwbd_001_beta"]["nll_per_sample"]
    ranking = sorted(
        [(k, v["nll_per_sample"]) for k, v in rows.items() if "nll_per_sample" in v], key=lambda kv: kv[1]
    )
    beaten_by = [k for k, v in ranking if v < ref]
    return {
        "available": True,
        "n_test_windows": int(te_x.shape[0]),
        "n_test_participants": int(len(set(te_s))),
        "n_train_windows": int(tr_x.shape[0]),
        "n_train_participants": int(len(set(tr_s))),
        "metric": "gaussian NLL, nats per channel per sample, sensor space, participant-clustered 95% CI",
        "results": rows,
        "ranking_best_first": ranking,
        "scwbd_beaten_by": beaten_by,
        "verdict": (
            "SC-WBD-001-beta is NOT the best model on this holdout"
            if beaten_by
            else "SC-WBD-001-beta has the lowest point estimate on this holdout"
        ),
        "interpretation": (
            "A lower point estimate is not a claim. Overlapping participant-clustered intervals mean "
            "the comparison is inconclusive at this sample size. Window-level generalisation is not "
            "individual generalisation."
        ),
    }


def _nparams(m: Any) -> int:
    v = getattr(m, "n_parameters", 0)
    try:
        return int(v() if callable(v) else v)
    except Exception:  # noqa: BLE001
        return 0


def _accepts_groups(fn) -> bool:
    import inspect

    try:
        return "groups" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


# ======================================================================
# posterior calibration
# ======================================================================
def posterior_calibration(trainer, *, n_datasets: int = 512, n_samples: int = 256) -> dict[str, Any]:
    """SBC + expected coverage on held-out simulated trajectories."""
    from .posterior import posterior_report

    loader = torch.utils.data.DataLoader(trainer.sim_val, batch_size=64, shuffle=False, num_workers=2)
    ys, ths = [], []
    n = 0
    c = trainer.cfg.data.context
    for b in loader:
        y = b["activity"][:, :c].to(trainer.device)
        ys.append(y)
        ths.append(b["theta"].to(trainer.device))
        n += y.shape[0]
        if n >= n_datasets:
            break
    if not ys:
        return {"available": False, "reason": "no simulated validation trajectories"}
    y = torch.cat(ys)[:n_datasets]
    th = torch.cat(ths)[:n_datasets]
    trainer.posterior.eval()
    rep = posterior_report(trainer.posterior, y, th, param_names=THETA_NAMES, n_samples=n_samples)
    trainer.posterior.train()
    rep["available"] = True
    return rep


# ======================================================================
# per-source contribution / negative transfer
# ======================================================================
def source_ablation(trainer, *, steps: int = 120, seed: int = 0) -> dict[str, Any]:
    """Leave-one-source-family-out: does each family earn its place?

    Appendix D, "Dataset-family breadth": "A longer source list is useful only
    when each family improves a specified port or exposes a failure."  Negative
    transfer -- a family whose removal *improves* the metric -- is reported with
    the same prominence as a gain.
    """
    import copy

    base_state = copy.deepcopy(trainer.model.state_dict())
    base_post = copy.deepcopy(trainer.posterior.state_dict())
    stage = trainer.cfg.train.stages[-2] if len(trainer.cfg.train.stages) > 1 else trainer.cfg.train.stages[0]
    families = [k for k, v in trainer.sources.items() if v.enabled and v.role not in ("negative_control", "evaluation_only")]
    out: dict[str, Any] = {"families": families, "steps_per_arm": steps}

    def short_train(drop: str | None) -> float:
        set_determinism(seed)
        trainer.model.load_state_dict(base_state)
        trainer.posterior.load_state_dict(base_post)
        saved = dict(trainer.sources)
        if drop is not None:
            trainer.sources = {k: v for k, v in saved.items() if k != drop}
        st = copy.deepcopy(stage)
        st.steps = steps
        st.ckpt_every = 10**9
        st.log_every = 10**9
        trainer.run_stage(st)
        trainer.sources = saved
        val = _sim_val_nll(trainer)
        return val

    out["with_all_sources"] = short_train(None)
    for f in families:
        v = short_train(f)
        out[f"without_{f}"] = v
        out[f"delta_{f}"] = v - out["with_all_sources"]
    out["negative_transfer"] = [f for f in families if out[f"delta_{f}"] < 0]
    out["interpretation"] = (
        "delta > 0 means removing the family HURT (the family contributed); delta < 0 is "
        "NEGATIVE TRANSFER: the family made the model worse and is reported as such."
    )
    trainer.model.load_state_dict(base_state)
    trainer.posterior.load_state_dict(base_post)
    return out


@torch.no_grad()
def _sim_val_nll(trainer, *, max_batches: int = 8) -> float:
    loader = torch.utils.data.DataLoader(trainer.sim_val, batch_size=64, shuffle=False, num_workers=2)
    c = trainer.cfg.data.context
    tot, n = 0.0, 0
    trainer.model.eval()
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        act = b["activity"].to(trainer.device)
        th = b["theta"].to(trainer.device)
        ctx, tgt = act[:, :c], act[:, c:]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=trainer.cfg.model.use_bf16):
            r = trainer.model.rollout(y_context=ctx, theta=th, n_steps=tgt.shape[1], enforce_r05=False)
        tot += float(gaussian_nll(tgt, r.activity.float(), r.activity_logvar.float())) * act.shape[0]
        n += act.shape[0]
    trainer.model.train()
    return tot / max(n, 1)


# ======================================================================
# backend comparison
# ======================================================================
@torch.no_grad()
def backend_comparison(trainer, *, max_batches: int = 6) -> dict[str, Any]:
    """Per-backend held-out forecast NLL of the learned operator.

    The simulated validation set carries a backend label, so this reports where
    the single learned operator succeeds and fails **across mechanistic
    families**.  It is not a claim that any family is neurally realized.
    """
    loader = torch.utils.data.DataLoader(trainer.sim_val, batch_size=64, shuffle=False, num_workers=2)
    c = trainer.cfg.data.context
    names = trainer.sim_val.backend_names
    acc: dict[str, list[float]] = {n: [] for n in names}
    trainer.model.eval()
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        act = b["activity"].to(trainer.device)
        th = b["theta"].to(trainer.device)
        ctx, tgt = act[:, :c], act[:, c:]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=trainer.cfg.model.use_bf16):
            r = trainer.model.rollout(y_context=ctx, theta=th, n_steps=tgt.shape[1], enforce_r05=False)
        lv = r.activity_logvar.float().clamp(-14, 14)
        nll = (0.5 * (math.log(2 * math.pi) + lv + (tgt - r.activity.float()) ** 2 * torch.exp(-lv))).mean(dim=(1, 2))
        for j, bi in enumerate(b["backend"].tolist()):
            acc[names[bi]].append(float(nll[j]))
    trainer.model.train()
    return {
        "per_backend_nll": {k: (float(np.mean(v)) if v else None) for k, v in acc.items()},
        "per_backend_n": {k: len(v) for k, v in acc.items()},
        "note": (
            "The learned operator is the equal-capacity control for every mechanistic claim "
            "(Appendix D). Matching a family's trajectories is not evidence that the family is "
            "neurally realized."
        ),
    }


# ======================================================================
# driver
# ======================================================================
def evaluate_model(
    trainer, *, quick: bool = False, out: str | Path | None = None, seed: int = 0
) -> dict[str, Any]:
    # Seed the whole evaluation. The posterior is sampled stochastically, and
    # measured run-to-run sd of the headline is 0.0075 nats -- larger than the
    # ar16 <-> var4 gap of 0.0053. An unseeded evaluation cannot distinguish two
    # baselines it is required to rank.
    set_determinism(seed)
    t = Timer()
    trainer.build_data()
    rep: dict[str, Any] = {
        "model_id": "SC-WBD-001-beta",
        "git_sha": git_sha(),
        "evaluated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": trainer.cfg.as_dict(),
        "n_parameters": trainer.model.parameter_report(),
        "eval_seed": seed,
        "anatomy": trainer.anat.summary(),
        "lead_field": trainer.model.eeg.lead_field_meta,
        "sensor_to_parcel": trainer.sensor_to_parcel.summary(),
    }
    rep["posterior_calibration"] = posterior_calibration(
        trainer, n_datasets=128 if quick else 512, n_samples=64 if quick else 256
    )
    rep["backend_comparison"] = backend_comparison(trainer, max_batches=2 if quick else 6)
    rep["real_eeg_holdout"] = real_eeg_holdout(trainer, max_batches=6 if quick else 40)
    rep["sim_val_nll"] = _sim_val_nll(trainer, max_batches=2 if quick else 8)
    rep["wall_seconds"] = t.elapsed
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(rep, indent=2, default=_jsonable))
    return rep


def _jsonable(o: Any):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:  # pragma: no cover
    import argparse

    from .train import FoundationTrainer

    p = argparse.ArgumentParser(description="evaluate SC-WBD-001-beta honestly")
    p.add_argument("--config", default="configs/scwbd_001_beta.yaml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out", default="reports/training/evaluation.json")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ablate-sources", action="store_true")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    tr = FoundationTrainer(cfg, resume=False, quick=a.quick)
    ckpt = a.checkpoint or str(Path(cfg.train.out_dir) / "last.pt")
    if Path(ckpt).exists():
        from .checkpoint import CheckpointError, load_checkpoint

        payload = load_checkpoint(
            ckpt, model=tr.model, posterior=tr.posterior, map_location=str(tr.device), strict=False
        )
        # `strict=False` is needed because the posterior/individualizer may be
        # absent, but load_checkpoint RECORDS what it dropped and the caller used
        # to discard it. torch.compile prefixes `local._orig_mod.*`, and the model
        # is compiled only when `cfg.model.compile and device.type == "cuda"` -- so
        # evaluating on CPU silently dropped 80.2% of the parameter mass (all of
        # `local` + `residual`) and scored RANDOM WEIGHTS while printing "loaded".
        report = payload.get("load_report", {})
        missing, unexpected = report.get("missing", []), report.get("unexpected", [])
        if missing or unexpected:
            raise CheckpointError(
                f"{ckpt} did not load cleanly: {len(missing)} missing, "
                f"{len(unexpected)} unexpected keys.\n"
                f"  first missing:    {list(missing)[:3]}\n"
                f"  first unexpected: {list(unexpected)[:3]}\n"
                "This is usually a torch.compile '_orig_mod.' prefix mismatch: the "
                "checkpoint was written by a compiled model and this process did not "
                "compile. Evaluating anyway would score randomly-initialised weights "
                "for those modules and report the result as the model's."
            )
        print(f"loaded {ckpt} ({len(payload.get('model', {}))} model tensors, no key mismatch)", flush=True)
    else:
        print(f"[warn] no checkpoint at {ckpt}: evaluating an untrained model", flush=True)
    rep = evaluate_model(tr, quick=a.quick, out=a.out, seed=a.seed)
    if a.ablate_sources:
        rep["source_ablation"] = source_ablation(tr, steps=60 if a.quick else 200)
        Path(a.out).write_text(json.dumps(rep, indent=2, default=_jsonable))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("config",)}, indent=2, default=_jsonable)[:4000])
    return rep


if __name__ == "__main__":  # pragma: no cover
    main()
