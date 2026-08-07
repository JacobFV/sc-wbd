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

import hashlib
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .config import FoundationConfig, load_config, designation as _cfg_designation
from .heads import gaussian_nll
from .simulate import THETA_NAMES, ThetaPrior
from .util import Timer, git_sha, set_determinism

__all__ = ["evaluate_model", "real_eeg_holdout", "posterior_calibration", "source_ablation", "main"]


# ======================================================================
# real EEG held-out likelihood
# ======================================================================
@torch.no_grad()
def _designation(cfg) -> str:
    """Delegates to :func:`scwbd.foundation.config.designation`.

    Kept as a name so existing call sites and tests do not move, but it must not
    carry a second copy of the rule: two derivations of one name is the same
    defect as two literals, one refactor later.
    """
    return _cfg_designation(cfg)


def _bind_mechanistic(model, theta) -> None:
    """Bind theta-conditioned ParamPacks before a rollout, if the arm needs them.

    Seventh site to require this.  Every one was a path declared for both arms
    and exercised only on the control, which has no mechanistic families -- see
    ``reports/decorative_guards.md``, "the arm-asymmetry class".  Without it a
    family-state checkpoint raises ``SpanViolation``; with a naive workaround it
    would silently run the engineered subcortical backends on their defaults.
    """
    bind = getattr(model, "set_mechanistic_theta", None)
    if bind is None or getattr(model, "family_layout", None) is None:
        return
    anat = getattr(model, "anat", None) or getattr(model, "_anat", None)
    if anat is None:
        from scwbd.foundation.anatomy import load_anatomy

        anat = load_anatomy()
    bind(theta, anat)


@torch.no_grad()
def _scwbd_scores(
    trainer,
    loader,
    *,
    n_theta_samples: int = 32,
    n_mean_samples: int = 256,
) -> dict[str, Any]:
    """Per-window sensor-space NLL and MSE of the foundation model.

    **Headline is PLUG-IN at the posterior mean, matching the baselines.**

    Every baseline is a plug-in estimator: ``ARBaseline`` and friends score at
    point-estimated coefficients and do not integrate over coefficient
    uncertainty.  Marginalising SC-WBD over theta while the baselines stay
    plug-in breaks like-for-like in exactly the way the units mismatch did --
    worth **0.0377 nats**, 7x the gap that decides a rank and larger than the
    whole 0.035-nat spread between the non-trivial baselines.

    The marginal is retained as a **separately labelled secondary** with its
    diagnostics, because on this posterior it is not a converged predictive:
    median effective sample size is ~1 of 64 draws, the best draw carries ~98%
    of the mass, and the estimate still drifts with K.  A genuine
    predictive-vs-predictive comparison needs coefficient uncertainty on the
    baselines, which is a project rather than a patch.
    """
    model, cfg = trainer.model, trainer.cfg
    model.eval()
    c = cfg.data.context
    nlls: list[np.ndarray] = []
    nlls_norm: list[np.ndarray] = []
    mses: list[np.ndarray] = []
    marg: list[np.ndarray] = []
    marg_half: list[np.ndarray] = []
    ess: list[np.ndarray] = []
    top_mass: list[np.ndarray] = []
    subs: list[str] = []
    seen_pids: set[int] = set()

    def _score(th: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        """Return (per-element NLL summed over the window, mean prediction)."""
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.model.use_bf16):
            _bind_mechanistic(model, th)
            roll = model.rollout(y_context=src, theta=th, n_steps=y.shape[1], enforce_r05=False)
            mu, lv = model.eeg(roll.state)
        m_k = mu.float()
        v_k = lv.float().clamp(-14, 14)
        # RAW data units, matching baselines._gaussian_nll. Rescaling by the
        # target's own per-window std would compare densities of two DIFFERENT
        # random variables (NLL_scaled = NLL_raw - log s).
        nll_el = 0.5 * (math.log(2 * math.pi) + v_k + (y - m_k) ** 2 * torch.exp(-v_k))
        return nll_el, m_k

    for batch in loader:
        eeg = batch["eeg"].to(trainer.device)
        ctx_e, tgt_e = eeg[:, :c], eeg[:, c:]
        src = trainer.sensor_to_parcel(ctx_e)
        src = src / src.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        y = tgt_e.float()
        n_elem = float(y.shape[1] * y.shape[2])
        s_ = tgt_e.std(dim=(1, 2), keepdim=True).clamp_min(1e-8).float()

        # -- HEADLINE: plug-in at the posterior mean ------------------------
        # Sampling theta is cheap (one posterior forward); only the rollout is
        # expensive. So the mean is estimated from many draws and spent on ONE
        # rollout, which makes it stable and, with the evaluation seeded,
        # deterministic. It is estimated, not exact -- see `theta_mean_samples`.
        th_bar = trainer.posterior.sample(ctx_e, n_mean_samples).mean(dim=1)
        th_bar = th_bar[:, : len(THETA_NAMES)]
        # B5a: train and eval must run the SAME forward pass. train.real_losses
        # applies the individualizer; this did not, so an individualised
        # checkpoint and a population checkpoint produced the same number.
        _ind = getattr(trainer, "individualizer", None)
        if _ind is not None:
            _pid = trainer.participant_index(list(batch.get("subject", [])))
            th_bar = _ind(participant=_pid, base=th_bar)
            seen_pids.update(int(v) for v in _pid.detach().cpu().tolist())
        nll_el, m_bar = _score(th_bar, y)
        nll_pw = nll_el.mean(dim=(1, 2))
        nlls.append(nll_pw.cpu().numpy())
        mses.append(((y - m_bar) ** 2).mean(dim=(1, 2)).cpu().numpy())
        nlls_norm.append((nll_pw - torch.log(s_).reshape(-1)).cpu().numpy())

        # -- SECONDARY: marginal over K draws, with diagnostics -------------
        if n_theta_samples and n_theta_samples > 0:
            th_all = trainer.posterior.sample(ctx_e, n_theta_samples)
            joint_ll: list[Tensor] = []
            for k in range(n_theta_samples):
                nll_k, _ = _score(th_all[:, k][:, : len(THETA_NAMES)], y)
                joint_ll.append(-nll_k.sum(dim=(1, 2)))
            L = torch.stack(joint_ll, dim=1)  # (B, K) log p(y | theta_k)
            logp = torch.logsumexp(L, dim=1) - math.log(float(n_theta_samples))
            marg.append((-logp / n_elem).cpu().numpy())
            # K/2 -> K drift: if the marginal has converged this is ~0.
            half = max(1, n_theta_samples // 2)
            logp_h = torch.logsumexp(L[:, :half], dim=1) - math.log(float(half))
            marg_half.append((-logp_h / n_elem).cpu().numpy())
            # Self-normalised effective sample size of the K likelihood weights.
            w = torch.softmax(L, dim=1)
            ess.append((1.0 / (w**2).sum(dim=1)).cpu().numpy())
            top_mass.append(w.max(dim=1).values.cpu().numpy())

        subs.extend(list(batch["subject"]))
    model.train()

    out: dict[str, Any] = {
        "nll_per_window": np.concatenate(nlls) if nlls else np.zeros(0),
        "mse_per_window": np.concatenate(mses) if mses else np.zeros(0),
        "nll_per_window_amplitude_normalised": (
            np.concatenate(nlls_norm) if nlls_norm else np.zeros(0)
        ),
        "subjects": subs,
        "estimator": "plug-in at posterior mean (matches the baselines' plug-in form)",
        "theta_mean_samples": int(n_mean_samples),
        "units_note": (
            "nll_per_window and mse_per_window are in RAW data units, matching "
            "baselines._gaussian_nll. nll_per_window_amplitude_normalised divides by "
            "the target's own per-window std and is NOT comparable to the baselines."
        ),
    }
    # The null case must WRITE SOMETHING. An individualiser that is applied but
    # sits at initialisation for every scored participant is indistinguishable in
    # the score from one that was never applied; these fields make the difference
    # visible in the output instead of inferable only from a checkpoint diff.
    _ind = getattr(trainer, "individualizer", None)
    if _ind is None:
        out["individualization"] = {
            "applied": False,
            "reason": "no individualizer on the trainer (population model)",
        }
    else:
        z = _ind.state_dict().get("z_person")
        fitted = 0
        at_init = 0
        if z is not None:
            for i in sorted(seen_pids):
                if i >= z.shape[0]:
                    continue
                if float(z[i].abs().max()) > 0.0:
                    fitted += 1
                else:
                    at_init += 1
        out["individualization"] = {
            "applied": True,
            "n_participants_scored": len(seen_pids),
            "n_individualised_participants": int(fitted),
            "n_at_initialisation": int(at_init),
            "note": (
                "n_at_initialisation counts scored participants whose person effect "
                "was never fitted. Their z_person row is exactly zero, so "
                "individualization contributes nothing for them and any G5-style "
                "claim over them is measuring the population model. A participant-"
                "disjoint holdout puts EVERY test participant in this column."
            ),
        }
    if marg:
        m = np.concatenate(marg)
        mh = np.concatenate(marg_half)
        e = np.concatenate(ess)
        tm = np.concatenate(top_mass)
        out["nll_per_window_marginal"] = m
        out["marginal_diagnostics"] = {
            "K": int(n_theta_samples),
            "ess_median": float(np.median(e)),
            "ess_mean": float(e.mean()),
            "frac_windows_ess_below_2": float((e < 2.0).mean()),
            "top_draw_weight_median": float(np.median(tm)),
            "drift_khalf_to_k_nats": float(m.mean() - mh.mean()),
            "note": (
                "SECONDARY, NOT COMPARABLE TO THE BASELINES: they are plug-in "
                "estimators. Also not a converged predictive on this posterior -- "
                "with ESS near 1 the logsumexp is effectively a best-of-K, and the "
                "estimate drifts with K rather than converging. Report K, ESS and "
                "the drift with any use of this number."
            ),
        }
    return out



def _window_subject(ds, idx: int) -> str:
    """Subject id of one window, from metadata only (no signal I/O)."""
    rec_idx, _ = ds.window_index[int(idx)]
    return str(ds.recordings[rec_idx]["subject"])


def _participant_stratified(ds, split_indices, per_participant: int, *, fold: str) -> list[int]:
    """Evenly spaced windows per participant.

    **The budget is fixed by participants, not batches.** A window budget
    (`max_batches`) re-creates the one-participant-per-side defect the moment the
    corpus grows or the fold ordering changes: 40 batches of 16 drew 640 windows
    from participant-ordered folds of ~2,650, so every baseline was fit on one
    person and every model scored on one different person.
    """
    by_sub: dict[str, list[int]] = defaultdict(list)
    for i in split_indices:
        by_sub[_window_subject(ds, i)].append(int(i))
    if len(by_sub) < 2:
        raise ValueError(
            f"{fold} fold resolves to {len(by_sub)} participant(s); a "
            "participant-clustered interval is undefined below 2 and the "
            "comparison would not be a holdout."
        )
    out: list[int] = []
    for sub in sorted(by_sub):
        idxs = by_sub[sub]
        k = int(min(per_participant, len(idxs)))
        sel = np.linspace(0, len(idxs) - 1, k).round().astype(int)
        out.extend(idxs[j] for j in dict.fromkeys(sel.tolist()))
    return out


def split_fingerprint(ds, split: Mapping[str, Sequence[int]]) -> dict[str, Any]:
    """Participant **ids** per fold plus a sha256 over them.

    Ids rather than indices: indices are meaningless if the corpus is rebuilt, so
    an index-based fingerprint would silently pass on a different dataset.
    """
    folds = {
        name: sorted({_window_subject(ds, i) for i in idxs}) for name, idxs in split.items()
    }
    blob = json.dumps(folds, sort_keys=True).encode()
    # `verified` defaults to False and is flipped only by an actual comparison
    # against a recorded fingerprint. A reader of evaluation.json must not be able
    # to mistake a recomputed sha256 for a checked one: an authoritative-looking
    # hash that was never compared is the failure this field exists to prevent.
    return {
        "participants_per_fold": folds,
        "sha256": hashlib.sha256(blob).hexdigest(),
        "verified": False,
        "verification": (
            "RECOMPUTED ONLY, NOT VERIFIED: no recorded fingerprint was compared "
            "against. This split has not been proven identical to the one that "
            "trained the checkpoint."
        ),
    }


def real_eeg_holdout(
    trainer,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    n_theta_samples: int = 32,
    n_mean_samples: int = 256,
    per_test_participant: int = 40,
    per_train_participant: int = 30,
) -> dict[str, Any]:
    """SC-WBD vs every required baseline on a participant-level holdout.

    Windows within a participant are correlated, so the interval is a **cluster
    bootstrap over participants**.  A window-level interval here would be a
    silent overstatement of precision, which is the exact failure Appendix D's
    "Participant or family leakage" row is about.
    """
    from .baselines import (
        ARBaseline,
        _paired_ci,
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
    ds = trainer.real_dataset
    split = trainer.real_split

    # B4: the split is rebuilt at evaluation time and must be proven identical to
    # the one that trained the checkpoint. Participant IDS, not indices.
    fp = split_fingerprint(ds, split)
    recorded = getattr(trainer, "_recorded_split_fingerprint", None)
    if recorded is None:
        # Carried in the ARTIFACT, not printed to stdout: evaluate_model() and
        # real_eeg_holdout() are both public and bypass main()'s warning.
        fp["verification"] = (
            "NOT VERIFIED: the checkpoint records no real_split fingerprint (written "
            "before the field existed). The evaluation split CANNOT be proven "
            "identical to the one that trained this checkpoint, and every number "
            "below rests on that unproven assumption."
        )
    elif recorded.get("sha256") != fp["sha256"]:
        raise RuntimeError(
            "real-EEG split does not match the checkpoint's: recorded sha256 "
            f"{recorded.get('sha256')}, recomputed {fp['sha256']}. Evaluating would "
            "score a model on participants it may have trained on."
        )
    else:
        fp["verified"] = True
        fp["verification"] = "verified against the fingerprint recorded in the checkpoint"

    # B1: budget in PARTICIPANTS, not batches.
    te_idx = _participant_stratified(ds, split["test"], per_test_participant, fold="test")
    tr_idx = _participant_stratified(ds, split["train"], per_train_participant, fold="train")
    bs = max(8, cfg.data.batch // 4)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, te_idx), batch_size=bs, shuffle=False, num_workers=2
    )
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, tr_idx), batch_size=bs, shuffle=False, num_workers=2
    )

    def collect(loader):
        xs, ss = [], []
        for b in loader:
            xs.append(b["eeg"])
            ss.extend(list(b["subject"]))
        return (torch.cat(xs) if xs else torch.zeros(0), ss)

    tr_x, tr_s = collect(train_loader)
    te_x, te_s = collect(test_loader)
    if te_x.numel() == 0:
        return {"available": False, "reason": "empty holdout"}
    dev = trainer.device
    tr_x, te_x = tr_x.to(dev), te_x.to(dev)
    ctx, tgt = te_x[:, :c], te_x[:, c:]

    scw = _scwbd_scores(
        trainer,
        test_loader,
        n_theta_samples=n_theta_samples,
        n_mean_samples=n_mean_samples,
    )
    n_model_params = sum(p.numel() for p in trainer.model.parameters())

    models: dict[str, Any] = {
        "persistence": PersistenceBaseline(),
        "ar16": ARBaseline(order=16),
        "var4": VARBaseline(order=4),
        "population_gaussian": PopulationGaussianBaseline(),
        "subject_specific_ar": SubjectSpecificBaseline(
            ARBaseline, base_kwargs={"order": 16}, name="subject_specific_ar"
        ),
        "dense_neural": DenseNeuralBaseline(target_parameters=n_model_params, steps=400, seed=seed),
    }
    rows: dict[str, Any] = {}
    per_window: dict[str, np.ndarray] = {}
    per_window_mse: dict[str, np.ndarray] = {}
    for name, m in models.items():
        t0 = time.time()
        try:
            m.fit(tr_x, groups=tr_s)
            r = m.score(ctx, tgt, groups=te_s) if _accepts_groups(m.score) else m.score(ctx, tgt)
            per = np.asarray(r.get("nll_per_window", r.get("per_window_nll", [])), dtype=float)
            pt, lo, hi = bootstrap_ci(per, np.asarray(te_s), n_boot=n_boot, seed=seed) if per.size else (r["nll_per_sample"], float("nan"), float("nan"))
            # The MSE gets the same treatment as the NLL. Discarding the
            # `per_window_mse` that `Baseline.score` already returns left the MSE
            # a point estimate, and a point estimate is not a claim -- which is
            # why run 1's report could not state that SC-WBD has the LOWEST MSE
            # of all seven arms. It does, decisively, against all six baselines.
            per_m = np.asarray(r.get("per_window_mse", r.get("mse_per_window", [])), dtype=float)
            m_pt, m_lo, m_hi = (
                bootstrap_ci(per_m, np.asarray(te_s), n_boot=n_boot, seed=seed)
                if per_m.size
                else (r.get("mse", float("nan")), float("nan"), float("nan"))
            )
            rows[name] = {
                "nll_per_sample": float(r["nll_per_sample"]),
                "nll_ci95": [float(lo), float(hi)],
                "mse": float(r.get("mse", float("nan"))),
                "mse_ci95": [float(m_lo), float(m_hi)],
                "n_parameters": int(_nparams(m)),
                "fit_seconds": round(time.time() - t0, 2),
                "describe": m.describe() if hasattr(m, "describe") else {},
            }
            per_window[name] = per
            per_window_mse[name] = per_m
        except Exception as exc:  # noqa: BLE001 - a baseline that cannot run is reported, not hidden
            rows[name] = {"error": f"{type(exc).__name__}: {exc}"}

    pt, lo, hi = bootstrap_ci(scw["nll_per_window"], np.asarray(scw["subjects"]), n_boot=n_boot, seed=seed)
    m_pt, m_lo, m_hi = bootstrap_ci(
        scw["mse_per_window"], np.asarray(scw["subjects"]), n_boot=n_boot, seed=seed
    )
    _eeg_head = trainer.model.eeg
    # The model's own arm name, derived like model_id.  It was the literal
    # "scwbd_001_beta", so a run-2 evaluation ranked its own arm under a run-1
    # name -- the same defect as the hardcoded model_id, one layer down, and in
    # the very table the model card reads.
    arm = _designation(trainer.cfg)
    rows[arm] = {
        "nll_per_sample": float(np.mean(scw["nll_per_window"])),
        "nll_ci95": [float(lo), float(hi)],
        "mse": float(np.mean(scw["mse_per_window"])),
        "mse_ci95": [float(m_lo), float(m_hi)],
        "n_parameters": int(n_model_params),
        # Run 1's `describe()` had three keys and none of them said how the
        # predictive variance was arrived at, while all six baselines reported
        # `variance_calibration`. The two arms with no HELD-OUT calibration were
        # exactly the two with positive excess NLL over the entropy floor. An
        # arm that does not declare its variance calibration cannot be compared
        # on a log score, so it declares it.
        "describe": {
            "name": _designation(cfg),
            "structured_state": True,
            "connectome_masked": True,
            "variance_calibration": (
                "state-dependent: sourced from X^uncertainty via the observation "
                "interface, plus a separately parameterised per-channel instrument floor"
                if getattr(_eeg_head, "state_dependent_variance", False)
                else "NONE: broadcast per-channel constant, never calibrated on held-out "
                "data. Not comparable on a log score with the held-out per-(horizon,"
                "channel) calibration every baseline receives."
            ),
            "noise_floor": _eeg_head.noise_floor_report()
            if hasattr(_eeg_head, "noise_floor_report")
            else {},
        },
    }
    per_window_mse[arm] = np.asarray(scw["mse_per_window"], dtype=float)
    # B6: the verdict rests on the PAIRED participant-clustered interval of the
    # per-window difference, not on two marginal intervals or two point estimates.
    # `_paired_ci` shares bootstrap draws across models, so the comparison is
    # paired in the draws as well as in the windows.
    ref = rows[arm]["nll_per_sample"]
    scw_pw = np.asarray(scw["nll_per_window"], dtype=float)
    groups = np.asarray(scw["subjects"])
    paired: dict[str, Any] = {}
    beaten_by: list[str] = []
    inconclusive: list[str] = []
    for name, per in per_window.items():
        if per.size != scw_pw.size:
            paired[name] = {"error": f"length mismatch {per.size} vs {scw_pw.size}"}
            continue
        # positive delta => SC-WBD scores WORSE (higher NLL) than the baseline
        d = _paired_ci(scw_pw - per, groups, n_boot=n_boot, alpha=0.05, seed=seed)
        paired[name] = d
        if d["excludes_zero"] and d["delta"] > 0:
            beaten_by.append(name)
        elif not d["excludes_zero"]:
            inconclusive.append(name)

    # The same paired, participant-clustered treatment for the CONDITIONAL MEAN.
    # NLL and MSE answer different questions and run 1 only reported one of them,
    # which is how a model with the best conditional mean of all seven arms was
    # filed as beaten by five of six.
    scw_mse_pw = per_window_mse.get(arm)
    paired_mse: dict[str, Any] = {}
    mse_beaten_by: list[str] = []
    mse_better_than: list[str] = []
    if scw_mse_pw is not None and scw_mse_pw.size:
        for name, per_m in per_window_mse.items():
            if name == arm or per_m.size != scw_mse_pw.size:
                continue
            d = _paired_ci(scw_mse_pw - per_m, groups, n_boot=n_boot, alpha=0.05, seed=seed)
            paired_mse[name] = d
            if d["excludes_zero"]:
                (mse_beaten_by if d["delta"] > 0 else mse_better_than).append(name)
    ranking = sorted(
        [(k, v["nll_per_sample"]) for k, v in rows.items() if "nll_per_sample" in v], key=lambda kv: kv[1]
    )
    return {
        "available": True,
        "n_test_windows": int(te_x.shape[0]),
        "n_test_participants": int(len(set(te_s))),
        "n_train_windows": int(tr_x.shape[0]),
        "n_train_participants": int(len(set(tr_s))),
        "windows_per_test_participant": int(per_test_participant),
        "windows_per_train_participant": int(per_train_participant),
        "sampling": (
            "participant-stratified, evenly spaced within participant; budget fixed "
            "by participants, not batches"
        ),
        "real_split": fp,
        "individualization": scw.get("individualization"),
        "metric": "gaussian NLL, nats per channel per sample, sensor space, participant-clustered 95% CI",
        "estimator": scw.get("estimator"),
        "results": rows,
        "ranking_best_first": ranking,
        "paired_vs_scwbd": paired,
        "scwbd_beaten_by": beaten_by,
        "inconclusive_vs_scwbd": inconclusive,
        "paired_mse_vs_scwbd": paired_mse,
        "scwbd_mse_beaten_by": mse_beaten_by,
        "scwbd_mse_better_than": mse_better_than,
        "mse_interpretation": (
            "Paired participant-clustered 95% interval of the per-window MSE difference "
            "(scwbd minus baseline; negative favours SC-WBD). This is the CONDITIONAL "
            "MEAN, scored separately from the NLL on purpose: the two can and do "
            "disagree. Run 1 reported only the NLL and therefore filed a model with the "
            "lowest MSE of all seven arms as beaten by five of six. The difference "
            "between the two verdicts is the predictive variance, and it is a defect in "
            "the variance channel rather than in the forecast."
        ),
        # Derived, never a literal.  This read "SC-WBD-001-beta" on both
        # branches -- and `verdict` is the single most quoted string on the
        # public model card, so a run-2 evaluation would have announced its
        # result under the run-1 name in the most visible place available.
        "verdict": (
            f"{_designation(trainer.cfg)} is beaten by "
            + ", ".join(beaten_by)
            + " on the paired participant-clustered 95% interval of the per-window NLL difference"
            if beaten_by
            else f"No baseline beats {_designation(trainer.cfg)} on the paired "
            "participant-clustered 95% interval of the per-window NLL difference"
        ),
        "interpretation": (
            "The verdict is the PAIRED interval, not the ranking: two overlapping marginal "
            "intervals can still admit a decisive paired difference, and a lower point estimate "
            "is not a claim. Models listed in `inconclusive_vs_scwbd` are neither better nor "
            "worse at this sample size. Window-level generalisation is not individual "
            "generalisation."
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

def _sim_backend_labels(ds) -> list[str]:
    """Backend name per window, from the corpus index (metadata only)."""
    by_path = {sh["path"]: sh["backend"] for sh in ds.index.shards}
    return [by_path[it[0]] for it in ds.items]


def _sim_stratified(
    ds, *, mode: str, total: int | None = None, per_backend: int | None = None,
    min_per_backend: int = 30, require_all: bool = False, caller: str = "",
) -> tuple[list[int], dict[str, int]]:
    """Backend-stratified window selection over a simulated corpus.

    Taking the first N windows from a `shuffle=False` loader is backend-biased:
    the val corpus is ordered by shard, so the first 512 of 1888 contained ZERO
    samples from two of the five backends.  `mode="equal"` gives every backend the
    same weight (for a per-backend table); `mode="proportional"` preserves the
    corpus mixture (for a single pooled number), with a floor so a rare backend
    still contributes a usable count.
    """
    labels = _sim_backend_labels(ds)
    by: dict[str, list[int]] = defaultdict(list)
    for i, b in enumerate(labels):
        by[b].append(i)
    if require_all:
        missing = [n for n in ds.backend_names if not by.get(n)]
        if missing:
            raise ValueError(
                f"{caller}: backends {missing} have zero windows in this fold. This "
                "function exists to produce a per-backend row; emitting None for a "
                "backend is honest and insufficient, because a reader cannot tell a "
                "backend that failed from one that was never sampled."
            )
    chosen: list[int] = []
    counts: dict[str, int] = {}
    for name in sorted(by):
        idxs = by[name]
        if mode == "equal":
            k = min(int(per_backend or min_per_backend), len(idxs))
        else:
            share = int(round((total or 0) * len(idxs) / max(len(labels), 1)))
            k = min(max(share, min_per_backend), len(idxs))
        sel = np.linspace(0, len(idxs) - 1, max(k, 1)).round().astype(int)
        picked = [idxs[j] for j in dict.fromkeys(sel.tolist())]
        chosen.extend(picked)
        counts[name] = len(picked)
    return chosen, counts


def posterior_calibration(trainer, *, n_datasets: int = 512, n_samples: int = 256) -> dict[str, Any]:
    """SBC + expected coverage on held-out simulated trajectories."""
    from .posterior import posterior_report

    idx, counts = _sim_stratified(
        trainer.sim_val, mode="proportional", total=n_datasets, caller="posterior_calibration"
    )
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(trainer.sim_val, idx), batch_size=64, shuffle=False, num_workers=2
    )
    ys, ths = [], []
    c = trainer.cfg.data.context
    for b in loader:
        ys.append(b["activity"][:, :c].to(trainer.device))
        ths.append(b["theta"].to(trainer.device))
    if not ys:
        return {"available": False, "reason": "no simulated validation trajectories"}
    y = torch.cat(ys)
    th = torch.cat(ths)
    trainer.posterior.eval()
    rep = posterior_report(trainer.posterior, y, th, param_names=THETA_NAMES, n_samples=n_samples)
    trainer.posterior.train()
    rep["available"] = True
    rep["n_datasets"] = int(y.shape[0])
    rep["backend_counts"] = counts
    rep["sampling"] = "backend-stratified, fold-proportional with a floor per backend"
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
def _sim_val_nll(trainer, *, n_windows: int = 512) -> float:
    idx, _ = _sim_stratified(
        trainer.sim_val, mode="proportional", total=n_windows, caller="_sim_val_nll"
    )
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(trainer.sim_val, idx), batch_size=64, shuffle=False, num_workers=2
    )
    c = trainer.cfg.data.context
    tot, n = 0.0, 0
    trainer.model.eval()
    for b in loader:
        act = b["activity"].to(trainer.device)
        th = b["theta"].to(trainer.device)
        ctx, tgt = act[:, :c], act[:, c:]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=trainer.cfg.model.use_bf16):
            _bind_mechanistic(trainer.model, th)
            r = trainer.model.rollout(y_context=ctx, theta=th, n_steps=tgt.shape[1], enforce_r05=False)
        tot += float(gaussian_nll(tgt, r.activity.float(), r.activity_logvar.float())) * act.shape[0]
        n += act.shape[0]
    trainer.model.train()
    return tot / max(n, 1)


# ======================================================================
# backend comparison
# ======================================================================
@torch.no_grad()
def backend_comparison(trainer, *, per_backend: int = 64) -> dict[str, Any]:
    """Per-backend held-out forecast NLL of the learned operator.

    The simulated validation set carries a backend label, so this reports where
    the single learned operator succeeds and fails **across mechanistic
    families**.  It is not a claim that any family is neurally realized.
    """
    idx, counts = _sim_stratified(
        trainer.sim_val, mode="equal", per_backend=per_backend,
        require_all=True, caller="backend_comparison",
    )
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(trainer.sim_val, idx), batch_size=64, shuffle=False, num_workers=2
    )
    c = trainer.cfg.data.context
    names = trainer.sim_val.backend_names
    acc: dict[str, list[float]] = {n: [] for n in names}
    trainer.model.eval()
    for b in loader:
        act = b["activity"].to(trainer.device)
        th = b["theta"].to(trainer.device)
        ctx, tgt = act[:, :c], act[:, c:]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=trainer.cfg.model.use_bf16):
            _bind_mechanistic(trainer.model, th)
            r = trainer.model.rollout(y_context=ctx, theta=th, n_steps=tgt.shape[1], enforce_r05=False)
        lv = r.activity_logvar.float().clamp(-14, 14)
        nll = (0.5 * (math.log(2 * math.pi) + lv + (tgt - r.activity.float()) ** 2 * torch.exp(-lv))).mean(dim=(1, 2))
        for j, bi in enumerate(b["backend"].tolist()):
            acc[names[bi]].append(float(nll[j]))
    trainer.model.train()
    return {
        "per_backend_nll": {k: (float(np.mean(v)) if v else None) for k, v in acc.items()},
        "per_backend_n": {k: len(v) for k, v in acc.items()},
        "sampling": "backend-stratified, equal windows per backend",
        "windows_selected_per_backend": counts,
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
        "model_id": _designation(trainer.cfg),
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
    rep["backend_comparison"] = backend_comparison(trainer, per_backend=16 if quick else 64)
    if quick:
        # A cost flag may reduce precision; it must never redefine the claim.
        # A shrunken holdout is not a cheaper version of this result, it is a
        # different and unstated one.
        rep["real_eeg_holdout"] = {
            "available": False,
            "reason": (
                "--quick refuses the real-EEG holdout. A reduced-cost variant would "
                "silently change the participant set the claim rests on."
            ),
        }
    else:
        rep["real_eeg_holdout"] = real_eeg_holdout(trainer, seed=seed)
    rep["sim_val_nll"] = _sim_val_nll(trainer, n_windows=128 if quick else 512)
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

    p = argparse.ArgumentParser(description="evaluate an SC-WBD checkpoint honestly")
    p.add_argument("--config", default="configs/scwbd_001_beta.yaml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out", default="reports/training/evaluation.json")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ablate-sources", action="store_true")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    tr = FoundationTrainer(cfg, resume=False, quick=a.quick)
    # Data first: the individualizer's row count comes from the participant list.
    tr.build_data()
    ckpt = a.checkpoint or str(Path(cfg.train.out_dir) / "last.pt")
    if Path(ckpt).exists():
        from .checkpoint import CheckpointError, load_checkpoint

        # B5a: an individualised checkpoint must be loaded WITH its individualizer,
        # or the evaluation silently scores the population model.
        import torch as _torch

        _peek = _torch.load(ckpt, map_location="cpu", weights_only=False)
        if _peek.get("individualizer") is not None and tr.individualizer is None:
            from .individual import Individualizer

            _np = max(len(tr._participant_ids()), 1)
            tr.individualizer = Individualizer(
                len(THETA_NAMES), n_groups=2, n_participants=_np, n_sessions=max(_np * 4, 1)
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
        # `strict=False` is needed because the posterior/individualizer may be
        # absent, but load_checkpoint RECORDS what it dropped and the caller used
        # to discard it. torch.compile prefixes `local._orig_mod.*`, and the model
        # is compiled only when `cfg.model.compile and device.type == "cuda"` -- so
        # evaluating on CPU silently dropped 80.2% of the parameter mass (all of
        # `local` + `residual`) and scored RANDOM WEIGHTS while printing "loaded".
        report = payload.get("load_report", {})
        missing = [
            *report.get("missing", []),
            *report.get("posterior_missing", []),
            *report.get("individualizer_missing", []),
        ]
        unexpected = [
            *report.get("unexpected", []),
            *report.get("posterior_unexpected", []),
            *report.get("individualizer_unexpected", []),
        ]
        # Absence is not a mismatch, but it must not read as a clean load either:
        # that is the silent-load failure one level up. Loud, recorded, not fatal --
        # a population checkpoint legitimately carries no individualizer.
        for _m in ("posterior", "individualizer"):
            if report.get(f"{_m}_absent"):
                print(
                    f"[warn] {ckpt} contains no {_m}: it was NOT loaded, and any "
                    f"metric that depends on it describes a model without one.",
                    flush=True,
                )
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
        # B4: hand the checkpoint's recorded split to the holdout for verification.
        tr._recorded_split_fingerprint = (payload.get("extra") or {}).get("real_split")
        if tr._recorded_split_fingerprint is None:
            print(
                f"[warn] {ckpt} records no real_split fingerprint (written before the "
                "field existed): the evaluation split CANNOT be proven identical to "
                "the one that trained it.",
                flush=True,
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
