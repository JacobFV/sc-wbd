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

from contextlib import nullcontext

from .families import layout_of_checkpoint

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

__all__ = ["evaluate_model", "real_eeg_holdout", "posterior_calibration", "source_ablation", "session_individualisation", "main"]


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
    source_id: str = "eegmmidb_real",
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
    # Both the projector and the observation head are per-MONTAGE. Hardcoding
    # `trainer.sensor_to_parcel` and `model.eeg` pinned this function to the
    # 64-channel founding montage, so calling it on sleep-EDFx's 2 channels
    # raised `einsum(): subscript c has size 2 ... does not broadcast with
    # previously seen size 64`. That is `train.eeg_projector`'s docstring
    # verbatim: sharing the 64-channel projector with a 2-channel source is a
    # shape error, and sharing it with a DIFFERENT 64-channel montage is worse,
    # because it projects through the wrong geometry and raises nothing.
    projector = trainer.eeg_projector(source_id)
    eeg_head = model.eeg_head_for(source_id)
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
            mu, lv = eeg_head(roll.state)
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
        src = projector(ctx_e)
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

    ``policy`` names the participant-assignment rule that produced these folds
    (``realdata.SPLIT_POLICIES``).  It is reported beside the ids and **not**
    folded into the sha256, deliberately: the sha256 is the thing three released
    checkpoints already recorded, and mixing a new field into it would make every
    one of them fail to verify against its own split.  The ids are what the
    verification rests on; the policy says how they came about.
    """
    folds = {
        name: sorted({_window_subject(ds, i) for i in idxs}) for name, idxs in split.items()
    }
    blob = json.dumps(folds, sort_keys=True).encode()
    policy = getattr(ds, "participant_split_policy", "unknown")
    # `verified` defaults to False and is flipped only by an actual comparison
    # against a recorded fingerprint. A reader of evaluation.json must not be able
    # to mistake a recomputed sha256 for a checked one: an authoritative-looking
    # hash that was never compared is the failure this field exists to prevent.
    return {
        "participants_per_fold": folds,
        "sha256": hashlib.sha256(blob).hexdigest(),
        "policy": policy,
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
            "score a model on participants it may have trained on. Recorded split "
            f"policy {recorded.get('policy', 'not recorded')!r}, this evaluation "
            f"rebuilt with {fp['policy']!r}: if those differ, the config being "
            "evaluated declares a different data.split_policy from the one the "
            "checkpoint trained under."
        )
    elif recorded.get("policy") not in (None, "unknown", fp["policy"]):
        # The ids matched under two different policies. That is possible on a
        # small roster and it is still wrong to proceed: the agreement is a
        # coincidence of this corpus, not a property of the split, and the next
        # participant to appear breaks it.
        raise RuntimeError(
            f"real-EEG participant ids match but the split POLICY does not: the "
            f"checkpoint recorded {recorded.get('policy')!r} and this evaluation "
            f"rebuilt with {fp['policy']!r}. The folds agree on this roster by "
            "coincidence; set data.split_policy to the recorded value."
        )
    else:
        fp["verified"] = True
        fp["verification"] = (
            "verified against the fingerprint recorded in the checkpoint"
            + (
                ""
                if recorded.get("policy")
                else " (which predates the `policy` field, so the split POLICY is "
                "unverified; the participant ids are what matched)"
            )
        )

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

    # `subject_specific_ar` IS NOT HERE, and its absence is the fix for
    # ISSUE-013. Fitted on the train participants and scored on the test
    # participants -- which R10 makes disjoint -- every scored window missed its
    # own model and was served by the pooled fallback, which at the same order
    # and the same seed is bit-for-bit `ar16`. It was a duplicate row carrying
    # the name of the hardest baseline the thesis names.
    #
    # A subject-specific baseline and a participant-disjoint holdout are in
    # direct conflict, so it is not repairable here: it needs a within-
    # participant temporal split, which is a different quantity from every row
    # in this table because it has SEEN the scored participant. That quantity is
    # measured, separately and under its own name, by
    # `within_participant_holdout`.
    models: dict[str, Any] = {
        "persistence": PersistenceBaseline(),
        "ar16": ARBaseline(order=16),
        "var4": VARBaseline(order=4),
        "population_gaussian": PopulationGaussianBaseline(),
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
        "baseline_protocol": "v2_no_pooled_subject_specific",
        # A row that disappears without a record is indistinguishable from a row
        # that was never there. This says which one it is, and why.
        "dropped_baselines": {
            "subject_specific_ar": (
                "DROPPED from this table under baseline protocol v2 (ISSUE-013). "
                "Refusal R10 makes the fit and score participant sets disjoint, so "
                "100% of scored windows routed to the pooled fallback and the row "
                "was bit-for-bit `ar16` -- a duplicate carrying the name of the "
                "hardest baseline the thesis names. Protocol v1 (runs 1-3) "
                "reported it; those numbers stand as `ar16`'s. The quantity it was "
                "supposed to measure is measured instead by "
                "`within_participant_holdout`, on a within-participant temporal "
                "split, and is NOT comparable with the rows here: it has seen the "
                "scored participant and every row here has not."
            )
        },
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


def _within_participant_temporal_split(
    ds,
    test_indices: Sequence[int],
    *,
    fit_fraction: float,
    gap_windows: int,
    per_participant_fit: int,
    per_participant_score: int,
) -> tuple[dict[str, list[int]], dict[str, list[int]], list[dict[str, Any]]]:
    """Per test participant: their earlier windows to fit on, later ones to score.

    The ordering is the corpus's own window order, which for every EEG source
    here is ``(recording, window within recording)`` -- and eegmmidb's recordings
    are discovered subject-major and sorted by run label ``R01``..``R14``, so
    ascending window index is ascending time for one person.

    ``gap_windows`` windows are dropped between the two halves.  Windows are
    non-overlapping by construction (``RealEEGConfig.window_stride_s = None``
    means hop == window), but the last fit window and the first scored window are
    still adjacent in time, and an AR fitted right up to the boundary is scored
    on the continuation of its own training signal.  The gap is what makes the
    scored future a future rather than a seam.

    Returns ``(fit_by_subject, score_by_subject, skipped)``.  A participant with
    too few windows on either side is **skipped and recorded**, never quietly
    served from the other half.
    """
    by_sub: dict[str, list[int]] = defaultdict(list)
    for i in test_indices:
        by_sub[_window_subject(ds, i)].append(int(i))

    fit_by: dict[str, list[int]] = {}
    score_by: dict[str, list[int]] = {}
    skipped: list[dict[str, Any]] = []
    for sub in sorted(by_sub):
        idxs = sorted(by_sub[sub])
        cut = int(round(fit_fraction * len(idxs)))
        early, late = idxs[:cut], idxs[cut + gap_windows :]
        if len(early) < per_participant_fit or len(late) < 2:
            skipped.append(
                {
                    "subject": sub,
                    "n_windows": len(idxs),
                    "n_earlier": len(early),
                    "n_later_after_gap": len(late),
                    "reason": (
                        f"needs at least {per_participant_fit} earlier windows to fit "
                        "a per-participant model and 2 later ones to score it"
                    ),
                }
            )
            continue
        fit_by[sub] = _evenly_spaced(early, per_participant_fit)
        score_by[sub] = _evenly_spaced(late, per_participant_score)
    return fit_by, score_by, skipped


def _evenly_spaced(pool: Sequence[int], k: int) -> list[int]:
    """``k`` evenly spaced entries of ``pool``, deduplicated, order preserved."""
    k = int(min(k, len(pool)))
    if k <= 0:
        return []
    sel = np.linspace(0, len(pool) - 1, k).round().astype(int)
    return [pool[j] for j in dict.fromkeys(sel.tolist())]


def within_participant_holdout(
    trainer,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    fit_fraction: float = 0.5,
    gap_windows: int = 8,
    per_participant_fit: int = 60,
    per_participant_score: int = 40,
    n_mean_samples: int = 256,
) -> dict[str, Any]:
    """The subject-specific baseline, on a split that can actually run it.

    This is the arm ``real_eeg_holdout`` no longer reports (ISSUE-013).  There it
    was fitted on the *train* participants and scored on the *test* participants,
    which R10 makes disjoint, so every scored window fell through to the pooled
    fallback and the row was bit-for-bit ``ar16``.

    Here the split is **within participant and temporal**, inside the held-out
    fold: each test participant's own earlier windows fit their own AR(16), a gap
    is dropped, and their later windows are scored.  The model never trained on
    any of these people, so the outer holdout is intact; what changes is that the
    *baseline* has seen the scored participant.

    **This is a different quantity from every row of the main table and must
    never be pooled with them.** It answers the question the thesis actually
    asks: does a model fitted on this person alone predict this person's future
    as well as a model pretrained across other people?  If it does, cross-
    participant pretraining transferred nothing.

    ``ar16_pooled`` -- an AR(16) fitted on the *train* fold, scored on the same
    later windows -- is reported beside it, because "your own past" is only
    interesting against "everyone else's data" on identical windows.

    ``n_theta_samples`` is not exposed and the marginal is not computed: the
    headline everywhere in this file is plug-in at the posterior mean, the
    marginal is a labelled secondary of the main holdout, and computing it here
    would cost 33 rollouts per batch for a diagnostic that already exists.
    """
    from .baselines import (
        ARBaseline,
        SubjectSpecificBaseline,
        _paired_ci,
        bootstrap_ci,
    )

    if getattr(trainer, "real_test", None) is None or len(trainer.real_test) == 0:
        return {"available": False, "reason": "no measured EEG holdout available"}
    cfg = trainer.cfg
    c = cfg.data.context
    ds = trainer.real_dataset
    split = trainer.real_split

    fit_by, score_by, skipped = _within_participant_temporal_split(
        ds,
        split["test"],
        fit_fraction=fit_fraction,
        gap_windows=gap_windows,
        per_participant_fit=per_participant_fit,
        per_participant_score=per_participant_score,
    )
    if len(score_by) < 2:
        return {
            "available": False,
            "reason": (
                f"{len(score_by)} test participant(s) have enough windows for a "
                "within-participant temporal split; a participant-clustered "
                "interval is undefined below 2"
            ),
            "skipped_participants": skipped,
        }

    fit_idx = [i for sub in sorted(fit_by) for i in fit_by[sub]]
    score_idx = [i for sub in sorted(score_by) for i in score_by[sub]]
    bs = max(8, cfg.data.batch // 4)

    def _loader(idx: Sequence[int]):
        return torch.utils.data.DataLoader(
            torch.utils.data.Subset(ds, list(idx)),
            batch_size=bs,
            shuffle=False,
            num_workers=2,
        )

    def collect(loader):
        xs, ss = [], []
        for b in loader:
            xs.append(b["eeg"])
            ss.extend(list(b["subject"]))
        return (torch.cat(xs) if xs else torch.zeros(0), ss)

    score_loader = _loader(score_idx)
    fit_x, fit_s = collect(_loader(fit_idx))
    sc_x, sc_s = collect(score_loader)
    if sc_x.numel() == 0:
        return {"available": False, "reason": "empty within-participant score fold"}
    dev = trainer.device
    fit_x, sc_x = fit_x.to(dev), sc_x.to(dev)
    ctx, tgt = sc_x[:, :c], sc_x[:, c:]

    # The population reference is fitted on the TRAIN fold, exactly as in the
    # main holdout, so the contrast is "this person's own past" against "other
    # people's data" and nothing else differs.
    tr_idx = _participant_stratified(ds, split["train"], 30, fold="train")
    tr_x, tr_s = collect(_loader(tr_idx))
    tr_x = tr_x.to(dev)

    ss_model = SubjectSpecificBaseline(
        ARBaseline, base_kwargs={"order": 16}, name="subject_specific_ar_within"
    ).fit(fit_x, groups=np.asarray(fit_s))
    pooled = ARBaseline(order=16).fit(tr_x, np.asarray(tr_s))

    groups = np.asarray(sc_s)
    rows: dict[str, Any] = {}
    per_window: dict[str, np.ndarray] = {}
    for name, m in (("subject_specific_ar_within", ss_model), ("ar16_pooled", pooled)):
        r = m.score(ctx, tgt, groups) if _accepts_groups(m.score) else m.score(ctx, tgt)
        per = np.asarray(r.get("nll_per_window", r.get("per_window_nll", [])), dtype=float)
        _pt, lo, hi = bootstrap_ci(per, groups, n_boot=n_boot, seed=seed)
        rows[name] = {
            "nll_per_sample": float(r["nll_per_sample"]),
            "nll_ci95": [float(lo), float(hi)],
            "mse": float(r.get("mse", float("nan"))),
            "n_parameters": int(_nparams(m)),
            "describe": m.describe() if hasattr(m, "describe") else {},
        }
        per_window[name] = per

    routing = (rows["subject_specific_ar_within"]["describe"] or {}).get(
        "score_time_routing", {}
    )
    frac_fallback = float(routing.get("fraction_via_pooled_fallback", 1.0))
    if frac_fallback > 0.0:
        # The whole point of this block is that every window reaches its own
        # participant's model. If any did not, the arm is partly `ar16` again and
        # the artifact says so rather than averaging the two together silently.
        rows["subject_specific_ar_within"]["degraded"] = (
            f"{100 * frac_fallback:.1f}% of scored windows were served by the "
            "pooled fallback, so this arm is not purely subject-specific. "
            "ISSUE-013 is only discharged at 0%."
        )

    scw = _scwbd_scores(
        trainer, score_loader, n_theta_samples=0, n_mean_samples=n_mean_samples
    )
    scw_pw = np.asarray(scw["nll_per_window"], dtype=float)
    arm = _designation(trainer.cfg)
    _pt, lo, hi = bootstrap_ci(scw_pw, np.asarray(scw["subjects"]), n_boot=n_boot, seed=seed)
    rows[arm] = {
        "nll_per_sample": float(np.mean(scw_pw)),
        "nll_ci95": [float(lo), float(hi)],
        "mse": float(np.mean(scw["mse_per_window"])),
        "n_parameters": int(sum(p.numel() for p in trainer.model.parameters())),
    }

    paired: dict[str, Any] = {}
    for name, per in per_window.items():
        if per.size != scw_pw.size:
            paired[name] = {"error": f"length mismatch {per.size} vs {scw_pw.size}"}
            continue
        paired[name] = _paired_ci(scw_pw - per, groups, n_boot=n_boot, alpha=0.05, seed=seed)

    ss_per = per_window["subject_specific_ar_within"]
    pooled_per = per_window["ar16_pooled"]
    own_vs_pooled = (
        _paired_ci(ss_per - pooled_per, groups, n_boot=n_boot, alpha=0.05, seed=seed)
        if ss_per.size == pooled_per.size
        else {"error": "length mismatch"}
    )
    identical_to_pooled = bool(
        ss_per.size == pooled_per.size and np.array_equal(ss_per, pooled_per)
    )

    d = paired.get("subject_specific_ar_within", {})
    if "delta" not in d:
        verdict = f"not decidable: {d.get('error', 'no paired interval')}"
    elif not d.get("excludes_zero"):
        verdict = (
            f"{arm} and a per-participant AR(16) fitted on that participant's own "
            "earlier windows are indistinguishable on this split "
            f"({d['delta']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]). "
            "Cross-participant pretraining is not shown to transfer anything a "
            "person's own past does not already give."
        )
    elif d["delta"] > 0:
        verdict = (
            f"{arm} is BEATEN by a per-participant AR(16) fitted on that "
            f"participant's own earlier windows ({d['delta']:+.4f} "
            f"[{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]). On this corpus, "
            "cross-participant pretraining bought less than the person's own past."
        )
    else:
        verdict = (
            f"{arm} beats a per-participant AR(16) fitted on that participant's "
            f"own earlier windows ({d['delta']:+.4f} [{d['ci_lo']:+.4f}, "
            f"{d['ci_hi']:+.4f}])."
        )

    return {
        "available": True,
        "protocol": "within_participant_temporal_v1",
        "split": (
            "OUTER: the participant-disjoint holdout, unchanged -- the model "
            "trained on none of these people. INNER: within each held-out "
            f"participant, the earliest {fit_fraction:.0%} of their windows fit "
            f"their own model, {gap_windows} windows are dropped as a gap, and "
            "their later windows are scored."
        ),
        "fit_fraction": fit_fraction,
        "gap_windows": gap_windows,
        "n_participants_scored": len(score_by),
        "n_participants_skipped": len(skipped),
        "skipped_participants": skipped,
        "n_fit_windows": int(fit_x.shape[0]),
        "n_score_windows": int(sc_x.shape[0]),
        "score_time_routing": routing,
        "fraction_via_pooled_fallback": frac_fallback,
        "identical_to_pooled_ar16": identical_to_pooled,
        "results": rows,
        "paired_vs_scwbd": paired,
        "subject_specific_minus_pooled": own_vs_pooled,
        "verdict": verdict,
        "not_comparable_with": (
            "`real_eeg_holdout.results`. Every arm there is fitted on OTHER "
            "people and has never seen the scored participant; the "
            "subject-specific arm here has seen that participant's earlier "
            "windows. The two numbers answer different questions and a table "
            "containing both rows would invite exactly the comparison that is "
            "invalid."
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
def _ablation_measured_loader(trainer, *, per_test_participant: int = 20):
    """The held-out measured windows each ablation arm is scored on, or None.

    Same dataset, same split and the same participant-stratified budget as
    `real_eeg_holdout`, so an arm's measured score sits on the same footing as
    the headline number. Fewer windows per participant (20 rather than 40),
    because this runs once per arm rather than once per run and the arms are
    compared against each other rather than against a baseline set.

    The split fingerprint is CHECKED, not assumed: scoring an arm on
    participants the checkpoint trained on would make every delta here a
    memorisation delta.
    """
    ds = getattr(trainer, "real_dataset", None)
    split = getattr(trainer, "real_split", None)
    if ds is None or not split or not split.get("test"):
        return None

    recorded = getattr(trainer, "_recorded_split_fingerprint", None)
    if recorded is not None:
        fp = split_fingerprint(ds, split)
        if recorded.get("sha256") != fp["sha256"]:
            raise RuntimeError(
                "real-EEG split does not match the checkpoint's, so the ablation "
                "arms would be scored on participants the model may have trained "
                f"on: recorded {recorded.get('sha256')}, recomputed {fp['sha256']}"
            )

    idx = _participant_stratified(ds, split["test"], per_test_participant, fold="test")
    if not idx:
        return None
    bs = max(8, trainer.cfg.data.batch // 4)
    return torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, idx), batch_size=bs, shuffle=False, num_workers=2
    )


def source_ablation(trainer, *, steps: int = 120, seed: int = 0) -> dict[str, Any]:
    """Leave-one-source-family-out: does each family earn its place?

    Appendix D, "Dataset-family breadth": "A longer source list is useful only
    when each family improves a specified port or exposes a failure."  Negative
    transfer -- a family whose removal *improves* the metric -- is reported with
    the same prominence as a gain.
    """
    import copy

    # The ablation RETRAINS, and the trainer's logger is keyed by
    # `cfg.train.run_name` -- so every arm appends to the production run's
    # training log. It did: `make release-003-ablate` put a `global_step=1`
    # row on the end of run 3's completed 13,400-step record, and
    # `make health-run3` then reported "log ends at global_step=1".
    #
    # `short_train` already sets `log_every = 10**9`, which is why this leaks
    # one row per arm rather than hundreds -- step 1 logs regardless. A bound
    # on the damage is not the same as not doing it: the log is the run's
    # transcript, and CLAUDE.md carries this exact trap ("`--out` moves
    # checkpoints, not logs") because it has cost real data here before.
    #
    # Redirected rather than silenced, so the arms remain inspectable.
    from .util import JsonlLogger

    # And the CHECKPOINTS, which is the half that destroys rather than pollutes.
    # `run_stage` writes `stage_<name>.pt` and `last.pt` into `trainer.out_dir`
    # when a stage ENDS, so `ckpt_every = 10**9` does not prevent it. Each arm
    # therefore overwrote the artifact being evaluated: run 3's completed
    # 13,400-step `last.pt` was replaced by a 200-step ablation arm, and
    # `stage_T4_simulator.pt` with it. The final weights survived only because
    # `stage_T5_measured_return.pt` had not yet been reached by the arm loop.
    #
    # An evaluation must not be able to modify its own subject. Redirected to a
    # scratch directory for the duration.
    # And `report_dir`, which is the third output this had to be told about
    # one at a time. `run_stage` also writes `mixture_<stage>.json` there, both
    # flat and run-scoped, so an arm overwrote run 3's published T4 mixture with
    # its own: ten sources became nine, `sleepedf_real` missing, because that
    # was the family the arm had dropped.
    #
    # Enumerating outputs to suppress is how this defect kept coming back --
    # `ckpt_every`, then `log_every`, then a logger redirect, then out_dir, and
    # the mixture report was still writing to the production path. Redirect the
    # DIRECTORIES; anything the trainer writes then follows automatically,
    # including artifacts added later that nobody thinks to check here.
    _saved_logger = trainer.logger
    _saved_out = trainer.out_dir
    _saved_reports = trainer.report_dir
    scratch = _saved_out.parent / f"{_saved_out.name}-ablation-scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    report_scratch = _saved_reports / f"{trainer.cfg.train.run_name}-ablation"
    report_scratch.mkdir(parents=True, exist_ok=True)
    trainer.out_dir = scratch
    trainer.report_dir = report_scratch
    trainer.logger = JsonlLogger(
        report_scratch / f"{trainer.cfg.train.run_name}_ablation_train.jsonl",
        echo=True,
        echo_every=1,
    )
    try:
        return _source_ablation_inner(trainer, steps=steps, seed=seed)
    finally:
        trainer.logger = _saved_logger
        trainer.out_dir = _saved_out
        trainer.report_dir = _saved_reports


def _source_ablation_inner(trainer, *, steps: int, seed: int) -> dict[str, Any]:
    import copy

    base_state = copy.deepcopy(trainer.model.state_dict())
    base_post = copy.deepcopy(trainer.posterior.state_dict())
    stage = trainer.cfg.train.stages[-2] if len(trainer.cfg.train.stages) > 1 else trainer.cfg.train.stages[0]
    families = [k for k, v in trainer.sources.items() if v.enabled and v.role not in ("negative_control", "evaluation_only")]
    out: dict[str, Any] = {"families": families, "steps_per_arm": steps}

    # THE MEASURED HOLDOUT, built ONCE and reused by every arm.
    #
    # `_sim_val_nll` alone cannot answer the question this ablation exists for.
    # It scores an arm on the simulator, and every measured gradient pulls the
    # model away from the simulator, so "dropping a measured source improves the
    # score" is close to a tautology -- run 3 returned nine negative deltas of
    # nine and the direction was predictable before it ran. Meanwhile that run
    # BEAT ITS BASELINES on measured EEG and nothing attributed the win.
    #
    # Same eleven arms, scored additionally on the same 27 held-out participants
    # the headline result uses. Both are reported; neither replaces the other,
    # and each says which it is.
    measured_loader = _ablation_measured_loader(trainer)

    def _measured_nll() -> float | None:
        if measured_loader is None:
            return None
        # Same discarded-marginal waste as above: only `nll_per_window` is read.
        scores = _scwbd_scores(
            trainer, measured_loader, n_theta_samples=0, n_mean_samples=64
        )
        arr = scores["nll_per_window"]
        return float(arr.mean()) if len(arr) else None

    def short_train(drop: str | None) -> tuple[float, float | None]:
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
        return _sim_val_nll(trainer), _measured_nll()

    sim_base, meas_base = short_train(None)
    out["with_all_sources"] = sim_base
    measured: dict[str, Any] = {
        "available": meas_base is not None,
        "metric": (
            "mean per-window gaussian NLL, raw data units, sensor space, on the "
            "same participant-disjoint holdout as real_eeg_holdout"
        ),
        "with_all_sources": meas_base,
    }
    for f in families:
        v, mv = short_train(f)
        out[f"without_{f}"] = v
        out[f"delta_{f}"] = v - sim_base
        if meas_base is not None and mv is not None:
            measured[f"without_{f}"] = mv
            measured[f"delta_{f}"] = mv - meas_base
    out["negative_transfer"] = [f for f in families if out[f"delta_{f}"] < 0]

    if meas_base is not None:
        deltas = {f: measured[f"delta_{f}"] for f in families if f"delta_{f}" in measured}
        measured["negative_transfer"] = [f for f, d in deltas.items() if d < 0]
        measured["contributed"] = [f for f, d in deltas.items() if d > 0]
        measured["interpretation"] = (
            "delta > 0 means removing the family made MEASURED prediction worse, i.e. "
            "the family carried information the model used on real data. This is the "
            "attribution the simulated metric cannot provide. Same caveats on "
            "magnitude as the simulated arm: one arm per family, no seed replication, "
            "no error bar -- the sign pattern is the result, the deltas are not "
            "effect sizes."
        )
    else:
        measured["reason"] = (
            "no real-EEG dataset or split on the trainer, so the arms could not be "
            "scored on measured data. Reported rather than omitted: without this the "
            "ablation answers only the simulated question."
        )
    out["measured"] = measured
    out["interpretation"] = (
        "delta > 0 means removing the family HURT (the family contributed); delta < 0 is "
        "NEGATIVE TRANSFER: the family made the model worse and is reported as such."
    )
    out["metric"] = "sim_val_nll"
    out["metric_warning"] = (
        "Scored on the SIMULATED validation set. Every measured gradient pulls the "
        "model away from the thing being scored, so a measured source showing "
        "negative transfer here is close to tautological. Run 3 returned nine "
        "negative deltas of nine and the direction was predictable before it ran. "
        "Read `measured` below instead for the question worth asking."
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
        _refusal = {
            "available": False,
            "reason": (
                "--quick refuses the real-EEG holdout. A reduced-cost variant would "
                "silently change the participant set the claim rests on."
            ),
        }
        rep["real_eeg_holdout"] = _refusal
        rep["within_participant_holdout"] = dict(_refusal)
        rep["session_individualisation"] = dict(_refusal)
    else:
        rep["real_eeg_holdout"] = real_eeg_holdout(trainer, seed=seed)
        # ISSUE-013's replacement for the dropped `subject_specific_ar` row. A
        # SEPARATE top-level key, not a row in the table above, because it is a
        # different quantity: the baseline has seen the scored participant and
        # every arm in that table has not.
        rep["within_participant_holdout"] = within_participant_holdout(trainer, seed=seed)
        # The ONLY block that measures the individualiser. Neither holdout above
        # does: `real_eeg_holdout` is participant-disjoint, so every scored
        # person's `z_person` row is exactly zero (its `individualization`
        # sub-block reports `n_at_initialisation` == every participant), and
        # `within_participant_holdout` scores the SC-WBD arm through
        # `_scwbd_scores` with no person effect fitted, so its subject-specific
        # AR row is the only arm there that has seen the participant.
        #
        # T6_individual trains a person effect and nothing in the report read it
        # until this call was added. The function has existed and been tested
        # since the split landed; it was never wired into the report, which is
        # the whole of HANDOFF-004 step (c)'s "what remains".
        rep["session_individualisation"] = session_individualisation(trainer, seed=seed)
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

    # Build the model in the state layout the CHECKPOINT was trained in, not the
    # one this tree currently defaults to. O-5b widened the shared interface from
    # D=59 to D=62, which is exactly the incompatibility ARCHITECTURE.md said it
    # would cause -- and a model constructed before that is consulted cannot load
    # run 2's weights at all.
    #
    # The trainer builds its model in __init__, so the layout has to be selected
    # BEFORE construction; wrapping the load site instead would be too late. That
    # is the whole reason this is a context manager around the constructor rather
    # than an argument to the loader.
    ckpt = a.checkpoint or str(Path(cfg.train.out_dir) / "last.pt")
    with layout_of_checkpoint(ckpt) if Path(ckpt).exists() else nullcontext():
        tr = FoundationTrainer(cfg, resume=False, quick=a.quick)
        # Data first: the individualizer's row count comes from the participant list.
        tr.build_data()
    if Path(ckpt).exists():
        from .checkpoint import CheckpointError, load_checkpoint

        # B5a: an individualised checkpoint must be loaded WITH its individualizer,
        # or the evaluation silently scores the population model.
        import torch as _torch

        _peek = _torch.load(ckpt, map_location="cpu", weights_only=False)
        if _peek.get("individualizer") is not None and tr.individualizer is None:
            # ONE constructor, the trainer's. This site used to size the module
            # itself -- `n_sessions = n_participants * 4` -- so a checkpoint
            # written with one session row per recorded session would not load
            # into it, and `strict=False` turned the shape mismatch into a
            # silently un-restored person effect.
            tr.ensure_individualizer()
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


# ======================================================================
# individualisation: the claim the participant-disjoint split cannot measure
# ======================================================================
@torch.no_grad()
def _theta_shift_spread(trainer, participants: Sequence[int]) -> dict[str, Any]:
    """Between-participant spread of the applied theta shift.

    On a participant-disjoint holdout this is **exactly** ``0.000e+00``, because
    no held-out person has a fitted person effect -- their ``z_person`` row is
    still zero. That is not a small effect; it is the split, and it is why
    ``subject_specific_ar`` comes out bit-identical to ``ar16``.

    Reported per theta dimension as well as pooled, because a shift that moves
    one parameter and not the others is a different finding from one that moves
    nothing.
    """
    ind = getattr(trainer, "individualizer", None)
    if ind is None:
        return {"available": False, "reason": "no individualizer on the trainer"}
    delta = ind.delta.detach()
    # DEDUPLICATED. `participants` arrives one entry per scored WINDOW, so the
    # raw list repeats each person once per window and `len(rows)` reported 1500
    # for 75 people under a field named `n_participants`. The std is unchanged
    # when every person contributes equally many windows and is silently
    # window-weighted when they do not, which is the case this guards.
    rows = sorted({int(p) for p in participants if 0 <= int(p) < delta.shape[0]})
    if not rows:
        return {"available": False, "reason": "no scored participant has a person-effect row"}
    d = delta[rows].float()
    per_dim = d.std(dim=0, unbiased=False)
    return {
        "available": True,
        "n_participants": len(rows),
        "spread_pooled": float(d.std(unbiased=False)),
        # The model's OWN prior scale for a person effect. Without it,
        # `spread_pooled` is a bare number and "near zero" is an eyeball
        # judgement; with it the ratio is checkable off the artifact.
        "prior_sd_person": [float(v) for v in ind.log_sd_person.detach().float().exp().flatten()],
        "spread_over_prior_sd": float(
            d.std(unbiased=False)
            / ind.log_sd_person.detach().float().exp().mean().clamp_min(1e-12)
        ),
        "spread_per_theta": [float(v) for v in per_dim],
        "n_rows_exactly_zero": int((d.abs().sum(dim=1) == 0).sum()),
        "note": (
            "spread_pooled == 0.0 exactly means the individualizer applied nothing "
            "to these participants. On a participant-disjoint split that is the "
            "expected value and the reason individualisation cannot be measured "
            "there at all."
        ),
    }


def session_individualisation(
    trainer,
    *,
    source_id: str = "sleepedf_real",
    seed: int = 0,
    per_participant: int = 20,
    n_boot: int = 2000,
) -> dict[str, Any]:
    """Does a fitted person effect predict that person's HELD-OUT session?

    The third capability on the landing page -- "fine-tuneable for personalized
    neurotechnology" -- has never been measured, because every run so far used a
    participant-disjoint split on which it is unmeasurable by construction.

    Sleep-EDFx makes it measurable: 75 of its 78 participants were recorded on
    two consecutive nights. `session_split` holds the participants fixed and
    splits by session, so the same person is on both sides and the held-out
    object is a NIGHT rather than a person.

    This is the arrangement refusal **R10** forbids for a generalisation claim,
    which is why it has its own splitter and its own audit. `leakage_check` is
    expected to fail on this split; `session_leakage_check` is the one that
    applies, and its failure modes are the opposite ones -- a night on both
    sides, or a test participant absent from train.

    Returns a report; ``ok`` is false and ``reason`` is set rather than raising,
    so a run without the corpus still produces an artifact that says why.
    """
    from .realdata import session_leakage_check, session_split

    ds = (getattr(trainer, "eeg_datasets", {}) or {}).get(source_id)
    if ds is None:
        return {
            "ok": False,
            "claim": "individualisation",
            "reason": f"{source_id} is not loaded on this trainer, so the "
            "individualisation claim cannot be measured. It is NOT thereby "
            "supported.",
        }

    try:
        split = session_split(ds, seed=seed)
    except RuntimeError as exc:
        return {"ok": False, "claim": "individualisation", "reason": str(exc)}

    audit = session_leakage_check(split, ds)
    if not audit["ok"]:
        raise RuntimeError(
            f"the session split does not hold: {audit['violations']}. Scoring it "
            "would report a memorisation result as individualisation."
        )

    subjects = list(getattr(ds, "window_subjects", []))
    test_idx = list(split["test"])
    if per_participant > 0 and test_idx:
        by_subj: dict[str, list[int]] = {}
        for i in test_idx:
            by_subj.setdefault(subjects[i], []).append(i)
        rng = np.random.default_rng(seed)
        test_idx = sorted(
            j
            for v in by_subj.values()
            for j in (v if len(v) <= per_participant
                      else list(rng.choice(v, per_participant, replace=False)))
        )
    if not test_idx:
        return {"ok": False, "claim": "individualisation", "reason": "empty held-out session fold"}

    bs = max(8, trainer.cfg.data.batch // 4)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, test_idx), batch_size=bs, shuffle=False, num_workers=2
    )
    # n_theta_samples=0: this block reads `nll_per_window` and `subjects` and
    # nothing else. The default of 32 computes the SECONDARY marginal -- 32 extra
    # full rollouts per batch -- and every one of them is discarded here. On the
    # first run that reached this code it cost ~33x the rollouts the returned
    # numbers needed. `within_participant_holdout` already passes 0 for the same
    # reason.
    scores = _scwbd_scores(
        trainer, loader, n_theta_samples=0, n_mean_samples=64, source_id=source_id
    )
    nll = np.asarray(scores["nll_per_window"], dtype=float)
    scored_subjects = list(scores["subjects"])

    pid = []
    if hasattr(trainer, "participant_index"):
        try:
            pid = [int(v) for v in trainer.participant_index(scored_subjects).tolist()]
        except Exception:  # noqa: BLE001 - a missing index is reported, not fatal
            pid = []

    # Per PARTICIPANT, not per window: 20 windows of one person are one
    # observation of that person, and averaging them as though they were 20
    # is the interval error `leakage_check`'s note already warns about.
    per_person: dict[str, list[float]] = {}
    for s, v in zip(scored_subjects, nll):
        per_person.setdefault(str(s), []).append(float(v))
    means = {k: float(np.mean(v)) for k, v in per_person.items()}

    boot = None
    if means and n_boot > 0:
        keys = sorted(means)
        vals = np.array([means[k] for k in keys])
        rng = np.random.default_rng(seed)
        draws = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
        boot = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]

    return {
        "ok": True,
        "claim": "individualisation",
        "refuses": "generalisation",
        "source": source_id,
        "split_audit": audit,
        "n_participants_individualisable": audit["n_participants_individualisable"],
        "n_test_windows": len(test_idx),
        "held_out_session_nll_per_participant": means,
        "held_out_session_nll": float(np.mean(list(means.values()))) if means else None,
        "held_out_session_nll_ci95": boot,
        "theta_shift": _theta_shift_spread(trainer, pid),
        "interval_note": (
            "The interval is a cluster bootstrap over PARTICIPANTS on the "
            "per-participant mean, not over windows. Windows within a night are "
            "correlated and a window-level interval would be far too narrow."
        ),
        "what_would_falsify": (
            "theta_shift.spread_pooled at or near zero means the individualizer "
            "applied nothing even on a split built to let it apply something, and "
            "the third capability is unsupported. Say so on the site in the terms "
            "runs 1 and 2 got."
        ),
    }


# The entry point is LAST on purpose. It used to sit above
# `session_individualisation`, which meant `python -m scwbd.foundation.evaluate`
# called `main()` before that `def` had executed -- so the function was
# unreachable from the CLI no matter what referenced it, and wiring it into
# `evaluate_model` produced a NameError rather than a number. Anything defined
# below this block is dead to the command line.
# Guarded by tests/foundation/test_individualisation_reaches_the_report.py.
if __name__ == "__main__":  # pragma: no cover
    main()
