"""E4 -- a baseline handicapped by construction invalidates the comparison too.

Two separate checks:

1. ``SubjectSpecificBaseline`` is fitted on the *train* participants and scored
   on the *test* participants.  Refusal R10 makes those sets disjoint by
   construction, so ``models_.get(subject, fallback_)`` misses for **every**
   scored window and the baseline degrades to its pooled fallback -- which, at
   the same order and the same seed, is bit-for-bit ``ar16``.  Nothing in
   ``describe()`` reports this: ``fallback_subjects`` records participants who
   were thin *at fit time*, not participants routed to the fallback *at score
   time*, so the field reads clean precisely when the routing is 100% fallback.

2. ``real_eeg_holdout`` fits every baseline on ``max_batches`` batches of the
   train fold while SC-WBD was trained on the whole of it.  That asymmetry is
   the mirror image of the units defect and equally invalidating.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


def _clustered_windows(n_subjects: int, n_per: int, t: int = 72, c: int = 8, seed: int = 0):
    """Windows with a per-participant AR character, so a subject-specific fit has
    something real to exploit and the test is not vacuous."""
    g = torch.Generator().manual_seed(seed)
    xs, subs = [], []
    for s in range(n_subjects):
        a = 0.55 + 0.35 * torch.rand(1, generator=g).item()  # participant-specific pole
        amp = 0.5 + torch.rand(1, generator=g).item()
        e = torch.randn(n_per, t, c, generator=g) * amp
        x = torch.zeros_like(e)
        for k in range(1, t):
            x[:, k] = a * x[:, k - 1] + e[:, k]
        xs.append(x)
        subs += [f"P{s:03d}"] * n_per
    return torch.cat(xs), np.asarray(subs)


def test_subject_specific_baseline_is_not_silently_its_pooled_fallback():
    """E4a verdict test.  Fails while the fit and score participant sets are disjoint."""
    from scwbd.foundation.baselines import ARBaseline, SubjectSpecificBaseline

    tr_x, tr_s = _clustered_windows(8, 40, seed=0)
    te_x, te_s = _clustered_windows(4, 40, seed=99)
    te_s = np.asarray([f"Q{s[1:]}" for s in te_s])  # disjoint ids, as R10 guarantees
    ctx, tgt = te_x[:, :24], te_x[:, 24:]

    ss = SubjectSpecificBaseline(base=ARBaseline, base_kwargs={"order": 16}).fit(tr_x, groups=tr_s)
    pooled = ARBaseline(order=16).fit(tr_x, groups=tr_s)
    a = np.asarray(ss.score(ctx, tgt, te_s)["per_window_nll"])
    b = np.asarray(pooled.score(ctx, tgt, te_s)["per_window_nll"])

    d = ss.describe()
    assert not np.array_equal(a, b), (
        f"subject_specific_ar is bit-for-bit identical to ar16 on every one of "
        f"{a.size} scored windows: none of the test participants has a fitted "
        f"model, so 100% of windows route to the pooled fallback. describe() "
        f"reports n_subject_models={d['n_subject_models']} and "
        f"fallback_subjects={len(d['fallback_subjects'])}, which reads healthy. "
        f"The thesis's hardest baseline is not being run; a duplicate of ar16 is."
    )


def test_subject_specific_baseline_reports_score_time_fallback_routing():
    """The absence variant: the null case must write something.

    ``fallback_subjects`` is only ever written at fit time.  There is no field
    anywhere that says "this fraction of scored windows used the fallback", so
    total degradation and healthy operation produce identical provenance.
    """
    from scwbd.foundation.baselines import ARBaseline, SubjectSpecificBaseline

    tr_x, tr_s = _clustered_windows(6, 30, seed=1)
    te_x, te_s = _clustered_windows(3, 20, seed=7)
    te_s = np.asarray([f"Q{s[1:]}" for s in te_s])
    ss = SubjectSpecificBaseline(base=ARBaseline, base_kwargs={"order": 16}).fit(tr_x, groups=tr_s)
    ss.score(te_x[:, :24], te_x[:, 24:], te_s)
    d = ss.describe()
    routing_keys = [k for k in d if "rout" in k or "score_time" in k or "unmatched" in k]
    assert routing_keys, (
        "SubjectSpecificBaseline.describe() has no field recording how many "
        "scored windows were routed to the pooled fallback. A field only ever "
        "written on success is not a record."
    )


def test_baselines_and_scwbd_are_fitted_on_comparable_training_data(cfg, real_eeg, real_split):
    """E4b verdict test: the training-set asymmetry, quantified."""
    bs = max(8, cfg.data.batch // 4)
    n_fold = len(real_split["train"])
    n_baseline = min(40 * bs, n_fold)
    subs = np.asarray(real_eeg.window_subjects)[np.asarray(real_split["train"])]
    n_subj_baseline = len(set(subs[:n_baseline]))
    n_subj_fold = len(set(subs))
    assert n_subj_baseline >= 0.5 * n_subj_fold, (
        f"every baseline in real_eeg_holdout is fitted on {n_baseline} windows "
        f"from {n_subj_baseline} participant(s), while SC-WBD was trained on the "
        f"full {n_fold} windows from {n_subj_fold} participants. The comparison "
        f"reads as 'structure beats AR' when what it measures is "
        f"'{n_subj_fold} participants of training data beats {n_subj_baseline}'."
    )
