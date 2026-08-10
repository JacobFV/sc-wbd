"""E4 -- a baseline handicapped by construction invalidates the comparison too.

Two separate checks, and after triage they have opposite verdicts.

1. **Was live, now discharged (ISSUE-013).** ``SubjectSpecificBaseline`` was
   fitted on the *train* participants and scored on the *test* participants.
   Refusal R10 makes those sets disjoint by construction, so
   ``models_.get(subject, fallback_)`` missed for **every** scored window and
   the baseline degraded to its pooled fallback -- which, at the same order and
   the same seed, is bit-for-bit ``ar16``.  It is a design conflict between a
   subject-specific baseline and a participant-disjoint holdout, not a coding
   slip, so the fix is a protocol change: the row is gone from the pooled table
   and the quantity is measured on a within-participant temporal split instead.

2. **Fixed.** ``describe()`` used to report nothing about score-time routing, so
   total degradation and healthy operation produced identical provenance.
   ``score_time_routing`` now records it and names the degradation, which is
   what makes finding 1 visible in the artifact rather than inferable only from
   a bit-comparison against ``ar16``.

3. **Was stale.** ``real_eeg_holdout`` used to fit every baseline on
   ``max_batches`` batches of the train fold -- a contiguous head slice, which
   landed inside one participant -- while SC-WBD was trained on the whole of it.
   That was repaired by ``_participant_stratified``.
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
    """E4a verdict test.  The arm must differ from ``ar16``, or not be reported.

    **This test used to assert the defect (ISSUE-013).** ``real_eeg_holdout``
    fitted ``SubjectSpecificBaseline`` on the *train* participants and scored it
    on the *test* participants, which R10 makes disjoint, so 100% of scored
    windows routed to the pooled fallback and the row was bit-for-bit ``ar16``
    -- a duplicate carrying the name of the hardest baseline the thesis names.

    A subject-specific baseline and a participant-disjoint holdout are in direct
    conflict, so the discharge is a protocol change and both halves of it are
    asserted here:

    * the pooled table no longer contains the row, and records why it went;
    * the quantity it was meant to measure is measured on a within-participant
      temporal split, where the arm demonstrably is **not** its pooled fallback.
    """
    import inspect

    from scwbd.foundation import evaluate
    from scwbd.foundation.baselines import ARBaseline, SubjectSpecificBaseline

    # -- half 1: the duplicate row is gone from the pooled comparison ---------
    src = inspect.getsource(evaluate.real_eeg_holdout)
    assert '"subject_specific_ar": SubjectSpecificBaseline(' not in src, (
        "real_eeg_holdout still fits a SubjectSpecificBaseline on the train "
        "participants and scores it on the test participants. R10 makes those "
        "sets disjoint, so every scored window routes to the pooled fallback and "
        "the row is bit-for-bit ar16. ISSUE-013."
    )
    # The QUOTED key. `"dropped_baselines" in src` was satisfied by renaming the
    # key to `_dropped_baselines_removed`, because the old name is a substring of
    # the new one -- the mutation that removes the record left this green.
    assert '"dropped_baselines": {' in src, (
        "the row was removed without a record. A row that vanishes silently is "
        "indistinguishable from one that never existed; the artifact must say "
        "which arm went, why, and where the quantity is measured instead."
    )
    assert '"subject_specific_ar": (' in src and "within_participant_holdout" in src, (
        "the dropped-baselines record does not name the arm that went, or does "
        "not say where the quantity it measured is measured instead"
    )

    # -- half 2: on a within-participant split the arm actually runs ----------
    # Same participants on both sides -- what the temporal split gives it -- and
    # the fit windows are disjoint from the scored ones.
    x, s = _clustered_windows(4, 80, seed=0)
    early = np.concatenate([np.flatnonzero(s == u)[:40] for u in np.unique(s)])
    late = np.concatenate([np.flatnonzero(s == u)[40:] for u in np.unique(s)])
    fit_x, fit_s = x[early], s[early]
    sc_x, sc_s = x[late], s[late]
    ctx, tgt = sc_x[:, :24], sc_x[:, 24:]

    ss = SubjectSpecificBaseline(base=ARBaseline, base_kwargs={"order": 16}).fit(
        fit_x, groups=fit_s
    )
    pooled = ARBaseline(order=16).fit(fit_x, groups=fit_s)
    a = np.asarray(ss.score(ctx, tgt, sc_s)["per_window_nll"])
    b = np.asarray(pooled.score(ctx, tgt, sc_s)["per_window_nll"])

    r = ss.describe()["score_time_routing"]
    assert r["fraction_via_pooled_fallback"] == 0.0, (
        f"on a within-participant temporal split every scored participant has a "
        f"fitted model, yet {r['fraction_via_pooled_fallback']:.3f} of windows "
        f"routed to the pooled fallback ({r['subjects_without_a_fitted_model']})"
    )
    assert not np.array_equal(a, b), (
        f"subject_specific_ar is bit-for-bit identical to ar16 on every one of "
        f"{a.size} scored windows even with each participant's own model fitted. "
        f"The thesis's hardest baseline is still not being run."
    )


def test_the_within_participant_arm_is_not_pooled_with_the_disjoint_table():
    """The replacement arm is a different quantity and must be reported as one.

    ``within_participant_holdout``'s baseline has seen the scored participant
    and every arm in ``real_eeg_holdout.results`` has not. Putting them in one
    table invites exactly the comparison that is invalid, which is how the
    duplicate ``subject_specific_ar`` row survived three runs.
    """
    import inspect

    from scwbd.foundation import evaluate

    assert hasattr(evaluate, "within_participant_holdout")
    src = inspect.getsource(evaluate.within_participant_holdout)
    assert "not_comparable_with" in src, (
        "the within-participant block does not declare what it may not be "
        "compared against"
    )
    driver = inspect.getsource(evaluate.evaluate_model)
    # The CALL, not the key. `rep["within_participant_holdout"]` is also written
    # on the --quick branch, where it is a refusal object; a test that looked for
    # the key alone stayed green with the real call deleted. Caught by mutating
    # exactly that (`... = within_participant_holdout(trainer, seed=seed)` ->
    # `pass`), which this assertion now fails on.
    assert "within_participant_holdout(trainer" in driver, (
        "evaluate_model never calls within_participant_holdout, so the arm that "
        "replaces the dropped row is not in any artifact"
    )
    # It is a SEPARATE top-level key, not a row folded back into the table.
    assert 'rep["within_participant_holdout"]' in driver


def test_subject_specific_baseline_reports_score_time_fallback_routing():
    """The absence variant: the null case must write something.

    **This test used to assert the defect.** ``fallback_subjects`` was only ever
    written at fit time and there was no field anywhere saying "this fraction of
    scored windows used the fallback", so total degradation and healthy
    operation produced identical provenance. ``describe()`` now carries
    ``score_time_routing``.

    It now checks that the record is *true*, not merely present: on a disjoint
    fit/score participant set the routing must read 100% fallback and say so in
    words. A red means the record has gone back to reading clean while the arm
    is degraded -- which would hide ISSUE-013 rather than report it.
    """
    from scwbd.foundation.baselines import ARBaseline, SubjectSpecificBaseline

    tr_x, tr_s = _clustered_windows(6, 30, seed=1)
    te_x, te_s = _clustered_windows(3, 20, seed=7)
    te_s = np.asarray([f"Q{s[1:]}" for s in te_s])
    ss = SubjectSpecificBaseline(base=ARBaseline, base_kwargs={"order": 16}).fit(tr_x, groups=tr_s)

    before = ss.describe()["score_time_routing"]
    assert before.get("scored") is False, (
        "describe() reports a score-time routing record before anything was "
        "scored. A field that reads populated on an unscored model is not a record."
    )

    ss.score(te_x[:, :24], te_x[:, 24:], te_s)
    r = ss.describe()["score_time_routing"]
    assert r.get("scored") is True, (
        "describe() has no score-time routing record after score(). "
        "SubjectSpecificBaseline can degrade to its pooled fallback for 100% of "
        "windows; a field only ever written on success is not a record."
    )
    assert r["n_windows"] == te_x.shape[0]
    assert r["fraction_via_pooled_fallback"] == pytest.approx(1.0), (
        f"every test participant id is disjoint from the fitted set, so every "
        f"window must be recorded as routed to the pooled fallback; the record "
        f"says {r['fraction_via_pooled_fallback']:.3f} over {r['n_windows']} "
        f"windows with {r['n_windows_via_subject_model']} served by a subject "
        f"model that cannot exist."
    )
    assert "degraded" in r and "EVERY scored window" in r["degraded"], (
        f"the routing record is numerically right but says nothing: {r}. At 100% "
        f"fallback the arm IS its pooled base model, and the artifact must state "
        f"that rather than leave a reader to divide two integers."
    )


def test_baselines_and_scwbd_are_fitted_on_comparable_training_data(cfg, real_eeg, real_split):
    """E4b: the training-set asymmetry, quantified on the sampler in use.

    **This test used to assert the defect.** It measured a contiguous head slice
    -- ``window_subjects[train_idx][: 40 * bs]`` -- and the fold is ordered by
    recording, so the slice landed inside one participant and every baseline was
    fitted on one person while SC-WBD had all of them. That WAS the evaluation's
    behaviour and it was repaired: ``real_eeg_holdout`` builds its training
    windows with ``_participant_stratified(ds, split["train"],
    per_train_participant, fold="train")``, whose docstring records the measured
    reason ("40 batches of 16 drew 640 windows from participant-ordered folds of
    ~2,650, so every baseline was fit on one person").

    The residual asymmetry is a window budget -- 30 evenly spaced windows per
    participant against SC-WBD's whole fold -- and it is deliberate: the
    baselines are refitted on every evaluation. It is not the invalidating one,
    because it is the same for every participant rather than concentrated in one
    person. A red here means the budget has gone back to being set in batches.
    """
    from scwbd.foundation.evaluate import _participant_stratified, _window_subject

    per_train = 30  # real_eeg_holdout's default
    tr_idx = _participant_stratified(
        real_eeg, real_split["train"], per_train, fold="train"
    )
    subs_used = {_window_subject(real_eeg, i) for i in tr_idx}
    subs_fold = {_window_subject(real_eeg, i) for i in real_split["train"]}
    assert len(subs_used) == len(subs_fold), (
        f"every baseline in real_eeg_holdout is fitted on windows from "
        f"{len(subs_used)} participant(s), while SC-WBD was trained on the full "
        f"train fold's {len(subs_fold)}. The comparison then reads as 'structure "
        f"beats AR' when what it measures is '{len(subs_fold)} participants of "
        f"training data beats {len(subs_used)}'."
    )
    # ...and the per-participant budget must actually be spent, or the fit is
    # participant-complete and data-starved, which is the same defect rescaled.
    got = len(tr_idx) / max(len(subs_used), 1)
    assert got >= 0.9 * per_train, (
        f"the train sampler returns {got:.1f} windows per participant against a "
        f"budget of {per_train}; the baselines are being starved."
    )
