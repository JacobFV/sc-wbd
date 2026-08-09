"""E2 -- what ``real_eeg_holdout`` samples from each fold.

``real_eeg_holdout`` used to build both loaders with ``shuffle=False`` and stop
after ``max_batches`` batches of ``max(8, batch // 4)`` windows.  The dataset is
laid out recording-by-recording, so a head slice of a participant-grouped fold
was a head slice of *one participant*, and everything downstream inherited it:
the baselines were fitted on that one training participant, every model was
scored on that one test participant, and the "participant-clustered 95% CI" was
computed from a single cluster.

``_participant_stratified`` replaced it and sets the budget in **participants**.
Its docstring records the measured reason ("40 batches of 16 drew 640 windows
from participant-ordered folds of ~2,650, so every baseline was fit on one
person").  The tests below exercise that sampler, so a red means it has
regressed to a window budget -- not that this file slices differently from the
code it audits, which is what the earlier version measured.

Use the ``window_subjects`` fixture, never ``real_eeg.window_subjects``: the
property is uncached and rebuilds one entry per window in the corpus on every
access.  See the fixture's docstring.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.parametrize("fold", ["train", "test"])
@pytest.mark.parametrize("per_participant", [6, 40])  # a small budget, and the released default
def test_holdout_sample_covers_more_than_one_participant(
    real_eeg, real_split, window_subjects, fold, per_participant
):
    """One participant per side is the defect this sampler was written to remove."""
    from scwbd.foundation.evaluate import _participant_stratified

    idx = _participant_stratified(real_eeg, real_split[fold], per_participant, fold=fold)
    seen = sorted(set(window_subjects[np.asarray(idx)].tolist()))
    all_subs = sorted(set(window_subjects[np.asarray(real_split[fold])].tolist()))
    assert len(seen) == len(all_subs), (
        f"_participant_stratified drew {len(seen)} participant(s) from the {fold} "
        f"fold's {len(all_subs)} at a budget of {per_participant} windows each. On "
        f"train that fits every baseline on a subset of the people SC-WBD trained "
        f"on; on test it scores every model on a subset and hands bootstrap_ci "
        f"fewer clusters than the fold has. The stratifier has regressed to a "
        f"window budget."
    )


@pytest.mark.parametrize("fold", ["train", "test"])
def test_holdout_sample_spends_its_budget_evenly_across_each_recording(
    real_eeg, real_split, window_subjects, fold
):
    """The sample must be even within each participant, and spend its budget.

    **The threshold this test used to carry measured the wrong quantity.** It
    required the sample to be ``>= 10%`` of the fold's *windows*, which the
    sampler does not do and should not: measured, it takes 2,130 of 189,765
    train windows (1.12%) and 1,080 of 71,670 test windows (1.51%), exactly 30
    and 40 per participant with every participant present.

    That threshold came from the head-slice era, where "a small fraction of the
    fold" and "one participant" were the same failure, and it was never run
    against the sampler that replaced them -- the file was quadratic (see the
    ``window_subjects`` fixture) and did not finish. Window count is not what
    sets the precision of the reported interval: ``bootstrap_ci`` resamples
    **participants**, so the cluster count (71 and 27, all of them -- the sibling
    test) governs the width, and windows within a participant are correlated
    enough that a tenfold increase would buy little. What must hold instead is
    that the budget is actually spent, evenly, and reaches the end of every
    recording.
    """
    from scwbd.foundation.evaluate import _participant_stratified

    per = 40 if fold == "test" else 30
    idx = _participant_stratified(real_eeg, real_split[fold], per, fold=fold)
    got = window_subjects[np.asarray(idx)]
    counts = {s: int((got == s).sum()) for s in set(got.tolist())}
    assert min(counts.values()) == max(counts.values()) == per, (
        f"the {fold} sample is uneven: per-participant counts run "
        f"{min(counts.values())}..{max(counts.values())} against a budget of "
        f"{per}. An uneven sample re-weights the fold by participant, and the "
        f"cluster bootstrap will report the mixture it was handed as if it were "
        f"the fold's."
    )

    # Evenly spread, not front-loaded: `np.linspace` over each participant's
    # windows is what makes this a sample rather than a head slice, and a head
    # slice would sit entirely in the first part of each participant's recording.
    fold_idx = np.asarray(real_split[fold])
    subs = window_subjects[fold_idx]
    picked = set(int(i) for i in idx)
    rel = []
    for s in sorted(set(subs.tolist())):
        loc = fold_idx[subs == s]
        pos = np.flatnonzero(np.isin(loc, list(picked)))
        if pos.size > 1:
            rel.append(pos.max() / (loc.size - 1))
    assert rel and float(np.mean(rel)) > 0.9, (
        f"the last window selected per participant sits at {100 * float(np.mean(rel)):.1f}% "
        f"of that participant's fold on average. An even sample reaches the end of "
        f"each recording; a head slice does not. The stratifier is front-loading."
    )


def test_bootstrap_ci_is_given_more_than_one_cluster(real_eeg, real_split, window_subjects):
    """The interval the report calls 'participant-clustered' must have clusters.

    ``bootstrap_ci`` returns ``(point, nan, nan)`` when ``n_clusters < 2``. It is
    right to refuse, and the caller does not notice: ``nll_ci95`` is written as
    ``[NaN, NaN]`` beside prose about overlapping intervals.

    **This test used to assert the defect.** It took a contiguous head slice --
    ``window_subjects[test_idx][: 40 * bs]`` -- and folds are ordered by
    recording, so the slice landed inside one participant and the assertion
    failed by construction. That WAS the evaluation's behaviour, and it was
    fixed. The test was never updated, so it kept reporting a repaired defect as
    live.
    """
    from scwbd.foundation.baselines import bootstrap_ci
    from scwbd.foundation.evaluate import _participant_stratified

    idx = _participant_stratified(real_eeg, real_split["test"], 40, fold="test")
    subs = window_subjects[np.asarray(idx)]
    n = len(set(subs.tolist()))
    assert n > 1, (
        f"_participant_stratified returned {n} participant(s) from a test fold of "
        f"{len(real_split['test'])} windows, so every participant-clustered "
        "interval would be [nan, nan]. The stratifier has regressed to a window "
        "budget."
    )

    point, lo, hi = bootstrap_ci(
        np.random.default_rng(0).normal(size=subs.size), subs, n_boot=200
    )
    assert np.isfinite(lo) and np.isfinite(hi), (
        f"{n} participants reached bootstrap_ci and it still returned a "
        "non-finite interval"
    )


def test_participant_stratified_refuses_a_single_participant_fold(real_eeg, real_split):
    """The null case must refuse, not return a degenerate sample.

    A fold that resolves to one participant cannot support a
    participant-clustered interval, and returning its windows anyway is how the
    original defect stayed invisible: ``bootstrap_ci`` emitted ``[nan, nan]`` and
    the report printed it beside prose about overlapping intervals.
    """
    from scwbd.foundation.evaluate import _participant_stratified, _window_subject

    one = [i for i in real_split["test"] if _window_subject(real_eeg, i) ==
           _window_subject(real_eeg, real_split["test"][0])]
    with pytest.raises(ValueError, match="participant"):
        _participant_stratified(real_eeg, one, 40, fold="test")
