"""E2 -- what ``real_eeg_holdout``'s ``[:max_batches]`` slice actually samples.

``real_eeg_holdout`` builds both loaders with ``shuffle=False`` and then stops
after ``max_batches`` batches of ``max(8, batch // 4)`` windows.  The dataset is
laid out recording-by-recording, so a head slice of a participant-grouped fold
is a head slice of *one participant*.

Everything downstream inherits it: the baselines are fitted on that one training
participant, every model is scored on that one test participant, and the
"participant-clustered 95% CI" is computed from a single cluster.
"""

from __future__ import annotations

import numpy as np
import pytest


def _head_participants(cfg, dataset, indices, max_batches):
    """Participants covered by the first ``max_batches`` batches, shuffle=False."""
    bs = max(8, cfg.data.batch // 4)
    fold_subs = np.asarray(dataset.window_subjects)[np.asarray(indices)]
    head = fold_subs[: max_batches * bs]
    return sorted(set(head)), sorted(set(fold_subs)), bs


@pytest.mark.parametrize("fold", ["train", "test"])
@pytest.mark.parametrize("max_batches", [6, 40])  # quick, and the released default
def test_holdout_slice_covers_more_than_one_participant(
    cfg, real_eeg, real_split, fold, max_batches
):
    # `max_batches` is retained as the parametrisation only to keep the ids
    # stable; the evaluation no longer budgets in batches. See the sibling
    # test's docstring: this file asserted a head slice that was fixed.
    import numpy as np

    from scwbd.foundation.evaluate import _participant_stratified

    per = 40 if fold == "test" else 30
    idx = _participant_stratified(real_eeg, real_split[fold], per, fold=fold)
    subs = np.asarray(real_eeg.window_subjects)[np.asarray(idx)]
    seen = sorted(set(subs.tolist()))
    all_subs = sorted({real_eeg.window_subjects[i] for i in real_split[fold]})
    assert len(seen) > 1, (
        f"_participant_stratified drew {len(seen)} participant(s) from the {fold} "
        f"fold's {len(all_subs)}. On train that fits every baseline on one person; "
        "on test it scores every model on one person and hands bootstrap_ci a "
        "single cluster. The stratifier has regressed to a window budget."
    )


@pytest.mark.parametrize("fold", ["train", "test"])
def test_holdout_slice_is_a_representative_fraction_of_its_fold(
    cfg, real_eeg, real_split, fold
):
    """A summary of a fold must see a non-negligible share of it."""
    # Measured on the sampler the evaluation calls, not on a head slice. The
    # budget is PARTICIPANTS x windows-per-participant, which is the change that
    # fixed the one-person defect this file was written for.
    from scwbd.foundation.evaluate import _participant_stratified

    per = 40 if fold == "test" else 30
    n_fold = len(real_split[fold])
    n_used = len(_participant_stratified(real_eeg, real_split[fold], per, fold=fold))
    frac = n_used / n_fold
    assert frac >= 0.10, (
        f"real_eeg_holdout summarises the {fold} fold from {n_used} of {n_fold} "
        f"windows ({100 * frac:.1f}%), taken as a contiguous head slice rather "
        f"than a sample."
    )


def test_bootstrap_ci_is_given_more_than_one_cluster(cfg, real_eeg, real_split):
    """The interval the report calls 'participant-clustered' must have clusters.

    ``bootstrap_ci`` returns ``(point, nan, nan)`` when ``n_clusters < 2``. It is
    right to refuse, and the caller does not notice: ``nll_ci95`` is written as
    ``[NaN, NaN]`` beside prose about overlapping intervals.

    **This test used to assert the defect.** It took a contiguous head slice --
    ``window_subjects[test_idx][: 40 * bs]`` -- and folds are ordered by
    recording, so the slice landed inside one participant and the assertion
    failed by construction. That WAS the evaluation's behaviour, and it was
    fixed: `_participant_stratified` now sets the budget in participants rather
    than batches, and its docstring records why ("40 batches of 16 drew 640
    windows from participant-ordered folds of ~2,650, so every baseline was fit
    on one person"). The test was never updated, so it kept reporting a repaired
    defect as live.

    It now exercises the sampler the evaluation actually calls. A red here means
    the stratifier has regressed, which is worth knowing; the old red meant only
    that this file sliced differently from the code it audits.
    """
    import numpy as np

    from scwbd.foundation.baselines import bootstrap_ci
    from scwbd.foundation.evaluate import _participant_stratified

    idx = _participant_stratified(real_eeg, real_split["test"], 40, fold="test")
    subs = np.asarray(real_eeg.window_subjects)[np.asarray(idx)]
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
