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
    seen, all_subs, bs = _head_participants(cfg, real_eeg, real_split[fold], max_batches)
    assert len(seen) > 1, (
        f"real_eeg_holdout(max_batches={max_batches}) consumes the first "
        f"{max_batches * bs} windows of the {fold} fold, which are ALL from "
        f"participant {seen[0]!r}: {len(seen)} of {len(all_subs)} participants. "
        f"On the train fold this fits every baseline on one person; on the test "
        f"fold it scores every model on one person and hands bootstrap_ci a "
        f"single cluster."
    )


@pytest.mark.parametrize("fold", ["train", "test"])
def test_holdout_slice_is_a_representative_fraction_of_its_fold(
    cfg, real_eeg, real_split, fold
):
    """A summary of a fold must see a non-negligible share of it."""
    bs = max(8, cfg.data.batch // 4)
    n_fold = len(real_split[fold])
    n_used = min(40 * bs, n_fold)
    frac = n_used / n_fold
    assert frac >= 0.10, (
        f"real_eeg_holdout summarises the {fold} fold from {n_used} of {n_fold} "
        f"windows ({100 * frac:.1f}%), taken as a contiguous head slice rather "
        f"than a sample."
    )


def test_bootstrap_ci_is_given_more_than_one_cluster(cfg, real_eeg, real_split):
    """The interval the report calls 'participant-clustered' must have clusters.

    ``bootstrap_ci`` returns ``(point, nan, nan)`` when ``n_clusters < 2``.  It
    is right to refuse, but the caller does not notice: ``nll_ci95`` is written
    as ``[NaN, NaN]`` beside prose about "overlapping participant-clustered
    intervals".
    """
    from scwbd.foundation.baselines import bootstrap_ci

    bs = max(8, cfg.data.batch // 4)
    idx = np.asarray(real_split["test"])
    subs = np.asarray(real_eeg.window_subjects)[idx][: 40 * bs]
    point, lo, hi = bootstrap_ci(np.random.default_rng(0).normal(size=subs.size), subs, n_boot=200)
    assert np.isfinite(lo) and np.isfinite(hi), (
        f"the scored sample contains {len(set(subs))} participant(s), so every "
        f"'participant-clustered 95% CI' in the released report is [nan, nan] "
        f"while the surrounding text describes intervals overlapping."
    )
