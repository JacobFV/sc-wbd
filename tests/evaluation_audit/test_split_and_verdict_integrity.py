"""E7/E8 -- split integrity at evaluation time, and the statistic the verdict uses.

E7. ``evaluate_model`` calls ``trainer.build_data()``, which *rebuilds* the
participant split from the dataset it happens to find on disk.  Nothing compares
that split to the one the checkpoint was trained under, and the checkpoint does
not record one.  ``_assign_groups`` shuffles the sorted participant list and
slices by count, so the assignment of every participant depends on the whole
set: add or remove one recording and folds move wholesale.  The failure is
silent and reads *better* when broken -- a test participant promoted into
training makes the held-out score improve.

E8. ``real_eeg_holdout`` decides ``scwbd_beaten_by`` on point estimates while
``baselines._paired_ci`` already computes the participant-clustered interval on
the per-window *difference*, which is both the correct statistic and much
tighter than comparing two marginal intervals.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest


# ---------------------------------------------------------------- E7
def test_participant_assignment_is_stable_under_a_changed_participant_set():
    """The mechanism, as a pure-function test on ``_assign_groups``."""
    from scwbd.foundation.realdata import _assign_groups

    full = [f"S{i:03d}" for i in range(1, 110)]
    # `a` is the split the run trained under; `b` is the split the evaluation
    # rebuilds after one recording fails to preprocess (or one more finishes
    # downloading).  The dangerous direction is train -> test: a participant the
    # model memorised is scored as held out, which *improves* the number.
    a = _assign_groups(full, test_fraction=0.25, val_fraction=0.1, seed=20260805)
    b = _assign_groups(full[:-1], test_fraction=0.25, val_fraction=0.1, seed=20260805)
    moved = [s for s in full[:-1] if a[s] != b[s]]
    leaked = [s for s in full[:-1] if a[s] == "train" and b[s] == "test"]
    assert not moved, (
        f"removing a single participant reassigns {len(moved)} of {len(full)-1} "
        f"others; {len(leaked)} of them move from train into test, so the "
        f"evaluation would score the model on {len(leaked)} people it was "
        f"trained on and report the result as held-out generalisation. Nothing "
        f"in the evaluation compares the rebuilt split to the trained one."
    )


def test_checkpoint_records_the_split_it_was_trained_under(compiled_checkpoint):
    """Without this the evaluation cannot check that it rebuilt the same split."""
    _, payload = compiled_checkpoint
    extra = payload.get("extra") or {}
    metrics = payload.get("metrics") or {}
    keys = set(extra) | set(metrics) | set(payload)
    has_split = any("split" in str(k).lower() for k in keys)
    assert has_split, (
        f"the checkpoint records anatomy, lead field, theta prior and parameter "
        f"counts but no participant split and no fingerprint of the measured "
        f"corpus (keys: extra={sorted(extra)}, metrics={sorted(metrics)}). "
        f"'the evaluation used the training split' is therefore not checkable "
        f"from the artifact."
    )


def test_evaluation_verifies_the_split_it_rebuilt(compiled_checkpoint):
    """Source-level: an unexercised verification path has no bug count."""
    from scwbd.foundation import evaluate

    src = inspect.getsource(evaluate)
    assert "real_split" in src or "split_hash" in src or "split_fingerprint" in src, (
        "scwbd.foundation.evaluate never references the realised split. It calls "
        "trainer.build_data(), takes whatever trainer.real_test happens to be, "
        "and reports it as 'a participant-level holdout'."
    )


def test_quick_mode_does_not_silently_change_the_holdout():
    """``--quick`` sets ``max_subjects=6``, which is a different split entirely."""
    from scwbd.foundation.realdata import _assign_groups

    full = [f"S{i:03d}" for i in range(1, 110)]
    a = _assign_groups(full, test_fraction=0.25, val_fraction=0.1, seed=20260805)
    q = _assign_groups(full[:6], test_fraction=0.25, val_fraction=0.1, seed=20260805)
    trained_on = [s for s in full[:6] if q[s] == "test" and a[s] != "test"]
    assert not trained_on, (
        f"`evaluate --quick` builds the dataset with max_subjects=6 and re-splits "
        f"it, putting {trained_on} in the held-out fold although the released run "
        f"trained on them. The flag reduces cost; it also silently changes what "
        f"'held out' means, and nothing in the output says so."
    )


# ---------------------------------------------------------------- E8
def test_verdict_rests_on_a_paired_interval_not_a_point_estimate():
    """E8 verdict test.

    ``scwbd_beaten_by = [k for k, v in ranking if v < ref]`` is a comparison of
    two point estimates.  The harness owns ``_paired_ci``, which resamples the
    per-window difference over participants using shared draws; that is the
    statistic a claim rests on, and it is strictly available here because both
    sides are scored on the same windows in the same order.
    """
    from scwbd.foundation import evaluate

    src = inspect.getsource(evaluate.real_eeg_holdout)
    assert "_paired_ci" in src or "paired" in src, (
        "real_eeg_holdout ranks models by point estimate and emits "
        "scwbd_beaten_by from it, while baselines._paired_ci -- the "
        "participant-clustered interval on the per-window difference -- is "
        "imported from the same module and never called. A point-estimate "
        "verdict is not admissible when the paired interval is available: "
        "it has no error bar at all, and the marginal intervals it is later "
        "compared against are the wrong (and much wider) contrast."
    )


def test_scwbd_and_baselines_are_scored_on_the_same_windows_in_the_same_order():
    """Pairing is only legitimate if the rows correspond; check the harness could.

    Both paths iterate ``test_loader`` with ``shuffle=False`` and the same
    ``max_batches``, so window ``i`` is the same window on both sides. That is
    what makes the missing paired interval a defect rather than an impossibility.
    """
    from scwbd.foundation import evaluate

    src = inspect.getsource(evaluate.real_eeg_holdout)
    assert "shuffle=False" in src and "max_batches" in src
    assert "_scwbd_scores(trainer, test_loader, max_batches=max_batches)" in src
