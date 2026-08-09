"""E7/E8 -- split integrity at evaluation time, and the statistic the verdict uses.

E7. ``evaluate_model`` calls ``trainer.build_data()``, which *rebuilds* the
participant split from the dataset it happens to find on disk.
``_assign_groups`` shuffles the sorted participant list and slices by count, so
the assignment of every participant depends on the whole set: add or remove one
recording and folds move wholesale.  The failure is silent and reads *better*
when broken -- a test participant promoted into training makes the held-out
score improve.  **Live, as ISSUE-014**, and the two tests that measure it are
deliberately red.

``split_fingerprint`` and the mismatch check in ``real_eeg_holdout`` landed
since, and they convert the leak into a refusal for any checkpoint that recorded
a fingerprint.  The released one did not, so detection is real but does not
currently fire on the artifact that matters; the ordering of the remaining work
is in ISSUE-014.

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
    """The mechanism, as a pure-function test on ``_assign_groups``.

    **Live finding, recorded as ISSUE-014, and meant to be red.** Measured on the
    109-participant roster at the released seed: removing one participant moves
    17 of the remaining 108, four of them from ``train`` into ``test``.
    ``_hash_assign_groups`` in the same module moves 0, but adopting it changes
    the split the released checkpoint trained under (67/31/11 rather than
    71/27/11), so it is a next-run change. Discharged by the sequence in
    ``reports/known_issues.md`` ISSUE-014, not by relaxing this assertion.
    """
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
    """``--quick`` sets ``max_subjects=6``, which is a different split entirely.

    **Live finding, the second half of ISSUE-014, and meant to be red.** The
    six-subject re-split puts ``S001`` and ``S004`` in the held-out fold, and the
    released run trained on both. Discharged by making ``--quick`` subset the
    recorded test fold or refuse, not by relaxing this assertion.
    """
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


def test_scwbd_and_baselines_are_scored_on_the_same_windows_in_the_same_order(
    cfg, real_eeg, real_split
):
    """Pairing is only legitimate if the rows correspond; check that they do.

    ``real_eeg_holdout`` builds one ``test_loader`` over
    ``Subset(ds, _participant_stratified(...))`` with ``shuffle=False``. The
    baselines are scored on ``ctx, tgt`` collected from it and SC-WBD is scored
    by iterating the same object, so window ``i`` is the same window on both
    sides. That is what makes the missing paired interval (the sibling test) a
    defect rather than an impossibility.

    **This test used to grep for source literals that no longer exist.** It
    asserted ``"max_batches" in src`` and the exact string
    ``_scwbd_scores(trainer, test_loader, max_batches=max_batches)``. The
    evaluation stopped budgeting in batches -- ``_participant_stratified`` sets
    the budget in participants -- so both literals vanished and the test failed
    on the *fix*. The property it was written to check survived the change
    intact, so it is measured here directly instead of grepped.
    """
    import torch.utils.data

    from scwbd.foundation import evaluate
    from scwbd.foundation.evaluate import _participant_stratified, _window_subject

    src = inspect.getsource(evaluate.real_eeg_holdout)
    # BOTH loaders, not "at least one": `"shuffle=False" in src` was satisfied by
    # the train loader while the test loader shuffled, so the mutation that
    # breaks the pairing left this test green. Caught by mutating one of the two.
    assert "shuffle=True" not in src, (
        "a loader in real_eeg_holdout shuffles. The baselines and SC-WBD iterate "
        "the test loader separately, so a shuffled order means row i of one is a "
        "different window from row i of the other and no paired statistic over "
        "them is valid."
    )
    assert src.count("shuffle=False") >= 2, (
        f"real_eeg_holdout pins the order of only {src.count('shuffle=False')} of "
        f"its two loaders"
    )
    assert "_scwbd_scores(" in src and "test_loader" in src, (
        "real_eeg_holdout no longer scores SC-WBD on test_loader; the baselines "
        "are collected from that loader, so a different source for SC-WBD breaks "
        "the row correspondence a paired interval needs"
    )

    # The property itself, on the corpus the evaluation reads.
    te_idx = _participant_stratified(real_eeg, real_split["test"], 40, fold="test")
    expected = [_window_subject(real_eeg, i) for i in te_idx]
    bs = max(8, cfg.data.batch // 4)

    def collect(loader):
        out: list[str] = []
        for b in loader:
            out.extend(list(b["subject"]))
        return out

    def build():
        return torch.utils.data.DataLoader(
            torch.utils.data.Subset(real_eeg, te_idx),
            batch_size=bs,
            shuffle=False,
            num_workers=0,
        )

    first, second = collect(build()), collect(build())
    assert first == expected, (
        f"iterating the holdout's test_loader yields a different window order "
        f"than its own index list: {sum(a != b for a, b in zip(first, expected))} "
        f"of {len(expected)} positions differ. Row i of the baselines' score and "
        f"row i of SC-WBD's are then different windows, and any paired statistic "
        f"over them is meaningless."
    )
    assert first == second, (
        "two iterations of the same shuffle=False loader disagree, so the "
        "baselines and SC-WBD -- which iterate it separately -- are not scored "
        "on the same windows in the same order"
    )
