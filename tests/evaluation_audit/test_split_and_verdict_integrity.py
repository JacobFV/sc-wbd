"""E7/E8 -- split integrity at evaluation time, and the statistic the verdict uses.

E7. ``evaluate_model`` calls ``trainer.build_data()``, which *rebuilds* the
participant split from the dataset it happens to find on disk, so the splitter
has to give the same answer every time it is asked.  ``shuffle_slice_v1`` does
not: it shuffles the sorted participant list and slices by count, so the
assignment of every participant depends on the whole set and adding or removing
one recording moves folds wholesale.  The failure is silent and reads *better*
when broken -- a training participant promoted into the test fold makes the
held-out score improve.

**ISSUE-014, discharged for run 4.** The policy is now versioned
(``realdata.SPLIT_POLICIES``) and declared per run.  ``stable_hash_v2`` is the
default and what run 4 declares; runs 1-3 declare ``shuffle_slice_v1`` in their
own configs, because that is the split their released checkpoints trained under
and their recorded ``real_split`` fingerprints verify against it.

The two tests below **used to assert the defect and now assert the fix**: the
run-4 policy moves nobody, and ``--quick`` under an order-dependent policy is
refused rather than silently re-split.  The historical policy's instability is
measured here too, so the reason runs 1-3 are pinned to it stays on the record
rather than in a commit message.

E8. ``real_eeg_holdout`` decides ``scwbd_beaten_by`` on point estimates while
``baselines._paired_ci`` already computes the participant-clustered interval on
the per-window *difference*, which is both the correct statistic and much
tighter than comparing two marginal intervals.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest


ROSTER = [f"S{i:03d}" for i in range(1, 110)]


def _moves(policy: str, seed: int = 20260805):
    """(reassigned, train->test) when one of 109 participants disappears."""
    from scwbd.foundation.realdata import assign_groups

    kw = {"test_fraction": 0.25, "val_fraction": 0.1, "seed": seed, "policy": policy}
    a = assign_groups(ROSTER, **kw)
    b = assign_groups(ROSTER[:-1], **kw)
    moved = [s for s in ROSTER[:-1] if a[s] != b[s]]
    leaked = [s for s in ROSTER[:-1] if a[s] == "train" and b[s] == "test"]
    return moved, leaked


# ---------------------------------------------------------------- E7
def test_participant_assignment_is_stable_under_a_changed_participant_set():
    """The run-4 policy must reassign nobody when the roster changes.

    **This test used to assert the defect (ISSUE-014).** It measured
    ``_assign_groups`` -- now the ``shuffle_slice_v1`` policy -- and recorded
    that removing one of 109 participants reassigns 17 of the remaining 108 at
    the default seed, four of them from ``train`` into ``test``. That is the
    direction that scores a model on people it memorised and reports the result
    as generalisation.

    The policy is now versioned and declared per run, the default is the
    order-independent one, and this asserts the property directly.
    """
    from scwbd.foundation.realdata import DEFAULT_SPLIT_POLICY

    moved, leaked = _moves(DEFAULT_SPLIT_POLICY)
    assert not moved, (
        f"the default split policy {DEFAULT_SPLIT_POLICY!r} reassigns "
        f"{len(moved)} of {len(ROSTER)-1} participants when one is removed; "
        f"{len(leaked)} of them move from train into test, so the evaluation "
        f"would score the model on {len(leaked)} people it was trained on and "
        f"report the result as held-out generalisation."
    )


def test_the_default_split_policy_is_the_order_independent_one():
    """A new run must not have to know about ISSUE-014 to avoid it."""
    from scwbd.foundation.realdata import (
        DEFAULT_SPLIT_POLICY,
        ORDER_INDEPENDENT_POLICIES,
    )
    from scwbd.foundation.config import DataConfig

    assert DEFAULT_SPLIT_POLICY in ORDER_INDEPENDENT_POLICIES
    assert DataConfig().split_policy == DEFAULT_SPLIT_POLICY, (
        "the config default and the splitter default disagree, so what a run "
        "gets depends on which of the two it reaches first"
    )


def test_the_historical_policy_is_still_unstable_and_still_available():
    """Why runs 1-3 are pinned to it, kept as a measurement rather than a memory.

    ``shuffle_slice_v1`` is not repaired and must not be: three released
    checkpoints trained under it and their recorded ``real_split`` fingerprints
    verify against its folds and no others. What it must not do is become the
    default again, which the sibling test pins.
    """
    from scwbd.foundation.realdata import ORDER_INDEPENDENT_POLICIES, SPLIT_POLICIES

    assert "shuffle_slice_v1" in SPLIT_POLICIES
    assert "shuffle_slice_v1" not in ORDER_INDEPENDENT_POLICIES
    moved, leaked = _moves("shuffle_slice_v1")
    assert len(moved) == 17 and len(leaked) == 4, (
        f"shuffle_slice_v1's measured instability changed: {len(moved)} moved / "
        f"{len(leaked)} leaked against the recorded 17 / 4 at seed 20260805. The "
        f"policy runs 1-3 trained under has been edited, and their recorded "
        f"split fingerprints will no longer verify."
    )


def test_runs_one_to_three_declare_the_policy_they_trained_under():
    """A released holdout is part of the artifact, so its config states it.

    Without this, changing the default silently re-splits three published
    checkpoints and ``real_eeg_holdout`` starts refusing to evaluate them -- or
    worse, evaluates them on a different set of people.
    """
    import pathlib

    from scwbd.foundation.config import load_config

    repo = pathlib.Path(__file__).resolve().parents[2]
    expected = {
        "configs/scwbd_001_beta.yaml": "shuffle_slice_v1",
        "configs/run2/pilot-families.yaml": "shuffle_slice_v1",
        "configs/run3/scwbd-003.yaml": "shuffle_slice_v1",
        "configs/run4/scwbd-004.yaml": "stable_hash_v2",
    }
    for rel, want in expected.items():
        got = load_config(str(repo / rel)).data.split_policy
        assert got == want, f"{rel} declares split_policy={got!r}, expected {want!r}"


class _FakeCorpus:
    """Enough of a dataset for ``_window_subject`` and ``split_fingerprint``."""

    def __init__(self, subjects, policy):
        self.recordings = [{"subject": s} for s in subjects]
        self.window_index = [(i, 0) for i in range(len(subjects))]
        self.participant_split_policy = policy

    def __len__(self):
        return len(self.window_index)


def test_the_fingerprint_records_which_policy_produced_it():
    """Otherwise a checkpoint states its folds and not how they were arrived at."""
    from scwbd.foundation.evaluate import split_fingerprint

    ds = _FakeCorpus(["A", "B", "C"], "stable_hash_v2")
    fp = split_fingerprint(ds, {"train": [0, 1], "val": [], "test": [2]})
    assert fp["policy"] == "stable_hash_v2"
    # ...and it is NOT in the sha256, or three released checkpoints stop
    # verifying against their own recorded hash.
    other = split_fingerprint(
        _FakeCorpus(["A", "B", "C"], "shuffle_slice_v1"),
        {"train": [0, 1], "val": [], "test": [2]},
    )
    assert other["sha256"] == fp["sha256"]


def test_a_policy_mismatch_is_refused_even_when_the_participant_ids_agree():
    """The strengthened half of the B4 guard.

    Two policies can deal the same folds on one roster. That agreement is a
    coincidence of that roster and the next participant to appear breaks it, so
    evaluating under a policy the checkpoint did not train with is refused rather
    than accepted on the strength of a matching hash.
    """
    import types

    import pytest as _pytest

    from scwbd.foundation import evaluate

    ds = _FakeCorpus(["A", "B", "C"], "stable_hash_v2")
    split = {"train": [0, 1], "val": [], "test": [2]}
    recorded = dict(evaluate.split_fingerprint(ds, split))
    recorded["policy"] = "shuffle_slice_v1"  # same ids, different splitter

    trainer = types.SimpleNamespace(
        real_test=[0],
        real_dataset=ds,
        real_split=split,
        cfg=types.SimpleNamespace(data=types.SimpleNamespace(context=8, batch=8)),
        _recorded_split_fingerprint=recorded,
    )
    with _pytest.raises(RuntimeError, match="policy"):
        evaluate.real_eeg_holdout(trainer)


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
    """``--quick`` sets ``max_subjects=6``; that must not redefine "held out".

    **This test used to assert the defect (ISSUE-014).** Under
    ``shuffle_slice_v1`` the six-subject re-split put ``S001`` and ``S004`` in
    the held-out fold although the released run trained on both -- a flag
    advertised as a cost saving changing the meaning of the headline metric.

    Two things discharge it and both are asserted here. Under the run-4 policy a
    participant's fold does not depend on the roster, so the reduced roster
    cannot promote a trained participant into the holdout. Under an
    order-dependent policy it still can, so ``build_data`` refuses outright
    rather than producing the smaller, wrong number.
    """
    from scwbd.foundation.realdata import (
        DEFAULT_SPLIT_POLICY,
        ORDER_INDEPENDENT_POLICIES,
        assign_groups,
    )

    kw = {"test_fraction": 0.25, "val_fraction": 0.1, "seed": 20260805}
    full = assign_groups(ROSTER, policy=DEFAULT_SPLIT_POLICY, **kw)
    quick = assign_groups(ROSTER[:6], policy=DEFAULT_SPLIT_POLICY, **kw)
    trained_on = [s for s in ROSTER[:6] if quick[s] == "test" and full[s] != "test"]
    assert not trained_on, (
        f"under {DEFAULT_SPLIT_POLICY!r} the six-participant --quick roster puts "
        f"{trained_on} in the held-out fold although the full-roster split trains "
        f"on them"
    )
    assert all(quick[s] == full[s] for s in ROSTER[:6]), (
        "the quick roster assigns at least one participant to a different fold "
        "from the full roster, so --quick is still a re-split"
    )

    # And the historical policy, which is still selectable, still leaks -- so the
    # refusal below is load-bearing rather than belt-and-braces.
    v1_full = assign_groups(ROSTER, policy="shuffle_slice_v1", **kw)
    v1_quick = assign_groups(ROSTER[:6], policy="shuffle_slice_v1", **kw)
    assert [s for s in ROSTER[:6] if v1_quick[s] == "test" and v1_full[s] != "test"] == [
        "S001",
        "S004",
    ], "shuffle_slice_v1's measured --quick leak changed; ISSUE-014's record is stale"
    assert "shuffle_slice_v1" not in ORDER_INDEPENDENT_POLICIES


def test_quick_refuses_an_order_dependent_split_policy():
    """The refusal itself, exercised rather than asserted from the source text.

    ``build_data`` wraps the measured-EEG block in a ``try/except`` that turns
    every exception into a ``[warn] real EEG unavailable`` and carries on, so
    this check has to fire BEFORE that block or it is not a refusal at all. The
    test therefore calls ``build_data`` and requires a raise -- a version that
    grepped the source would pass on a check placed three lines too low.
    """
    import types

    import pytest as _pytest

    from scwbd.foundation.train import FoundationTrainer

    trainer = object.__new__(FoundationTrainer)
    trainer._data_ready = False
    trainer.quick = True
    trainer.cfg = types.SimpleNamespace(
        data=types.SimpleNamespace(split_policy="shuffle_slice_v1")
    )
    with _pytest.raises(RuntimeError, match="split_policy"):
        FoundationTrainer.build_data(trainer)

    # ...and it is the POLICY that refuses, not `--quick` as such.
    trainer.cfg.data.split_policy = "stable_hash_v2"
    with _pytest.raises(AttributeError):
        # gets past the refusal and dies on the next thing this stub lacks
        FoundationTrainer.build_data(trainer)


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
