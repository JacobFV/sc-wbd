"""The individualisation split: sessions disjoint, participants **shared**.

`participant_split` and `session_split` are deliberately two functions rather
than one function with a flag, because they support claims that contradict each
other and the wrong choice is invisible in the returned dict -- both give back
`{"train": [...], "val": [...], "test": [...]}` of ints.

What each fold arrangement licenses:

* participant-disjoint -> "predicts an unseen person" (generalisation, R10)
* session-disjoint, participant-shared -> "predicts this person's unseen night"

Scoring the second and reporting the first is the failure this file guards, and
it cannot be caught downstream: the number looks the same either way.
"""

from __future__ import annotations

import pytest

from scwbd.foundation.realdata import (
    leakage_check,
    session_leakage_check,
    session_split,
)


class FakeDataset:
    """Minimal stand-in: the splitters only ever ask for the two window lists.

    Built from a `{subject: [session, ...]}` map with a fixed number of windows
    per session, so a test can state the corpus shape it needs in one line.
    """

    source = "fake"

    def __init__(self, sessions: dict[str, list[str]], per_session: int = 3) -> None:
        self.window_subjects: list[str] = []
        self.window_sessions: list[str] = []
        for subj in sorted(sessions):
            for sess in sessions[subj]:
                for _ in range(per_session):
                    self.window_subjects.append(subj)
                    self.window_sessions.append(f"{subj}/{sess}")

    def __len__(self) -> int:
        return len(self.window_subjects)


#: 75 of sleep-edfx's 78 sleep-cassette participants have two nights. The three
#: that do not are the reason single-session handling is specified rather than
#: assumed.
TWO_NIGHTS = {f"SC4{i:02d}": ["night1", "night2"] for i in range(10)}


def test_every_session_lands_in_exactly_one_fold() -> None:
    ds = FakeDataset(TWO_NIGHTS)
    split = session_split(ds, seed=0)
    rep = session_leakage_check(split, ds)
    assert rep["ok"], rep["violations"]
    assert not any(v["kind"] == "session_across_folds" for v in rep["violations"])


def test_the_test_participants_are_all_present_in_train() -> None:
    """The half that makes it individualisation rather than generalisation."""
    ds = FakeDataset(TWO_NIGHTS)
    split = session_split(ds, seed=0)
    train_subj = {ds.window_subjects[i] for i in split["train"]}
    test_subj = {ds.window_subjects[i] for i in split["test"]}
    assert test_subj, "empty test fold measures nothing"
    assert test_subj <= train_subj, sorted(test_subj - train_subj)


def test_a_test_participant_missing_from_train_is_a_violation() -> None:
    """Made to fail on purpose: drop one person's train night from the split.

    Without this the audit would pass on a split that quietly mixes an
    individualisation score and a generalisation score into one mean.
    """
    ds = FakeDataset(TWO_NIGHTS)
    split = session_split(ds, seed=0)
    victim = ds.window_subjects[split["test"][0]]
    split["train"] = [i for i in split["train"] if ds.window_subjects[i] != victim]
    rep = session_leakage_check(split, ds)
    assert not rep["ok"]
    bad = [v for v in rep["violations"] if v["kind"] == "test_participant_absent_from_train"]
    assert bad and victim in bad[0]["subjects"]


def test_the_same_session_on_both_sides_is_a_violation() -> None:
    """The other mutation: put a test window back into train."""
    ds = FakeDataset(TWO_NIGHTS)
    split = session_split(ds, seed=0)
    split["train"] = split["train"] + [split["test"][0]]
    rep = session_leakage_check(split, ds)
    assert not rep["ok"]
    assert any(v["kind"] == "session_across_folds" for v in rep["violations"])


def test_single_session_participants_are_train_only_and_not_in_the_denominator() -> None:
    """They are usable data and unusable evidence, and the report says which."""
    corpus = dict(TWO_NIGHTS)
    corpus["SC4900"] = ["night1"]  # one of sleep-edfx's three
    ds = FakeDataset(corpus)
    split = session_split(ds, seed=0)

    folds = {n: {ds.window_subjects[i] for i in idx} for n, idx in split.items()}
    assert "SC4900" in folds["train"]
    assert "SC4900" not in folds["test"] and "SC4900" not in folds["val"]

    rep = session_leakage_check(split, ds)
    assert rep["ok"], rep["violations"]
    assert rep["n_participants_total"] == 11
    assert rep["n_participants_individualisable"] == 10
    assert any("one session" in w for w in rep["warnings"])


def test_a_corpus_with_no_repeated_session_is_refused_not_split() -> None:
    """`max_runs_per_subject: 1` produces a structurally valid, empty-test split.

    Refusing is the point: the alternative returns folds that look right and
    measure nothing, which is how a capped loader gets published as a result.
    """
    ds = FakeDataset({f"SC4{i:02d}": ["night1"] for i in range(5)})
    with pytest.raises(RuntimeError, match="two or more sessions"):
        session_split(ds, seed=0)


def test_no_validation_fold_is_stated_rather_than_faked() -> None:
    """Two nights cannot support three session-disjoint folds. Say so."""
    ds = FakeDataset(TWO_NIGHTS)
    split = session_split(ds, seed=0)
    assert split["val"] == []
    rep = session_leakage_check(split, ds)
    assert any("three or more sessions" in w for w in rep["warnings"])


def test_three_sessions_do_support_a_validation_fold() -> None:
    ds = FakeDataset({f"S{i}": ["a", "b", "c"] for i in range(6)})
    split = session_split(ds, seed=0)
    assert split["val"], "a three-session corpus should yield a val fold"
    rep = session_leakage_check(split, ds)
    assert rep["ok"], rep["violations"]


def test_the_holdout_is_stable_when_another_participant_is_added() -> None:
    """Per-participant seeding, so a corpus that grows does not reshuffle nights.

    A global shuffle would silently re-draw every held-out night whenever a
    download finished, and two runs would then be scored on different data
    while reporting the same split name.
    """
    a = FakeDataset(TWO_NIGHTS)
    grown = dict(TWO_NIGHTS)
    grown["SC4999"] = ["night1", "night2"]
    b = FakeDataset(grown)

    def held_out(ds: FakeDataset, split: dict[str, list[int]]) -> set[str]:
        return {ds.window_sessions[i] for i in split["test"]}

    assert held_out(a, session_split(a, seed=0)) <= held_out(b, session_split(b, seed=0))


def test_the_real_property_qualifies_the_session_by_subject() -> None:
    """The production path, not the fake: `window_sessions` must not be bare.

    Every sleep-edfx subject has a session called ``night1``. A bare session id
    therefore collapses 78 participants into two groups, and `session_split`
    would hold out *everyone's* second night as one block -- still
    session-disjoint, so both audits pass, while the per-participant structure
    the claim rests on is gone.

    `FakeDataset` supplies the qualified list directly, so nothing above this
    exercises the property that builds it.
    """
    from scwbd.foundation.realdata import RealEEGDataset, _window_sessions

    ds = RealEEGDataset.__new__(RealEEGDataset)  # no I/O: the property reads two fields
    ds.recordings = [
        {"subject": "SC400", "session": "night1"},
        {"subject": "SC401", "session": "night1"},
    ]
    ds.window_index = [(0, 0), (0, 1), (1, 0)]

    assert ds.window_sessions == ["SC400/night1", "SC400/night1", "SC401/night1"]
    assert len(set(ds.window_sessions)) == 2, (
        "two participants' night1 collapsed into one group"
    )
    assert _window_sessions(ds) == ds.window_sessions


def test_a_subset_reindexes_sessions_the_same_way_as_subjects() -> None:
    """`Subset` is how every fold is materialised, so the two must stay aligned.

    If `_window_sessions` fell through to the `dataset[i]` path here it would
    index the *underlying* dataset, silently pairing each window with another
    window's session.
    """
    from torch.utils.data import Subset

    from scwbd.foundation.realdata import _window_sessions, _window_subjects

    base = FakeDataset(TWO_NIGHTS)
    picked = [0, 5, 11, 17]
    sub = Subset(base, picked)

    assert _window_sessions(sub) == [base.window_sessions[i] for i in picked]
    assert _window_subjects(sub) == [base.window_subjects[i] for i in picked]


def test_r10_fails_on_this_split_and_points_at_the_right_repair() -> None:
    """`leakage_check` MUST reject a session split -- every participant is shared.

    Asserted rather than avoided. The failure is correct, and the danger is that
    someone reads it as a bug in R10 and relaxes the check that protects every
    generalisation number in the project.
    """
    ds = FakeDataset(TWO_NIGHTS)
    split = session_split(ds, seed=0)
    rep = leakage_check(split, ds)
    assert not rep["ok"]
    assert any(v["kind"] == "participant_across_folds" for v in rep["violations"])
    assert any("session_leakage_check" in w for w in rep["warnings"]), (
        "R10 rejected the split without naming the alternative, so the obvious "
        "next move is to weaken R10"
    )


# ======================================================================
# the evaluation that consumes the split
# ======================================================================
def test_individualisation_refuses_rather_than_reporting_nothing() -> None:
    """An absent corpus must not read as an unsupported claim being supported.

    "fine-tuneable for personalized neurotechnology" is on the landing page and
    has never been measured. A silent absence here would let it stay there
    unexamined, which is the same shape as a guard that cannot fire.
    """
    from scwbd.foundation.evaluate import session_individualisation

    class _NoCorpus:
        eeg_datasets: dict = {}

    rep = session_individualisation(_NoCorpus(), source_id="sleepedf_real")
    assert rep["ok"] is False
    assert "NOT thereby supported" in rep["reason"], (
        "the report does not say that an unmeasurable claim is unmeasured rather "
        "than confirmed"
    )


def test_individualisation_refuses_a_split_that_does_not_hold() -> None:
    """A leaky split must raise, not be scored.

    The failure mode here is not R10's -- every participant is deliberately on
    both sides. It is the same NIGHT on both sides, which would turn a held-out
    score into a memorisation score.
    """
    from scwbd.foundation import evaluate as ev
    from scwbd.foundation import realdata as rd

    ds = FakeDataset(TWO_NIGHTS)

    class _T:
        eeg_datasets = {"sleepedf_real": ds}

    # `session_individualisation` imports the audit at CALL time, so patching it
    # on its source module is what reaches the caller.
    real_check = rd.session_leakage_check
    rd.session_leakage_check = lambda split, dataset: {
        "ok": False,
        "violations": [{"kind": "session_across_folds", "session": "SC400/night1"}],
    }
    try:
        with pytest.raises(RuntimeError, match="memorisation"):
            ev.session_individualisation(_T(), source_id="sleepedf_real")
    finally:
        rd.session_leakage_check = real_check


def test_the_theta_shift_spread_is_the_named_falsifier() -> None:
    """Zero spread means the individualizer applied nothing. Say so numerically.

    On a participant-disjoint holdout this is exactly 0.000e+00 by construction.
    Reporting it on the SESSION split is the point: if it is still zero there,
    the capability is unsupported on a split built to let it show.
    """
    import torch

    from scwbd.foundation.evaluate import _theta_shift_spread

    class _Ind:
        def __init__(self, delta):
            self.delta = delta

    unfitted = _Ind(torch.zeros(5, 6))
    rep = _theta_shift_spread(type("T", (), {"individualizer": unfitted})(), [0, 1, 2])
    assert rep["available"] and rep["spread_pooled"] == 0.0
    assert rep["n_rows_exactly_zero"] == 3

    fitted = torch.zeros(5, 6)
    fitted[0, 0], fitted[1, 0], fitted[2, 0] = 0.5, -0.5, 0.25
    rep2 = _theta_shift_spread(type("T", (), {"individualizer": _Ind(fitted)})(), [0, 1, 2])
    assert rep2["spread_pooled"] > 0.0
    assert rep2["n_rows_exactly_zero"] == 0
    assert rep2["spread_per_theta"][0] > 0.0 and rep2["spread_per_theta"][1] == 0.0, (
        "per-dimension spread is what separates 'moved one parameter' from "
        "'moved everything', and they are different findings"
    )
