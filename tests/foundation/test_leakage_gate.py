"""The leakage barrier must refuse, not report.

A participant on both sides of a split turns memorisation of that individual
into a reported generalisation (refusal R10). Unlike the corpus limitations it
cannot be caveated afterwards — it invalidates every held-out number.

The routine existed in `realdata.py` and was simply never called: `train.py`
built a split and went straight to training. Stage III was gated by a
coordinator remembering to ask, which worked exactly once.

**Every test here breaks something on purpose.** A gate nobody has watched fail
is the thing this project's register is about.
"""

from __future__ import annotations

import numpy as np
import pytest

from scwbd.foundation.realdata import leakage_check


class _FakeEEG:
    """Minimal stand-in exposing what `leakage_check` recomputes from."""

    source = "fake_eeg"

    def __init__(self, subjects_per_window):
        self._subj = list(subjects_per_window)
        self.participant_split_backend = "grouped_splitter"
        self.participant_split_fallback_reason = ""

    def __len__(self):
        return len(self._subj)

    # `_window_subjects` looks for these in turn
    @property
    def window_subjects(self):
        return list(self._subj)


def _clean_split():
    """6 participants, 4 windows each, folds disjoint by participant."""
    subj = [f"S{p:02d}" for p in range(6) for _ in range(4)]
    ds = _FakeEEG(subj)
    idx = {s: [i for i, x in enumerate(subj) if x == s] for s in sorted(set(subj))}
    split = {
        "train": idx["S00"] + idx["S01"] + idx["S02"],
        "val": idx["S03"],
        "test": idx["S04"] + idx["S05"],
    }
    return ds, split, idx


# ----------------------------------------------------------------------
# the gate fires
# ----------------------------------------------------------------------
def test_leaked_participant_is_caught():
    """The failure the gate exists for: one person on both sides."""
    ds, split, idx = _clean_split()
    split["test"] = split["test"] + idx["S00"]  # S00 is already in train

    rep = leakage_check(split, ds)
    assert rep["ok"] is False, "a participant in two folds must fail the audit"
    kinds = {v["kind"] for v in rep["violations"]}
    assert "participant_across_folds" in kinds
    assert any(v.get("code") == "R10" for v in rep["violations"])
    offenders = {v.get("subject") for v in rep["violations"] if "subject" in v}
    assert "S00" in offenders


def test_a_single_leaked_window_is_enough():
    """Not a threshold — one window of one participant crossing is a failure."""
    ds, split, idx = _clean_split()
    split["val"] = split["val"] + [idx["S05"][0]]  # S05 is in test

    rep = leakage_check(split, ds)
    assert rep["ok"] is False
    assert "participant_across_folds" in {v["kind"] for v in rep["violations"]}


def test_duplicate_window_index_is_caught():
    ds, split, _ = _clean_split()
    split["train"] = split["train"] + [split["train"][0]]
    rep = leakage_check(split, ds)
    assert rep["ok"] is False
    assert "duplicate_window_index" in {v["kind"] for v in rep["violations"]}


def test_out_of_range_index_is_caught():
    ds, split, _ = _clean_split()
    split["test"] = split["test"] + [10_000]
    rep = leakage_check(split, ds)
    assert rep["ok"] is False
    assert "index_out_of_range" in {v["kind"] for v in rep["violations"]}


# ----------------------------------------------------------------------
# ...and passes what it should
# ----------------------------------------------------------------------
def test_clean_split_passes_with_zero_overlap():
    ds, split, _ = _clean_split()
    rep = leakage_check(split, ds)
    assert rep["ok"] is True, rep["violations"]
    assert rep["violations"] == []
    per = rep["subjects_per_fold"]
    assert set(per["train"]) & set(per["val"]) == set()
    assert set(per["train"]) & set(per["test"]) == set()
    assert set(per["val"]) & set(per["test"]) == set()
    assert rep["n_subjects_total"] == 6


def test_report_recomputes_participants_rather_than_trusting_the_split():
    """The audit must derive membership from the dataset, not from the caller.

    Otherwise it can only catch mistakes made by `participant_split`, and not
    hand-edited splits, concatenations, or off-by-one index bugs.
    """
    ds, split, idx = _clean_split()
    rep = leakage_check(split, ds)
    assert sorted(rep["subjects_per_fold"]["val"]) == ["S03"]
    # move a window without telling the audit anything about participants
    split["train"] = split["train"] + idx["S03"]
    rep2 = leakage_check(split, ds)
    assert rep2["ok"] is False


# ----------------------------------------------------------------------
# the trainer refuses, rather than reporting
# ----------------------------------------------------------------------
def test_trainer_gate_raises_on_a_leaked_split():
    """`_audit_real_split` must raise. Warning is not enough at this boundary."""
    from types import SimpleNamespace

    from scwbd.foundation.train import FoundationTrainer

    ds, split, idx = _clean_split()
    split["test"] = split["test"] + idx["S00"]
    stub = SimpleNamespace(sources={}, leakage_audit=None)

    with pytest.raises(RuntimeError, match="leakage audit FAILED"):
        FoundationTrainer._audit_real_split(stub, split, ds)


def test_trainer_gate_rejects_a_non_lineage_aware_backend():
    """A disjoint split is not the same as one constructed to be disjoint.

    R10 requires grouping by immutable lineage *before* splitting, so a hash
    split must be refused even when it happens to contain no leak.
    """
    from types import SimpleNamespace

    from scwbd.foundation.train import FoundationTrainer

    ds, split, _ = _clean_split()
    ds.participant_split_backend = "hash_fallback"
    ds.participant_split_fallback_reason = "ImportError: no GroupedSplitter"
    stub = SimpleNamespace(sources={}, leakage_audit=None)

    with pytest.raises(RuntimeError, match="not 'grouped_splitter'"):
        FoundationTrainer._audit_real_split(stub, split, ds)


def test_trainer_gate_passes_a_clean_grouped_split_and_marks_the_sources():
    """On success the audit must be what licenses `leakage_checked`."""
    from types import SimpleNamespace

    from scwbd.foundation.mixture import SourceSpec
    from scwbd.foundation.train import FoundationTrainer

    ds, split, _ = _clean_split()
    sources = {
        "real": SourceSpec(id="real", role="likelihood", losses=("likelihood",)),
        "sim": SourceSpec(id="sim", role="prior", losses=("prior",), is_simulated=True),
    }
    stub = SimpleNamespace(sources=sources, leakage_audit=None)

    audit = FoundationTrainer._audit_real_split(stub, split, ds)
    assert audit["ok"] is True
    assert stub.sources["real"].leakage_audited is True
    assert stub.sources["sim"].leakage_audited is False, (
        "a simulated source has no participants; asserting a leakage check over one "
        "is the same empty claim in a different place"
    )


# ----------------------------------------------------------------------
# the schema must not assert what nothing established
# ----------------------------------------------------------------------
def test_leakage_checked_is_false_until_an_audit_runs():
    """`compiler_bridge` previously hard-coded True on every observation card."""
    cb = pytest.importorskip("scwbd.foundation.compiler_bridge")
    if not cb.compiler_available():
        pytest.skip("scwbd.compiler unavailable")

    from pathlib import Path

    from scwbd.foundation.anatomy import load_anatomy
    from scwbd.foundation.config import load_config
    from scwbd.foundation.mixture import SourceSpec

    repo = Path(__file__).resolve().parents[2]
    cfg = load_config(repo / "configs" / "scwbd_001_beta.yaml")
    anat = load_anatomy(device="cpu", n_cortex=40, n_subcortex=12, n_cerebellum=8,
                        density=0.15, seed=7)
    specs = SourceSpec.load_dir(cfg.mixture_cards)

    unaudited = [SourceSpec(**{**s.as_dict(), "leakage_audited": False})
                 for s in specs.values()]
    schema = cb.build_foundation_schema(anat, unaudited)
    flags = [c.observation.leakage_checked for c in schema.sources if c.observation]
    assert flags and not any(flags), (
        "no audit has run, so no observation card may claim leakage_checked=True"
    )

    audited = [SourceSpec(**{**s.as_dict(), "leakage_audited": True})
               for s in specs.values()]
    schema2 = cb.build_foundation_schema(anat, audited)
    flags2 = [c.observation.leakage_checked for c in schema2.sources if c.observation]
    assert all(flags2), "an audited source must be able to say so"
