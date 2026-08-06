"""Appendix D audits must catch a deliberately leaked split."""

from __future__ import annotations

import numpy as np
import pytest

from scwbd.bench.harness import Dataset
from scwbd.bench.leakage import (
    APPENDIX_D_ROWS,
    audit_dataset_family_breadth,
    audit_derived_data_duplication,
    audit_participant_family_leakage,
    audit_site_device_shortcuts,
    audit_teacher_simulator_domination,
    audit_tms_tfus_decision_claim,
    run_all_audits,
)
from scwbd.bench.synthetic import RidgeGaussian
from scwbd.sources.lineage import Lineage, Record
from scwbd.sources.splits import GroupedSplitter, leakage_audit


def _records(n_participants=8, derived=False, duplicate_hash=False):
    recs = []
    for p in range(n_participants):
        for s in range(2):
            rid = f"P{p}-S{s}"
            recs.append(
                Record(
                    id=rid,
                    source_id="fixture",
                    lineage=Lineage(
                        participant=f"P{p}", family=f"F{p}", site="siteA", device="dev1",
                        session=f"S{s}",
                        content_hash=("same" if duplicate_hash else f"h{p}{s}"),
                    ),
                    stimulus_ids=(f"stim{p % 3}",),
                )
            )
            if derived:
                recs.append(
                    Record(
                        id=rid + "-tractogram",
                        source_id="fixture",
                        lineage=Lineage(
                            participant=f"P{p}", family=f"F{p}", site="siteA",
                            device="dev1", session=f"S{s}", derived_from=rid,
                            content_hash=f"d{p}{s}",
                        ),
                        stimulus_ids=(f"stim{p % 3}",),
                    )
                )
    return recs


# --------------------------------------------------------------------------
def test_all_twelve_appendix_d_rows_are_implemented():
    assert len(APPENDIX_D_ROWS) == 12
    reports = run_all_audits()
    assert len(reports) == 12
    assert {r.manifest.claim_id for r in reports} == set(APPENDIX_D_ROWS)


def test_participant_audit_passes_a_clean_grouped_split():
    rep = audit_participant_family_leakage(records=_records(), n_folds=4, seed=0)
    grouped = next(s for s in rep.subchecks if s.name == "grouped_split")
    assert grouped.status == "PASS"
    assert rep.artifacts["split"]["level"] == "family"


def test_participant_audit_catches_an_intentionally_leaked_split():
    """Hand-build a split that puts one participant on both sides."""
    recs = _records(n_participants=6)
    split = GroupedSplitter(mode="participant", n_folds=3, seed=0).split(recs)
    fold0 = split.folds[0]
    leaked_id = fold0.test_ids[0]
    bad_fold = type(fold0)(
        index=0,
        train_ids=tuple(list(fold0.train_ids) + [leaked_id]),   # <- the leak
        test_ids=fold0.test_ids,
        test_groups=fold0.test_groups,
    )
    bad_split = type(split)(
        mode=split.mode, level=split.level, seed=split.seed,
        folds=(bad_fold,) + split.folds[1:], group_of=split.group_of,
    )
    report = leakage_audit(bad_split, recs)
    assert not report.ok, "agent B's audit must reject a train/test id overlap"


def test_derived_records_are_grouped_with_their_parent():
    rep = audit_derived_data_duplication(records=_records(derived=True), n_folds=4, seed=0)
    sub = next(s for s in rep.subchecks if s.name == "hash_lineage_audit")
    assert sub.status == "PASS"
    assert rep.artifacts["lineage"]["n_derived"] > 0


def test_duplicate_content_hashes_are_flagged():
    rep = audit_derived_data_duplication(records=_records(duplicate_hash=True), n_folds=4,
                                         seed=0)
    assert rep.artifacts["lineage"]["duplicate_content_hashes"] > 0
    sub = next(s for s in rep.subchecks if s.name == "hash_lineage_audit")
    assert sub.status == "FAIL"


def test_retrieval_audit_catches_near_duplicate_records():
    rng = np.random.default_rng(0)
    train = rng.normal(0, 1, size=(200, 8))
    leaky_test = train[:50] + 1e-9         # the same scans under new accession numbers
    clean_test = rng.normal(0, 1, size=(50, 8))
    leaky = audit_participant_family_leakage(train_embeddings=train,
                                             test_embeddings=leaky_test, seed=0)
    clean = audit_participant_family_leakage(train_embeddings=train,
                                             test_embeddings=clean_test, seed=0)
    lsub = next(s for s in leaky.subchecks if s.name == "retrieval_audit")
    csub = next(s for s in clean.subchecks if s.name == "retrieval_audit")
    assert lsub.status == "FAIL"
    assert csub.status == "PASS"


def test_nuisance_only_classifier_detects_a_site_shortcut():
    rng = np.random.default_rng(1)
    n = 400
    label = np.array(["a"] * (n // 2) + ["b"] * (n // 2))
    shortcut = np.where(label == "a", 1.0, -1.0)[:, None] + rng.normal(0, 0.2, size=(n, 1))
    innocuous = rng.normal(0, 1, size=(n, 1))
    bad = audit_site_device_shortcuts(nuisance_features=shortcut, nuisance_labels=label,
                                      seed=0)
    good = audit_site_device_shortcuts(nuisance_features=innocuous, nuisance_labels=label,
                                       seed=0)
    assert next(s for s in bad.subchecks
                if s.name == "nuisance_only_classifier").status == "FAIL"
    assert next(s for s in good.subchecks
                if s.name == "nuisance_only_classifier").status == "PASS"


def test_within_site_permutation_null_is_checked():
    rng = np.random.default_rng(2)
    null = rng.normal(0, 1, size=200)
    real = audit_site_device_shortcuts(
        permutation_scores={"observed": [4.0], "permuted": null}, seed=0)
    fake = audit_site_device_shortcuts(
        permutation_scores={"observed": [0.0], "permuted": null}, seed=0)
    assert next(s for s in real.subchecks
                if s.name == "within_site_label_permutation").status == "PASS"
    assert next(s for s in fake.subchecks
                if s.name == "within_site_label_permutation").status == "FAIL"


def test_dataset_family_breadth_detects_negative_transfer():
    rng = np.random.default_rng(3)
    n = 400
    useful = rng.normal(0, 1, size=(n, 2))
    junk = rng.normal(0, 1, size=(n, 6))
    y = useful @ np.array([1.0, -1.0]) + rng.normal(0, 0.3, size=n)
    groups = np.array([f"P{i//40}" for i in range(n)])
    ds = Dataset(name="fam", targets=y, inputs={"useful": useful, "junk": junk},
                 strata={"site": np.array(["s"] * n)}, groups=groups)
    train = ds.subset(np.arange(0, 200), name="fam.train")
    test = ds.subset(np.arange(200, 400), name="fam.test")
    rep = audit_dataset_family_breadth(
        train=train, test=test,
        model_factory=lambda blocks: RidgeGaussian(name="+".join(blocks), blocks=blocks,
                                                   alpha=1e-6),
        families={"empirical": ["useful"], "synthetic": ["junk"]},
        roles={"empirical": "likelihood", "synthetic": "distillation"},
        seed=0,
    )
    contrib = rep.artifacts["contributions"]
    assert contrib["empirical"]["contribution"] > 0.1
    # the junk family should not be reported as a contribution
    assert contrib["synthetic"]["contribution"] < 0.05


def test_tms_decision_claim_is_a_standing_refusal():
    """No inputs can make this run: the dataset it needs is not held."""
    rep = audit_tms_tfus_decision_claim(seed=0, anything="supplied", more_data=[1, 2, 3])
    assert rep.status == "COULD_NOT_RUN"
    reason = " ".join(rep.blocking_reasons)
    assert "UNSUPPORTABLE BY CONSTRUCTION" in reason
    assert "No such dataset is held" in reason
    assert "not wellness or treatment efficacy" in " ".join(rep.notes)


def test_teacher_audit_is_off_by_default():
    rep = audit_teacher_simulator_domination(seed=0)
    assert rep.status == "COULD_NOT_RUN"
    assert "OFF by default" in " ".join(rep.blocking_reasons)


def test_delegated_rows_say_they_are_delegated():
    """A delegated row and its gate are one experiment, not two."""
    reps = {r.manifest.claim_id: r for r in run_all_audits()}
    for rid in ("D05_scale_hallucination", "D07_connectome_prior_value",
                "D09_individualization_claim"):
        assert any("Delegated to gate" in n for n in reps[rid].notes)
        assert any("double-counted" in n for n in reps[rid].notes)
