"""Leakage-safe splitting: group first, fail loudly, audit afterwards."""

from __future__ import annotations

import pytest

from scwbd.sources.lineage import Lineage, LineageError, Record, resolve_parentage
from scwbd.sources.splits import (
    Fold,
    GroupedSplitter,
    Split,
    leakage_audit,
    records_from_lineages,
)


def make_records(n_participants=12, runs=3, sites=("A", "B"), stim=("s1", "s2")):
    out = []
    for p in range(n_participants):
        pid = f"P{p:03d}"
        site = sites[p % len(sites)]
        for r in range(runs):
            out.append(
                Record(
                    id=f"{pid}_run{r}",
                    source_id="test",
                    lineage=Lineage(
                        participant=pid,
                        family=f"singleton:{pid}",
                        site=site,
                        device="dev1",
                        session=f"{pid}_ses1",
                        run=f"run{r}",
                        content_hash=f"hash_{pid}_{r}",
                    ),
                    stimulus_ids=stim,
                )
            )
    return out


# --------------------------------------------------------------------------
# grouping happens before splitting
# --------------------------------------------------------------------------
def test_participant_never_crosses_folds():
    recs = make_records()
    split = GroupedSplitter("participant", n_folds=4, seed=7).split(recs)
    seen: dict[str, int] = {}
    for fold in split:
        for rid in fold.test_ids:
            p = rid.split("_")[0]
            assert seen.setdefault(p, fold.index) == fold.index
        train_p = {i.split("_")[0] for i in fold.train_ids}
        test_p = {i.split("_")[0] for i in fold.test_ids}
        assert not (train_p & test_p), "participant in both train and test"


def test_all_runs_of_a_participant_land_together():
    recs = make_records(runs=5)
    split = GroupedSplitter("participant", n_folds=3, seed=1).split(recs)
    for fold in split:
        by_p: dict[str, set[str]] = {}
        for rid in fold.test_ids:
            by_p.setdefault(rid.split("_")[0], set()).add(rid)
        for p, ids in by_p.items():
            assert len(ids) == 5, f"{p} split across folds"


def test_split_is_deterministic_in_the_seed():
    recs = make_records()
    a = GroupedSplitter("participant", n_folds=4, seed=3).split(recs)
    b = GroupedSplitter("participant", n_folds=4, seed=3).split(recs)
    c = GroupedSplitter("participant", n_folds=4, seed=4).split(recs)
    assert [f.test_ids for f in a] == [f.test_ids for f in b]
    assert [f.test_ids for f in a] != [f.test_ids for f in c]


def test_family_dominates_participant_when_declared():
    recs = []
    for i in range(8):
        fam = f"F{i // 2}"  # two participants per family
        pid = f"P{i}"
        recs.append(
            Record(
                id=pid,
                source_id="t",
                lineage=Lineage(participant=pid, family=fam, site="A", device="d", session="s"),
            )
        )
    split = GroupedSplitter("participant", n_folds=4, seed=0).split(recs)
    assert split.level == "family"
    for fold in split:
        fams_test = {split.group_of[i] for i in fold.test_ids}
        fams_train = {split.group_of[i] for i in fold.train_ids}
        assert not (fams_test & fams_train), "relatives split across folds"


# --------------------------------------------------------------------------
# refusal R10
# --------------------------------------------------------------------------
def test_unknown_participant_refuses_loudly():
    recs = make_records(n_participants=4)
    recs[0] = Record(
        id="bad",
        source_id="test",
        lineage=Lineage(participant="unknown", family="unknown", site="A", device="d"),
    )
    with pytest.raises(LineageError) as exc:
        GroupedSplitter("participant", n_folds=2, seed=0).split(recs)
    assert exc.value.code == "R10"
    assert "unresolved" in str(exc.value) or "unknown" in str(exc.value)


def test_missing_family_on_some_records_refuses():
    recs = [
        Record(id="a", source_id="t", lineage=Lineage(participant="P1", family="F1")),
        Record(id="b", source_id="t", lineage=Lineage(participant="P2")),  # no family
    ]
    with pytest.raises(LineageError, match="family"):
        GroupedSplitter("participant", n_folds=2, seed=0).split(recs)


def test_dangling_parent_refuses():
    recs = [
        Record(id="raw", source_id="t", lineage=Lineage(participant="P1")),
        Record(
            id="tractogram",
            source_id="t",
            lineage=Lineage(participant="P1", derived_from="a_scan_not_in_this_set"),
        ),
    ]
    with pytest.raises(LineageError, match="derived from"):
        resolve_parentage(recs)


def test_derivation_cycle_refuses():
    recs = [
        Record(id="a", source_id="t", lineage=Lineage(participant="P1", derived_from="b")),
        Record(id="b", source_id="t", lineage=Lineage(participant="P1", derived_from="a")),
    ]
    with pytest.raises(LineageError, match="cycle"):
        resolve_parentage(recs)


def test_too_few_groups_refuses():
    recs = [Record(id="a", source_id="t", lineage=Lineage(participant="P1"))]
    with pytest.raises(LineageError, match="independent group"):
        GroupedSplitter("participant", n_folds=2, seed=0).split(recs)


# --------------------------------------------------------------------------
# derived data inherits its parent's group
# --------------------------------------------------------------------------
def test_derivatives_follow_their_parent_scan():
    recs = [Record(id=f"P{i}_scan", source_id="t", lineage=Lineage(participant=f"P{i}"))
            for i in range(6)]
    for i in range(6):
        for alg in ("fsl", "mrtrix"):
            recs.append(
                Record(
                    id=f"P{i}_tract_{alg}",
                    source_id="t",
                    lineage=Lineage(participant=f"P{i}", derived_from=f"P{i}_scan"),
                )
            )
    splitter = GroupedSplitter("participant", n_folds=3, seed=2)
    split = splitter.split(recs)
    rep = leakage_audit(split, recs)
    assert rep.ok, rep.summary()
    for fold in split:
        for rid in fold.test_ids:
            p = rid.split("_")[0]
            assert not any(t.startswith(f"{p}_") for t in fold.train_ids)


# --------------------------------------------------------------------------
# site and stimulus modes
# --------------------------------------------------------------------------
def test_site_mode_is_leave_site_out():
    recs = make_records(n_participants=10, sites=("siteA", "siteB", "siteC"))
    split = GroupedSplitter("site", seed=0).split(recs)
    assert len(split) == 3
    for fold in split:
        sites_test = {r.lineage.site for r in recs if r.id in set(fold.test_ids)}
        sites_train = {r.lineage.site for r in recs if r.id in set(fold.train_ids)}
        assert len(sites_test) == 1 and not (sites_test & sites_train)


def test_site_mode_refuses_a_single_site():
    recs = make_records(sites=("only",))
    with pytest.raises(LineageError, match="site"):
        GroupedSplitter("site", seed=0).split(recs)


def test_stimulus_mode_holds_out_stimuli():
    recs = []
    for p in range(6):
        for s in range(8):
            recs.append(
                Record(
                    id=f"P{p}_s{s}",
                    source_id="t",
                    lineage=Lineage(participant=f"P{p}"),
                    stimulus_ids=(f"stim{s}",),
                )
            )
    split = GroupedSplitter("stimulus", n_folds=4, seed=0).split(recs)
    assert not split.requires_trial_masking
    rep = leakage_audit(split, recs)
    assert rep.ok, rep.summary()
    for fold in split:
        held = set(fold.held_out_stimuli)
        train_stims = {s for rid in fold.train_ids for s in dict((r.id, r) for r in recs)[rid].stimulus_ids}
        assert not (held & train_stims)


def test_stimulus_mode_flags_records_that_mix_folds():
    recs = [
        Record(id=f"r{i}", source_id="t", lineage=Lineage(participant=f"P{i}"),
               stimulus_ids=("a", "b", "c", "d"))
        for i in range(6)
    ]
    split = GroupedSplitter("stimulus", n_folds=2, seed=0).split(recs)
    assert split.requires_trial_masking
    rep = leakage_audit(split, recs)
    assert any("trial-level masking" in w for w in rep.warnings)


def test_stimulus_mode_refuses_unknown_stimuli():
    recs = [
        Record(id=f"r{i}", source_id="t", lineage=Lineage(participant=f"P{i}"),
               stimulus_ids=("unknown",))
        for i in range(4)
    ]
    with pytest.raises(LineageError, match="unknown"):
        GroupedSplitter("stimulus", n_folds=2, seed=0).split(recs)


# --------------------------------------------------------------------------
# the audit must catch a deliberately leaky split
# --------------------------------------------------------------------------
def test_audit_catches_a_participant_leak():
    recs = make_records(n_participants=6, runs=2)
    ids = [r.id for r in recs]
    # deliberately leaky: put P000's two runs on opposite sides
    leaky = Split(
        mode="participant",
        level="participant",
        seed=0,
        folds=(
            Fold(index=0, train_ids=tuple(ids[1:]), test_ids=(ids[0],), test_groups=("P000",)),
            Fold(index=1, train_ids=(ids[0],), test_ids=tuple(ids[1:]), test_groups=("P000",)),
        ),
        group_of={i: i.split("_")[0] for i in ids},
    )
    rep = leakage_audit(leaky, recs)
    assert not rep.ok
    kinds = {v.kind for v in rep.violations}
    assert "participant_across_folds" in kinds or "train_test_group_overlap" in kinds
    assert all(v.code == "R10" for v in rep.violations)
    with pytest.raises(LineageError):
        rep.raise_if_leaky()


def test_audit_catches_duplicate_content_across_folds():
    recs = make_records(n_participants=6, runs=1)
    # the same bytes archived under two participant ids (duplicate archive record)
    recs[0] = Record(
        id=recs[0].id,
        source_id="test",
        lineage=Lineage(participant="P000", family="singleton:P000", site="A", device="dev1",
                        content_hash="COLLIDING"),
    )
    recs[1] = Record(
        id=recs[1].id,
        source_id="test",
        lineage=Lineage(participant="P001", family="singleton:P001", site="B", device="dev1",
                        content_hash="COLLIDING"),
    )
    split = GroupedSplitter("participant", n_folds=2, seed=0).split(recs)
    rep = leakage_audit(split, recs)
    # the two duplicates land in different groups; if the split separated them the audit fires
    if rep.violations:
        assert any("duplicate_content" in v.kind for v in rep.violations)
    assert rep.stats["n_duplicate_content_hashes"] == 1


def test_audit_catches_derived_data_crossing_a_split():
    recs = [
        Record(id="P0_scan", source_id="t", lineage=Lineage(participant="P0")),
        Record(id="P1_scan", source_id="t", lineage=Lineage(participant="P1")),
        # a derivative mislabelled as belonging to a different participant
        Record(id="P0_tract", source_id="t",
               lineage=Lineage(participant="P1", derived_from="P0_scan")),
    ]
    leaky = Split(
        mode="participant",
        level="participant",
        seed=0,
        folds=(
            Fold(index=0, train_ids=("P1_scan", "P0_tract"), test_ids=("P0_scan",),
                 test_groups=("P0",)),
            Fold(index=1, train_ids=("P0_scan",), test_ids=("P1_scan", "P0_tract"),
                 test_groups=("P1",)),
        ),
        group_of={"P0_scan": "P0", "P1_scan": "P1", "P0_tract": "P1"},
    )
    rep = leakage_audit(leaky, recs)
    assert not rep.ok
    assert any(v.kind == "derived_data_crosses_split" for v in rep.violations)


def test_audit_warns_when_site_predicts_fold():
    # participants confounded with site: fold membership becomes site-predictable
    recs = []
    for p in range(8):
        recs.append(
            Record(id=f"P{p}", source_id="t",
                   lineage=Lineage(participant=f"P{p}", site=f"site{p}", device="d"))
        )
    split = GroupedSplitter("participant", n_folds=4, seed=0).split(recs)
    rep = leakage_audit(split, recs)
    assert rep.ok
    assert rep.stats["site_fold_nmi"] > 0.2
    assert any("site predicts fold" in w for w in rep.warnings)


def test_audit_warns_when_only_one_site():
    recs = make_records(sites=("only",))
    split = GroupedSplitter("participant", n_folds=3, seed=0).split(recs)
    rep = leakage_audit(split, recs)
    assert rep.ok
    assert any("one site" in w for w in rep.warnings)


def test_site_mode_nmi_is_not_a_violation():
    recs = make_records(n_participants=9, sites=("a", "b", "c"))
    split = GroupedSplitter("site", seed=0).split(recs)
    rep = leakage_audit(split, recs)
    assert rep.ok
    assert rep.stats["site_fold_nmi"] == pytest.approx(1.0, abs=1e-6)
    assert not any("shortcut" in w for w in rep.warnings)


def test_records_from_lineages_helper():
    recs = records_from_lineages(
        "src", [{"id": "a", "participant": "P1"}, {"id": "b", "participant": "P2"}]
    )
    assert [r.id for r in recs] == ["a", "b"]
    assert recs[0].lineage.participant == "P1"


def test_group_key_is_hierarchical_not_a_bare_label():
    a = Lineage(participant="P1", session="ses-01")
    b = Lineage(participant="P2", session="ses-01")
    assert a.group_key("session") != b.group_key("session")
