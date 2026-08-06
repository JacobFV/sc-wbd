"""Immutable lineage and the leakage barrier (rule 6, refusal R10)."""

from __future__ import annotations

import pytest

from scwbd.schema import Identity, LineageError, LineageGraph, LineageUnit
from scwbd.schema.examples import build_three_region_schema


def _ident(pid: str, parents: tuple[str, ...] = ()) -> Identity:
    return Identity(persistent_id=pid, version="1.0.0", parent_ids=parents)


def test_lineage_closure_is_transitive():
    registry = {
        "root": _ident("root"),
        "mid": _ident("mid", ("root",)),
        "leaf": _ident("leaf", ("mid",)),
    }
    assert registry["leaf"].lineage_closure(registry) == {"mid", "root"}
    assert registry["mid"].lineage_closure(registry) == {"root"}
    assert registry["root"].lineage_closure(registry) == frozenset()


def test_lineage_closure_handles_diamonds_and_cycles():
    registry = {
        "a": _ident("a", ("b", "c")),
        "b": _ident("b", ("d",)),
        "c": _ident("c", ("d",)),
        "d": _ident("d", ("a",)),  # a cycle; must terminate, not hang
    }
    assert registry["a"].lineage_closure(registry) == {"a", "b", "c", "d"}


def test_lineage_closure_without_registry_returns_direct_parents():
    ident = _ident("leaf", ("mid",))
    assert ident.lineage_closure() == {"mid"}


def test_strict_closure_fails_on_unresolved_parentage():
    ident = _ident("leaf", ("ghost",))
    with pytest.raises(LineageError, match="unresolved"):
        ident.lineage_closure({}, strict=True)
    assert ident.unresolved_parents({}) == ("ghost",)


def test_identity_rejects_self_parentage_and_bad_hashes():
    with pytest.raises(ValueError, match="its own parent"):
        Identity(persistent_id="x", version="1", parent_ids=("x",))
    with pytest.raises(ValueError, match="hex digest"):
        Identity(persistent_id="x", version="1", file_hashes={"f": "not a hash"})
    with pytest.raises(ValueError, match="version"):
        Identity(persistent_id="x", version="  ")


def test_reproducibility_flags():
    mutable = Identity(persistent_id="x", version="1", mutable_download=True)
    assert not mutable.is_reproducible
    assert "mutable_download" in mutable.missing_reproducibility_fields()


def test_lineage_graph_groups_sessions_with_their_participant():
    units = [
        LineageUnit(id="sub-01", kind="participant"),
        LineageUnit(id="sub-01_ses-01", kind="session", parent_ids=("sub-01",)),
        LineageUnit(id="sub-01_ses-02", kind="session", parent_ids=("sub-01",)),
        LineageUnit(id="sub-02", kind="participant"),
    ]
    graph = LineageGraph(units)
    assert graph.group_of("sub-01_ses-01") == graph.group_of("sub-01_ses-02")
    assert graph.group_of("sub-01") != graph.group_of("sub-02")
    assert graph.ancestors("sub-01_ses-01") == {"sub-01"}
    assert len(graph.groups()) == 2


def test_relatives_are_grouped_too():
    units = [
        LineageUnit(id="fam-1", kind="family"),
        LineageUnit(id="sub-a", kind="participant", related_ids=("fam-1",)),
        LineageUnit(id="sub-b", kind="participant", related_ids=("fam-1",)),
    ]
    graph = LineageGraph(units)
    assert graph.group_of("sub-a") == graph.group_of("sub-b")


def test_crossing_assignments_are_detected():
    units = [
        LineageUnit(id="sub-01", kind="participant"),
        LineageUnit(id="sub-01_ses-01", kind="session", parent_ids=("sub-01",)),
        LineageUnit(id="sub-01_ses-02", kind="session", parent_ids=("sub-01",)),
    ]
    graph = LineageGraph(units)
    ok = {"sub-01": "train", "sub-01_ses-01": "train", "sub-01_ses-02": "train"}
    assert graph.crossing_assignments(ok) == ()
    bad = dict(ok, **{"sub-01_ses-02": "test"})
    crossings = graph.crossing_assignments(bad)
    assert len(crossings) == 1
    assert crossings[0][1] == ("test", "train")


def test_unresolved_parents_are_reported():
    graph = LineageGraph([LineageUnit(id="s1", kind="session", parent_ids=("ghost",))])
    assert graph.unresolved_parents() == (("s1", "ghost"),)


def test_unassigned_units_are_reported():
    graph = LineageGraph([LineageUnit(id="a", kind="participant")])
    assert graph.unassigned({}) == ("a",)
    with pytest.raises(LineageError, match="unknown lineage unit"):
        graph.crossing_assignments({"ghost": "train"})


def test_example_schema_lineage_is_clean():
    schema = build_three_region_schema()
    graph = schema.lineage_graph()
    assert graph.unresolved_parents() == ()
    # three participants, each with its own sessions
    assert len(graph.groups()) == 3
    assignments = {}
    for source in schema.sources:
        assignments.update(source.split_policy.fold_assignments)
    assert graph.crossing_assignments(assignments) == ()
    assert graph.unassigned(assignments) == ()


def test_group_ids_are_deterministic():
    units = [
        LineageUnit(id="z", kind="participant"),
        LineageUnit(id="a", kind="session", parent_ids=("z",)),
    ]
    first = LineageGraph(units).groups()
    second = LineageGraph(list(reversed(units))).groups()
    assert first == second
