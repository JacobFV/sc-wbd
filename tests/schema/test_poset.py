"""The resolution poset is a real partial order (thesis sec. 2.6).

"Some pairs have no defensible map."  A resolution *index* cannot express that;
a poset can, and incomparability is what stops the compiler inventing a
restriction between a spectral band and a cortical parcel.
"""

from __future__ import annotations

import pytest

from scwbd.schema import (
    MapSpec,
    ResolutionPoset,
    ScaleId,
    ScaleMapPair,
    ScaleNode,
    UncertaintyLedger,
)
from scwbd.schema.examples import build_three_region_schema


def _nodes(*ids: str) -> tuple[ScaleNode, ...]:
    return tuple(ScaleNode(id=ScaleId(i)) for i in ids)


def test_reflexive_transitive_antisymmetric():
    poset = ResolutionPoset(
        nodes=_nodes("a", "b", "c"),
        relations=((ScaleId("a"), ScaleId("b")), (ScaleId("b"), ScaleId("c"))),
    )
    assert poset.leq("a", "a")  # reflexive
    assert poset.leq("a", "c")  # transitive
    assert not poset.leq("c", "a")  # antisymmetric
    assert poset.lt("a", "c")
    assert not poset.lt("a", "a")


def test_cyclic_relations_are_not_a_poset():
    with pytest.raises(ValueError, match="cyclic"):
        ResolutionPoset(
            nodes=_nodes("a", "b"),
            relations=((ScaleId("a"), ScaleId("b")), (ScaleId("b"), ScaleId("a"))),
        )


def test_incomparability_is_first_class():
    poset = ResolutionPoset(
        nodes=_nodes("vertex", "parcel", "band"),
        relations=((ScaleId("vertex"), ScaleId("parcel")),),
    )
    assert poset.incomparable("band", "parcel")
    assert poset.incomparable("parcel", "band")
    assert not poset.comparable("band", "vertex")
    assert not poset.incomparable("vertex", "parcel")
    assert ("band", "parcel") in poset.incomparable_pairs() or (
        "parcel", "band"
    ) in poset.incomparable_pairs()


def test_example_poset_has_the_expected_order_and_incomparabilities():
    poset = build_three_region_schema().resolution_poset
    assert poset.leq("surface_vertex", "parcel")
    assert poset.leq("surface_vertex", "network")  # transitive
    assert not poset.leq("network", "parcel")
    # A frequency band and a task event window are not coarser or finer than a
    # cortical parcel; they are simply other supports.
    assert poset.incomparable("spectral_band", "parcel")
    assert poset.incomparable("event_window", "surface_vertex")
    assert poset.incomparable("spectral_band", "event_window")


def test_bounds():
    poset = build_three_region_schema().resolution_poset
    assert "network" in poset.upper_bounds("surface_vertex")
    assert "surface_vertex" in poset.lower_bounds("network")
    assert poset.upper_bounds("spectral_band") == ()


def test_relations_must_reference_declared_scales():
    with pytest.raises(ValueError, match="undeclared scale"):
        ResolutionPoset(nodes=_nodes("a"), relations=((ScaleId("a"), ScaleId("ghost")),))


def test_registered_restriction_prolongation_pair():
    poset = build_three_region_schema().resolution_poset
    pair = poset.map_pair("surface_vertex", "parcel")
    assert pair is not None
    assert pair.restriction is not None and pair.prolongation is not None
    assert pair.prolongation.returns_distribution
    assert pair.coverage_tested()
    assert poset.orphan_prolongations() == ()


def test_prolongation_without_restriction_is_an_orphan():
    poset = ResolutionPoset(
        nodes=_nodes("fine", "coarse"),
        relations=((ScaleId("fine"), ScaleId("coarse")),),
        maps=(
            ScaleMapPair(
                fine=ScaleId("fine"),
                coarse=ScaleId("coarse"),
                prolongation=MapSpec(name="upsample", kind="prolongation"),
                roundtrip_tested=False,
            ),
        ),
    )
    orphans = poset.orphan_prolongations()
    assert len(orphans) == 1
    assert orphans[0].restriction is None


def test_map_between_incomparable_scales_is_detected():
    poset = ResolutionPoset(
        nodes=_nodes("parcel", "band"),
        maps=(
            ScaleMapPair(
                fine=ScaleId("band"),
                coarse=ScaleId("parcel"),
                restriction=MapSpec(name="bogus", kind="restriction"),
            ),
        ),
    )
    assert len(poset.unordered_maps()) == 1


def test_map_slots_are_type_checked():
    with pytest.raises(ValueError, match="restriction slot"):
        ScaleMapPair(
            fine=ScaleId("a"),
            coarse=ScaleId("b"),
            restriction=MapSpec(name="wrong", kind="prolongation"),
        )


def test_gluing_certificate_names_the_failed_path():
    from scwbd.schema import CocycleCheck, GluingPolicy

    gluing = GluingPolicy(
        materialize_global=True,
        cocycle_checks=(
            CocycleCheck(
                path=(ScaleId("a"), ScaleId("b"), ScaleId("c")), residual=2.0, tolerance=0.5
            ),
            CocycleCheck(
                path=(ScaleId("a"), ScaleId("b"), ScaleId("d")), residual=0.1, tolerance=0.5
            ),
        ),
    )
    cert = gluing.certificate()
    assert cert.is_obstructed
    assert cert.failed_paths == (("a", "b", "c"),)
    assert cert.residuals == (2.0,)


def test_map_ledgers_are_collected_by_the_schema():
    schema = build_three_region_schema()
    paths = [p for p, _ in schema.all_ledgers()]
    assert any(p.startswith("scale_map:") for p in paths)
    assert all(isinstance(l, UncertaintyLedger) for _, l in schema.all_ledgers())
