"""Byte-identical variants collapse to an alias instead of minting a tag.

Three names for one checkpoint is a distinction that does not exist. The
decision is made by weight hash, so it cannot be talked out of.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from scwbd.release.collapse import CollapseError, collapse_identical
from scwbd.release.families import STRUCTURALLY_IDENTICAL_VARIANTS

UTC = timezone.utc
WHEN = datetime(2026, 8, 6, 11, 46, 23, tzinfo=UTC)

H_A = "a" * 64
H_B = "b" * 64
H_C = "c" * 64


def test_two_identical_checkpoints_collapse_to_one_tag_and_an_alias():
    """The explicit two-identical-artifact case required by the brief."""
    res = collapse_identical(
        {"with-simulation": H_A, "with-simulation-and-synthetic": H_A}, when=WHEN
    )
    assert len(res.minted) == 1
    assert res.minted[0].variant == "with-simulation", "the narrower claim keeps the tag"
    assert len(res.aliases) == 1

    alias = res.aliases[0]
    assert alias.variant == "with-simulation-and-synthetic"
    assert alias.canonical_variant == "with-simulation"
    assert alias.canonical == "scwbd-001-beta-with-simulation-20260806T114623Z"
    assert alias.weights_sha256 == H_A
    assert "byte-identical" in alias.reason
    assert res.collapsed is True


def test_the_tribe_unusable_scenario_collapses_three_names_to_one():
    """If TRIBE is unusable, +synthetic and combined are the +simulation bytes."""
    res = collapse_identical(
        {
            "with-simulation": H_A,
            "with-simulation-and-synthetic": H_A,
            "combined": H_A,
        },
        when=WHEN,
    )
    assert res.as_dict()["n_distinct_artifacts"] == 1
    assert res.as_dict()["n_names_requested"] == 3
    assert [a.variant for a in res.aliases] == [
        "with-simulation-and-synthetic",
        "combined",
    ]
    assert all(a.canonical_variant == "with-simulation" for a in res.aliases)


def test_genuinely_different_artifacts_are_not_collapsed():
    """The negative control: distinct bytes must still get distinct tags."""
    res = collapse_identical(
        {"raw": H_A, "with-simulation": H_B, "combined": H_C}, when=WHEN
    )
    assert len(res.minted) == 3
    assert res.aliases == ()
    assert res.collapsed is False
    assert [t.variant for t in res.minted] == ["raw", "with-simulation", "combined"]


def test_partial_collapse_keeps_the_distinct_arm():
    res = collapse_identical(
        {"raw": H_A, "with-simulation": H_B, "combined": H_B}, when=WHEN
    )
    assert {t.variant for t in res.minted} == {"raw", "with-simulation"}
    assert [a.variant for a in res.aliases] == ["combined"]


def test_narrowest_variant_always_keeps_the_tag_regardless_of_input_order():
    """Insertion order must not decide which provenance claim survives."""
    for order in (
        {"combined": H_A, "raw": H_A},
        {"raw": H_A, "combined": H_A},
    ):
        res = collapse_identical(order, when=WHEN)
        assert res.minted[0].variant == "raw"
        assert res.aliases[0].variant == "combined"


# ======================================================================
# refusals
# ======================================================================
def test_missing_weight_hash_is_refused():
    """Collapse is decided by bytes; a placeholder hash would corrupt it."""
    with pytest.raises(CollapseError, match="not a sha256"):
        collapse_identical({"raw": ""}, when=WHEN)
    with pytest.raises(CollapseError, match="not a sha256"):
        collapse_identical({"raw": "deadbeef"}, when=WHEN)
    with pytest.raises(CollapseError, match="not a sha256"):
        collapse_identical({"raw": None}, when=WHEN)


def test_unknown_variant_is_refused():
    with pytest.raises(CollapseError, match="unknown variant"):
        collapse_identical({"with-vibes": H_A}, when=WHEN)


def test_empty_candidate_set_is_refused():
    with pytest.raises(CollapseError, match="nothing to mint"):
        collapse_identical({}, when=WHEN)


def test_all_minted_tags_share_one_timestamp():
    """Arms of one release event must sort together as one event."""
    res = collapse_identical({"raw": H_A, "with-simulation": H_B}, when=WHEN)
    assert len({t.timestamp_text for t in res.minted}) == 1


# ======================================================================
# the taxonomy already contains a redundant pair
# ======================================================================
def test_combined_and_with_simulation_and_synthetic_are_structurally_identical():
    """Under the owner's own definition these two variants name the same set.

    They can never describe different training mixtures, so any run that
    produces both will always collapse. Recorded as a taxonomy fact so it is
    not rediscovered as a surprise at release time.
    """
    assert STRUCTURALLY_IDENTICAL_VARIANTS == (
        ("with-simulation-and-synthetic", "combined"),
    )
