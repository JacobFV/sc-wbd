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
    """If TRIBE is unusable, the +synthetic arm is the +simulation bytes."""
    res = collapse_identical(
        {"with-simulation": H_A, "with-simulation-and-synthetic": H_A},
        when=WHEN,
    )
    assert res.as_dict()["n_distinct_artifacts"] == 1
    assert res.as_dict()["n_names_requested"] == 2
    assert [a.variant for a in res.aliases] == ["with-simulation-and-synthetic"]
    assert all(a.canonical_variant == "with-simulation" for a in res.aliases)


def test_genuinely_different_artifacts_are_not_collapsed():
    """The negative control: distinct bytes must still get distinct tags."""
    res = collapse_identical(
        {"raw": H_A, "with-simulation": H_B, "with-simulation-and-synthetic": H_C},
        when=WHEN,
    )
    assert len(res.minted) == 3
    assert res.aliases == ()
    assert res.collapsed is False
    assert [t.variant for t in res.minted] == [
        "raw", "with-simulation", "with-simulation-and-synthetic",
    ]


def test_partial_collapse_keeps_the_distinct_arm():
    res = collapse_identical(
        {"raw": H_A, "with-simulation": H_B, "with-simulation-and-synthetic": H_B},
        when=WHEN,
    )
    assert {t.variant for t in res.minted} == {"raw", "with-simulation"}
    assert [a.variant for a in res.aliases] == ["with-simulation-and-synthetic"]


def test_narrowest_variant_always_keeps_the_tag_regardless_of_input_order():
    """Insertion order must not decide which provenance claim survives."""
    for order in (
        {"with-simulation-and-synthetic": H_A, "raw": H_A},
        {"raw": H_A, "with-simulation-and-synthetic": H_A},
    ):
        res = collapse_identical(order, when=WHEN)
        assert res.minted[0].variant == "raw"
        assert res.aliases[0].variant == "with-simulation-and-synthetic"


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
def test_no_two_live_variants_claim_the_same_family_set():
    """The redundancy check that found ``-combined``, kept pointed at the future.

    ``-combined`` claimed exactly what ``-with-simulation-and-synthetic``
    claims and was retired on 2026-08-06. The check is retained so the next
    redundant pair is caught the same way rather than shipping as two names
    for one artifact.
    """
    assert STRUCTURALLY_IDENTICAL_VARIANTS == ()


def test_retired_variant_cannot_be_minted():
    """Collapse must not resurrect a withdrawn name."""
    with pytest.raises(CollapseError, match="unknown variant"):
        collapse_identical({"combined": H_A}, when=WHEN)
