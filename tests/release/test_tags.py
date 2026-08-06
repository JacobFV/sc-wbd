"""Tag grammar: what it accepts, and — mostly — what it refuses.

Every refusal in :mod:`scwbd.release.tags` gets a test that constructs the bad
input on purpose. A guard nobody has watched fire is indistinguishable from one
that cannot fire.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from scwbd.release.tags import (
    ALIAS_VARIANT,
    BASE,
    VARIANT_ORDER,
    CheckpointTag,
    TagFormatError,
    format_timestamp,
    parse_timestamp,
    sort_tags,
)

UTC = timezone.utc


# ======================================================================
# timestamps
# ======================================================================
def test_the_owners_example_timestamp_is_refused():
    """``116423`` has minute 64. It is not a time and must not parse.

    This is the specific string from the project owner's brief. It is tested by
    name because it is the one malformed timestamp we know somebody actually
    wrote down.
    """
    with pytest.raises(TagFormatError, match="not a real UTC instant|not ISO 8601"):
        parse_timestamp("20260806T116423Z")

    # and the same string in a whole tag, which is how it would really arrive
    with pytest.raises(TagFormatError):
        CheckpointTag.parse("scwbd-001-beta-with-simulation-20260806T116423Z")


def test_owner_example_without_the_z_is_also_refused():
    """The owner's example also lacked a ``Z``; a bare local time is ambiguous."""
    with pytest.raises(TagFormatError, match="not ISO 8601"):
        parse_timestamp("20260806T116423")
    with pytest.raises(TagFormatError, match="not ISO 8601"):
        parse_timestamp("20260806T114623")  # legal time, still no zone


@pytest.mark.parametrize(
    "bad",
    [
        "20260806T246000Z",   # hour 24
        "20260806T115960Z",   # second 60
        "20261306T114623Z",   # month 13
        "20260832T114623Z",   # day 32
        "20260229T114623Z",   # 2026 is not a leap year
    ],
)
def test_out_of_range_fields_are_refused_not_wrapped(bad):
    """An out-of-range field must raise, never roll over into a valid instant."""
    with pytest.raises(TagFormatError):
        parse_timestamp(bad)


@pytest.mark.parametrize(
    "bad",
    [
        "2026-08-06T11:46:23Z",  # extended format
        "20260806t114623z",      # lowercase
        "20260806T114623",       # no zone
        "20260806T1146Z",        # minutes resolution
        "20260806T114623.5Z",    # fractional seconds
        "202608061146233Z",      # no T
        "",
        None,
    ],
)
def test_wrong_shape_is_refused(bad):
    with pytest.raises(TagFormatError):
        parse_timestamp(bad)


def test_valid_timestamp_round_trips_in_utc():
    dt = parse_timestamp("20260806T114623Z")
    assert (dt.year, dt.month, dt.day) == (2026, 8, 6)
    assert (dt.hour, dt.minute, dt.second) == (11, 46, 23)
    assert dt.tzinfo is not None
    assert format_timestamp(dt) == "20260806T114623Z"


def test_naive_datetime_is_refused_rather_than_assumed_utc():
    """Assuming UTC silently shifts the artifact's timestamp by the local offset."""
    with pytest.raises(TagFormatError, match="naive"):
        format_timestamp(datetime(2026, 8, 6, 11, 46, 23))
    with pytest.raises(TagFormatError, match="naive"):
        CheckpointTag.mint("raw", datetime(2026, 8, 6, 11, 46, 23))


def test_non_utc_timezone_is_converted_not_rejected():
    plus_two = timezone(timedelta(hours=2))
    tag = CheckpointTag.mint("raw", datetime(2026, 8, 6, 13, 46, 23, tzinfo=plus_two))
    assert tag.timestamp_text == "20260806T114623Z"


# ======================================================================
# variants and the alias
# ======================================================================
def test_each_known_variant_round_trips():
    when = datetime(2026, 8, 6, 11, 46, 23, tzinfo=UTC)
    for v in VARIANT_ORDER:
        text = CheckpointTag.mint(v, when).format()
        assert text == f"{BASE}-{v}-20260806T114623Z"
        assert CheckpointTag.parse(text).variant == v


def test_alias_resolves_to_combined():
    """A bare ``scwbd-001-beta-<ts>`` means ``combined`` and says so."""
    tag = CheckpointTag.parse("scwbd-001-beta-20260806T114623Z")
    assert tag.variant == ALIAS_VARIANT == "combined"
    assert tag.is_alias is True
    assert tag.format() == "scwbd-001-beta-20260806T114623Z"
    # and the alias compares equal on variant with the thing it points at
    explicit = CheckpointTag.parse("scwbd-001-beta-combined-20260806T114623Z")
    assert tag.variant == explicit.variant
    assert explicit.is_alias is False


def test_multi_hyphen_variant_is_not_torn_apart():
    """``with-simulation-and-synthetic`` contains hyphens; naive splitting breaks it."""
    tag = CheckpointTag.parse(
        "scwbd-001-beta-with-simulation-and-synthetic-20260806T114623Z"
    )
    assert tag.variant == "with-simulation-and-synthetic"


def test_unknown_variant_is_refused():
    """An unrecognised variant cannot be checked against a manifest, so it is an error."""
    with pytest.raises(TagFormatError, match="not in the known variant set"):
        CheckpointTag.parse("scwbd-001-beta-with-vibes-20260806T114623Z")
    with pytest.raises(TagFormatError, match="unknown variant"):
        CheckpointTag(variant="with-vibes", timestamp=parse_timestamp("20260806T114623Z"))


def test_foreign_release_line_is_refused():
    with pytest.raises(TagFormatError, match="does not belong to"):
        CheckpointTag.parse("scwbd-002-alpha-raw-20260806T114623Z")


def test_only_combined_may_be_written_in_alias_form():
    """Writing ``-raw`` as a bare alias would point the documented name elsewhere."""
    when = datetime(2026, 8, 6, 11, 46, 23, tzinfo=UTC)
    with pytest.raises(TagFormatError, match="alias form"):
        CheckpointTag.mint("raw", when).format(as_alias=True)
    # combined may
    assert CheckpointTag.mint("combined", when).format(as_alias=True) == (
        "scwbd-001-beta-20260806T114623Z"
    )


# ======================================================================
# ordering
# ======================================================================
def test_tags_sort_chronologically():
    tags = [
        "scwbd-001-beta-combined-20260806T114623Z",
        "scwbd-001-beta-raw-20260101T000000Z",
        "scwbd-001-beta-with-simulation-20260803T235959Z",
    ]
    got = [t.format() for t in sort_tags(tags)]
    assert got == [
        "scwbd-001-beta-raw-20260101T000000Z",
        "scwbd-001-beta-with-simulation-20260803T235959Z",
        "scwbd-001-beta-combined-20260806T114623Z",
    ]


def test_same_timestamp_ties_break_by_variant_breadth():
    """Arms of one release event sort narrowest-first, not alphabetically."""
    ts = "20260806T114623Z"
    tags = sort_tags(
        [
            f"scwbd-001-beta-combined-{ts}",
            f"scwbd-001-beta-raw-{ts}",
            f"scwbd-001-beta-with-simulation-and-synthetic-{ts}",
            f"scwbd-001-beta-with-simulation-{ts}",
        ]
    )
    assert [t.variant for t in tags] == list(VARIANT_ORDER)


def test_sorting_an_unparseable_tag_raises_rather_than_ordering_it_arbitrarily():
    with pytest.raises(TagFormatError):
        sort_tags(
            [
                "scwbd-001-beta-raw-20260101T000000Z",
                "scwbd-001-beta-raw-20260806T116423Z",  # minute 64
            ]
        )
