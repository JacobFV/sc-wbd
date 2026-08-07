"""Checkpoint tags: parse, validate, format, order.

A tag is a **claim about provenance** written into a filename.  This module
makes the claim well-formed; :mod:`scwbd.release.manifest` makes it *true* by
checking it against the run's own source cards.  Neither job is done by the
other, and a tag that parses is not thereby verified.

Grammar (strict)::

    <base>[-<variant>]-<YYYYMMDD>T<HHMMSS>Z

``base`` is ``scwbd-001-beta``.  ``variant`` is drawn from a closed set
(:data:`VARIANTS`); an unknown variant is an error, not a new family.  Omitting
the variant yields the **release alias**, which resolves to
``with-simulation-and-synthetic``.  Retired variants (:data:`RETIRED_VARIANTS`)
are refused by name with the reason they were withdrawn.

Timestamps are **ISO 8601 basic, UTC, seconds resolution**.  The project
owner's worked example was ``-20260806T116423``, which is not a time: minute
``64`` does not exist and the trailing ``Z`` is missing, so the string does not
name an instant at all.  It is rejected here rather than normalised, because a
timestamp that silently became ``12:04:23`` would misdate an artifact for the
rest of its life.  See ``reports/checkpoint_family.md`` §"Timestamp format".

Nothing in this module warns.  A tag either parses or raises
:class:`TagFormatError`; the taxonomy has no "probably fine" state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Mapping

__all__ = [
    "TagFormatError",
    "BASE",
    "VARIANTS",
    "VARIANT_ORDER",
    "ALIAS_VARIANT",
    "TIMESTAMP_FORMAT",
    "RETIRED_VARIANTS",
    "CheckpointTag",
    "format_timestamp",
    "parse_timestamp",
    "sort_tags",
]


class TagFormatError(ValueError):
    """A checkpoint tag is malformed, or names a variant that does not exist.

    Deliberately an error and never a warning: a tag is the only thing many
    downstream consumers will ever read about an artifact's provenance, so a
    tag nobody could parse must stop the pipeline rather than flow through it
    as a string.
    """


#: The release line this taxonomy covers.
BASE = "scwbd-001-beta"

#: Canonical variant names, **ordered from fewest source families to most**.
#: The order is load-bearing twice over: it breaks ties when two tags carry the
#: same timestamp, and it decides which variant keeps the tag when two variants
#: turn out to be the same bytes (:mod:`scwbd.release.collapse` keeps the
#: *minimal* claim).
VARIANT_ORDER: tuple[str, ...] = (
    "raw",
    "with-simulation",
    "with-simulation-and-synthetic",
)

VARIANTS: frozenset[str] = frozenset(VARIANT_ORDER)

#: Variants that once existed and have been withdrawn.  ``combined`` claimed
#: exactly the family set ``with-simulation-and-synthetic`` claims, so it could
#: never denote a different artifact; the owner retired it on 2026-08-06.
#:
#: Retired names are listed rather than forgotten so the parser can refuse them
#: *by name and with a reason*. A retired tag that still parses is a name for
#: nothing, and one that fails as a generic "unknown variant" tells a reader
#: holding an old checkpoint nothing about what happened to it.
RETIRED_VARIANTS: Mapping[str, str] = {
    "combined": (
        "retired 2026-08-06: it claimed the same source families as "
        "'with-simulation-and-synthetic' and could never denote a different "
        "artifact. The bare alias 'scwbd-001-beta-<ts>' now resolves to "
        "'with-simulation-and-synthetic'."
    ),
}

#: The variant that a bare ``scwbd-001-beta-<ts>`` tag resolves to.  The alias
#: exists because the owner wants one obvious name to hand out; it is a
#: *pointer*, and :attr:`CheckpointTag.is_alias` keeps that visible so a reader
#: can tell a pointer from an arm of the ablation.
ALIAS_VARIANT = "with-simulation-and-synthetic"

#: ``strptime``/``strftime`` pattern for ISO 8601 basic UTC at seconds
#: resolution.  ``%M`` accepts ``00``-``59`` only, which is what rejects the
#: owner's ``116423``.
TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"

#: Shape gate applied *before* ``strptime``.  ``strptime`` alone is too
#: forgiving about field widths (it will read ``2026081T1146Z``-style strings
#: in some builds), and a tag whose timestamp is the wrong *length* is
#: malformed even when the numbers happen to be legal.
_TS_SHAPE = re.compile(r"^\d{8}T\d{6}Z$")



def parse_timestamp(text: str) -> datetime:
    """Parse ``YYYYMMDDTHHMMSSZ`` into a timezone-aware UTC ``datetime``.

    Raises :class:`TagFormatError` for anything else, including the near-misses
    that matter in practice: a missing ``Z`` (which would make the instant
    ambiguous), lowercase separators, extended-format hyphens and colons, and
    out-of-range fields such as minute ``64`` or hour ``24``.
    """
    if not isinstance(text, str) or not _TS_SHAPE.match(text):
        raise TagFormatError(
            f"timestamp {text!r} is not ISO 8601 basic UTC. Expected exactly "
            f"YYYYMMDDTHHMMSSZ (e.g. 20260806T114623Z): 8 digits, 'T', 6 digits, 'Z'. "
            "A timestamp without a 'Z' does not name an instant, and this taxonomy "
            "does not guess a timezone."
        )
    try:
        dt = datetime.strptime(text, TIMESTAMP_FORMAT)
    except ValueError as exc:
        raise TagFormatError(
            f"timestamp {text!r} is well-shaped but not a real UTC instant ({exc}). "
            "Out-of-range fields are refused, never wrapped: a minute of '64' is not "
            "'one hour and four minutes', it is a typo, and normalising it would "
            "misdate the artifact permanently."
        ) from exc
    return dt.replace(tzinfo=timezone.utc)


def format_timestamp(when: datetime) -> str:
    """Render a ``datetime`` as ``YYYYMMDDTHHMMSSZ``, converting to UTC.

    A naive ``datetime`` is refused rather than assumed to be UTC: assuming is
    how an artifact acquires a timestamp that is wrong by the author's offset
    and looks perfectly well-formed forever after.
    """
    if when.tzinfo is None:
        raise TagFormatError(
            "refusing to format a naive datetime: it carries no timezone, and "
            "assuming UTC would silently shift the artifact's timestamp by the "
            "author's local offset. Pass an aware datetime "
            "(datetime.now(timezone.utc))."
        )
    return when.astimezone(timezone.utc).strftime(TIMESTAMP_FORMAT)


@dataclass(frozen=True)
class CheckpointTag:
    """One checkpoint name, parsed into its parts.

    ``variant`` is always the *resolved* variant, so the alias and the thing it
    points at compare equal on this field.  ``is_alias`` records that the
    written form omitted the variant, which is the only way a reader can tell
    ``scwbd-001-beta-<ts>`` from an explicitly-written variant after parsing.
    """

    variant: str
    timestamp: datetime
    base: str = BASE
    is_alias: bool = False

    def __post_init__(self) -> None:
        if self.variant in RETIRED_VARIANTS:
            raise TagFormatError(
                f"variant {self.variant!r} has been retired. "
                f"{RETIRED_VARIANTS[self.variant]}"
            )
        if self.variant not in VARIANTS:
            raise TagFormatError(
                f"unknown variant {self.variant!r}; known variants are "
                f"{sorted(VARIANTS)}. The variant set is closed: a new source family "
                "needs a taxonomy change and a report entry, not a new filename."
            )
        if self.timestamp.tzinfo is None:
            raise TagFormatError("CheckpointTag.timestamp must be timezone-aware UTC")

    # -- parsing ----------------------------------------------------------
    @classmethod
    def parse(cls, text: str) -> "CheckpointTag":
        """Parse a tag string.  Raises :class:`TagFormatError` on anything odd."""
        if not isinstance(text, str) or not text:
            raise TagFormatError(f"tag must be a non-empty string, got {text!r}")
        # The timestamp is the trailing ``-<8digits>T<6digits>Z`` group.  Split
        # on it first: variants contain hyphens, so splitting left-to-right
        # would tear ``with-simulation-and-synthetic`` apart.
        m = re.match(r"^(?P<stem>.+?)-(?P<ts>\d{8}T\d{6}Z)$", text)
        if not m:
            # Re-raise through parse_timestamp when there is *something*
            # timestamp-shaped, so the caller gets the specific reason.
            tail = text.rsplit("-", 1)[-1]
            parse_timestamp(tail)  # raises with the precise complaint
            raise TagFormatError(  # pragma: no cover - defensive
                f"tag {text!r} has no trailing -YYYYMMDDTHHMMSSZ timestamp"
            )
        stem = m.group("stem")
        timestamp = parse_timestamp(m.group("ts"))

        if stem == BASE:
            return cls(variant=ALIAS_VARIANT, timestamp=timestamp, base=BASE, is_alias=True)
        if not stem.startswith(BASE + "-"):
            raise TagFormatError(
                f"tag {text!r} does not belong to the {BASE!r} release line "
                f"(stem was {stem!r})."
            )
        variant = stem[len(BASE) + 1 :]
        if variant in RETIRED_VARIANTS:
            raise TagFormatError(
                f"tag {text!r} names retired variant {variant!r}. "
                f"{RETIRED_VARIANTS[variant]}"
            )
        if variant not in VARIANTS:
            raise TagFormatError(
                f"tag {text!r} names variant {variant!r}, which is not in the known "
                f"variant set {sorted(VARIANTS)}. Unknown variants are refused rather "
                "than accepted as new families: the tag is a provenance claim, and an "
                "unrecognised claim cannot be checked against a manifest."
            )
        return cls(variant=variant, timestamp=timestamp, base=BASE, is_alias=False)

    # -- construction -----------------------------------------------------
    @classmethod
    def mint(
        cls, variant: str, when: datetime, *, base: str = BASE, alias: bool = False
    ) -> "CheckpointTag":
        """Build a tag for ``variant`` at ``when`` (aware datetime, any zone)."""
        if when.tzinfo is None:
            raise TagFormatError(
                "refusing to mint a tag from a naive datetime; pass "
                "datetime.now(timezone.utc)"
            )
        return cls(
            variant=variant,
            timestamp=when.astimezone(timezone.utc).replace(microsecond=0),
            base=base,
            is_alias=alias,
        )

    # -- rendering --------------------------------------------------------
    @property
    def timestamp_text(self) -> str:
        return format_timestamp(self.timestamp)

    def format(self, *, as_alias: bool | None = None) -> str:
        """Render the tag.  ``as_alias`` overrides the parsed/minted form."""
        alias = self.is_alias if as_alias is None else as_alias
        if alias:
            if self.variant != ALIAS_VARIANT:
                raise TagFormatError(
                    f"only the {ALIAS_VARIANT!r} variant may be written in alias form; "
                    f"{self.variant!r} may not, because the bare name "
                    f"{BASE!r} is documented to mean {ALIAS_VARIANT!r} and would "
                    "otherwise point at a different set of source families."
                )
            return f"{self.base}-{self.timestamp_text}"
        return f"{self.base}-{self.variant}-{self.timestamp_text}"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.format()

    # -- ordering ---------------------------------------------------------
    @property
    def sort_key(self) -> tuple[datetime, int, str]:
        return (self.timestamp, VARIANT_ORDER.index(self.variant), self.base)

    def __lt__(self, other: "CheckpointTag") -> bool:
        if not isinstance(other, CheckpointTag):  # pragma: no cover - defensive
            return NotImplemented
        return self.sort_key < other.sort_key


def sort_tags(tags: Iterable[CheckpointTag | str]) -> list[CheckpointTag]:
    """Chronological sort, oldest first; ties broken by variant breadth.

    Accepts strings or parsed tags.  Strings are parsed, so an unparseable tag
    raises here rather than sorting to an arbitrary position.
    """
    parsed = [t if isinstance(t, CheckpointTag) else CheckpointTag.parse(t) for t in tags]
    return sorted(parsed, key=lambda t: t.sort_key)
