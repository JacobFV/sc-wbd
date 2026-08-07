"""Identical-artifact collapse: refuse to mint distinct tags for one checkpoint.

If 🎓 Ramón's discrepancy assessment finds TRIBE v2 unusable, then
``-with-simulation-and-synthetic`` and ``-combined`` are trained from exactly
the sources that trained ``-with-simulation``, and all three are the same
bytes. Three names for one artifact is a distinction that does not exist: it
inflates the apparent size of the release family and invites a reader to
believe an ablation arm was run when it was not.

So the decision is made **by weight hash**, not by intent. Variants whose
weights are byte-identical collapse to a single minted tag plus recorded
aliases, each carrying the reason it is an alias.

Which variant keeps the tag: the **narrowest** one (earliest in
:data:`~scwbd.release.tags.VARIANT_ORDER`). Naming a checkpoint
``-with-simulation-and-synthetic`` when it is byte-identical to the
no-synthetic artifact would assert that a synthetic corpus contributed, which
is precisely the overclaim :meth:`SourceFamilyManifest.validate_tag` exists to
catch. The minimal claim is the only honest one.

This is a *measurement*, not a policy: if the arms genuinely differ, they get
distinct tags and no alias is emitted. The collapse path only fires when the
bytes say it should.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping

from .tags import VARIANT_ORDER, CheckpointTag

__all__ = ["CollapseError", "TagAlias", "CollapseResult", "collapse_identical"]


class CollapseError(ValueError):
    """A collapse could not be decided — e.g. a candidate with no weight hash."""


@dataclass(frozen=True)
class TagAlias:
    """A name that resolves to another checkpoint instead of minting its own."""

    alias: str
    #: The tag actually minted, whose bytes this name points at.
    canonical: str
    variant: str
    canonical_variant: str
    weights_sha256: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "alias": self.alias,
            "canonical": self.canonical,
            "variant": self.variant,
            "canonical_variant": self.canonical_variant,
            "weights_sha256": self.weights_sha256,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CollapseResult:
    """Which tags were minted, which became aliases, and why."""

    minted: tuple[CheckpointTag, ...] = ()
    aliases: tuple[TagAlias, ...] = ()

    @property
    def collapsed(self) -> bool:
        return bool(self.aliases)

    def as_dict(self) -> dict[str, Any]:
        return {
            "minted": [t.format() for t in self.minted],
            "aliases": [a.as_dict() for a in self.aliases],
            "collapsed": self.collapsed,
            "n_distinct_artifacts": len(self.minted),
            "n_names_requested": len(self.minted) + len(self.aliases),
        }


def collapse_identical(
    candidates: Mapping[str, str],
    *,
    when,
    base: str | None = None,
) -> CollapseResult:
    """Mint tags for ``{variant: weights_sha256}``, collapsing duplicates.

    Parameters
    ----------
    candidates
        Variant name -> sha256 of that arm's weight file. Every variant that
        was actually trained should appear; a variant that was not trained
        must be **absent**, not present with a placeholder hash.
    when
        Timezone-aware ``datetime`` for the whole family. One timestamp is used
        for every minted tag so that the arms of an ablation sort together and
        are visibly one release event rather than several.

    Returns
    -------
    CollapseResult
        ``minted`` holds one tag per distinct weight hash; ``aliases`` holds a
        recorded pointer for every other requested name.
    """
    if not candidates:
        raise CollapseError("no candidate variants supplied; nothing to mint")

    unknown = sorted(set(candidates) - set(VARIANT_ORDER))
    if unknown:
        raise CollapseError(
            f"unknown variant(s) {unknown}; the variant set is closed "
            f"({list(VARIANT_ORDER)}). A new arm needs a taxonomy change, not a "
            "new key in this mapping."
        )
    for variant, digest in candidates.items():
        if not digest or not isinstance(digest, str) or len(digest) != 64:
            raise CollapseError(
                f"variant {variant!r} has weight hash {digest!r}, which is not a "
                "sha256. Collapse is decided by bytes; a missing or placeholder "
                "hash would let two different artifacts share a name, or one "
                "artifact acquire two. Hash the weights first."
            )

    by_hash: dict[str, list[str]] = defaultdict(list)
    for variant, digest in candidates.items():
        by_hash[digest].append(variant)

    order = {v: i for i, v in enumerate(VARIANT_ORDER)}
    minted: list[CheckpointTag] = []
    aliases: list[TagAlias] = []

    for digest, variants in by_hash.items():
        variants.sort(key=lambda v: order[v])
        keeper, *rest = variants
        kwargs = {"base": base} if base else {}
        canonical = CheckpointTag.mint(keeper, when, **kwargs)
        minted.append(canonical)
        for other in rest:
            other_tag = CheckpointTag.mint(other, when, **kwargs)
            aliases.append(
                TagAlias(
                    alias=other_tag.format(),
                    canonical=canonical.format(),
                    variant=other,
                    canonical_variant=keeper,
                    weights_sha256=digest,
                    reason=(
                        f"byte-identical to the {keeper!r} arm (sha256 {digest[:16]}…). "
                        f"Minting {other!r} as a distinct tag would assert that the "
                        "extra source families changed the artifact, and they did not. "
                        "Recorded as an alias so the name still resolves, and so the "
                        "absence of a distinct arm is visible rather than inferred."
                    ),
                )
            )

    minted.sort(key=lambda t: t.sort_key)
    aliases.sort(key=lambda a: order[a.variant])
    return CollapseResult(minted=tuple(minted), aliases=tuple(aliases))
