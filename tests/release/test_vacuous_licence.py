"""A licence field that states no terms must read ``unknown``, never ``False``.

Written 2026-08-06 (🍃 Mendel) for the defect in `reports/licence_audit.md` §1:
``is_noncommercial_text`` returned ``None`` only for *absent* or literally
``"unknown"`` text, so a **present but vacuous** field — ``"As distributed via
neuromaps"`` — fell through to ``False`` and read downstream as permissive.

Six of the twelve sources on the anatomy default path were classified that way.
The module's own docstring already stated the principle: *"an unlicensed dataset
is not thereby commercially usable, and saying False here would be the
laundering step."* It implemented it for one of the two cases.

This file is the mechanism, not a reminder (decorative_guards rec. 7).
"""

from __future__ import annotations

import pytest

from scwbd.anatomy.sources import SRC
from scwbd.release.licence import (
    is_noncommercial_text,
    is_share_alike_text,
    is_vacuous_licence_text,
    term_from_licence_text,
)

#: Every vacuous string that was live in the registry when this was written.
#: Each one previously classified as "no non-commercial restriction".
VACUOUS_IN_TREE = [
    "As distributed via neuromaps",
    "As released with the cited papers",
    "As released with the cited papers; redistributed by netneurolab",
    "See repository (open, academic use, citation required)",
    "See repository LICENSE (open, academic use)",
    "HCP open-access terms",
    "HCP open-access data-use terms",
    "FreeSurfer license",
    "FreeSurfer license (free for research use)",
    "EBRAINS terms; account required for programmatic access",
    "BSD-3-Clause (toolbox); per-annotation source terms",
    "BSD-3-Clause code; HCP open-access data-use terms for the underlying scans",
    "MIT (CBIG); underlying GSP data under its own terms",
    "BSD-3-Clause (code); data as released with the cited papers",
]

#: Strings that genuinely state terms. These must keep resolving, or the fix has
#: traded one laundering direction for a wall of false unknowns.
DETERMINED = {
    "CC-BY-NC-SA-4.0": (True, True),
    "CC-BY-4.0 (BigBrain derived data)": (False, False),
    "BSD-3-Clause": (False, False),
    "FSL license (free for non-commercial research)": (True, False),
    "CC0 1.0 Universal (public domain dedication)": (False, False),
    "Open Data Commons Attribution License v1.0 (ODC-By 1.0)": (False, False),
    "Open Data Commons Public Domain Dedication and License (PDDL)": (False, False),
    "Creative Commons Attribution-NonCommercial 3.0 Unported (CC BY-NC 3.0)": (True, False),
}


@pytest.mark.parametrize("text", VACUOUS_IN_TREE)
def test_a_field_that_states_no_terms_is_unknown_not_permissive(text):
    """Watched fail by deleting ``is_vacuous_licence_text`` from ``_undetermined``.

    Every one of these then returns ``False`` — "no restriction established" —
    which is the laundering step this module exists to refuse.
    """
    assert is_vacuous_licence_text(text) is True, text
    assert is_noncommercial_text(text) is None, text
    assert is_share_alike_text(text) is None, text


@pytest.mark.parametrize("text,expected", sorted(DETERMINED.items()))
def test_real_licences_still_resolve(text, expected):
    """The fix must not turn every field into an unknown.

    Watched fail by making ``is_vacuous_licence_text`` return ``True``
    unconditionally: all eight then read ``None``.
    """
    assert is_vacuous_licence_text(text) is False, text
    assert (is_noncommercial_text(text), is_share_alike_text(text)) == expected


def test_a_negated_constraint_is_not_read_as_the_constraint():
    """``\\bnon-commercial\\b`` matches "no non-commercial term" too.

    Found by writing exactly that phrase into ``SRC["tian2020"]["license"]`` as
    a helpful clarification and watching a test declare the atlas
    non-commercial. The permissive atlas would have been rejected by the very
    substitution that chose it.

    Watched fail by removing the ``_NEGATION`` check from ``_matches``.
    """
    assert is_noncommercial_text("Attribution required; no non-commercial term") is False
    assert is_share_alike_text("Attribution required; no share-alike term") is False
    # and the un-negated forms must still fire
    assert is_noncommercial_text("Non-commercial use only") is True
    assert is_share_alike_text("Share-alike required") is True


def test_the_term_built_from_a_vacuous_field_is_not_verified():
    """A term whose licence establishes nothing must not look established."""
    t = term_from_licence_text(
        "x", "As distributed via neuromaps", provenance="p", verified=True
    )
    assert t.noncommercial is None and t.share_alike is None


def test_every_registry_source_is_classified_and_none_is_silently_false():
    """Regression over the live registry.

    A source is either determined (its field names terms) or unknown. What must
    never happen again is a *vacuous* field reporting ``False``.
    """
    for key, meta in SRC.items():
        text = meta["license"]
        if is_vacuous_licence_text(text):
            assert is_noncommercial_text(text) is None, key
            assert is_share_alike_text(text) is None, key
        else:
            assert is_noncommercial_text(text) is not None, key


def test_the_sources_corrected_against_vendored_licence_text():
    """Two registry fields were wrong against the licence files in the tree.

    Both were found while choosing a subcortical atlas, by reading
    ``assets/src/*/LICENSE`` instead of trusting the one-line field.

    * ``tian2020`` said "open, academic use"; the licence grants use **without
      restriction** subject to citation — the field understated the grant and
      implied a limit the licence does not contain.
    * ``diedrichsen2009`` said "open, academic use, citation required"; the
      licence is **CC BY-NC 3.0**. A real non-commercial source was recorded as
      permissive, and the classifier agreed.
    """
    assert is_noncommercial_text(SRC["tian2020"]["license"]) is False
    assert is_share_alike_text(SRC["tian2020"]["license"]) is False
    assert SRC["tian2020"]["license_text"].endswith("license.txt")

    assert is_noncommercial_text(SRC["diedrichsen2009"]["license"]) is True
    assert "NON-COMMERCIAL" in SRC["diedrichsen2009"]["license"]
    assert SRC["diedrichsen2009"]["license_text"].endswith("LICENSE")
