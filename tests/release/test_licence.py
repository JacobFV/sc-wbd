"""Licence union: most restrictive wins, unknown is never permissive.

The distinction under test throughout is **inheritance vs policy**. A licence a
source forces cannot be removed by anyone here; a licence an owner chose can be
removed by that owner. Collapsing them into one boolean destroys the only
information a future reader needs.
"""

from __future__ import annotations

import pytest

from scwbd.release.licence import (
    LicenceTerm,
    UNKNOWN_TERM,
    is_noncommercial_text,
    is_share_alike_text,
    term_from_dataset_card,
    union_of,
)

ODC_BY = LicenceTerm(
    source_id="eegmmidb", name="Open Data Commons Attribution License v1.0 (ODC-By 1.0)",
    noncommercial=False, share_alike=False, attribution=True,
    redistribution="full", provenance="scwbd/sources/cards/eegmmidb.yaml", verified=True,
    obligations=("attribution_required",),
)
CC0 = LicenceTerm(
    source_id="ds000117", name="CC0 1.0 Universal", noncommercial=False,
    share_alike=False, attribution=False, redistribution="full",
    provenance="scwbd/sources/cards/ds000117.yaml", verified=True,
)
HANSEN = LicenceTerm(
    source_id="anatomical_prior", name="CC-BY-NC-SA-4.0", noncommercial=True,
    share_alike=True, attribution=True, redistribution="unknown",
    provenance="assets/MANIFEST.json", verified=True,
)
TRIBE = LicenceTerm(
    source_id="tribe_v2_teacher", name="CC BY-NC 4.0 (asserted)", noncommercial=True,
    share_alike=False, attribution=True, redistribution="unknown",
    provenance="declared:brief", verified=False,
)
MNE_UNKNOWN = LicenceTerm(
    source_id="mne-sample", name="unknown - the archive ships no LICENSE file",
    noncommercial=None, share_alike=None, attribution=None,
    redistribution="unknown", provenance="scwbd/sources/cards/mne-sample.yaml",
    verified=False,
)


# ======================================================================
# text classification
# ======================================================================
@pytest.mark.parametrize(
    "text,expected",
    [
        ("CC-BY-NC-SA-4.0", True),
        ("CC BY-NC 4.0", True),
        ("FSL license (free for non-commercial research)", True),
        ("Open Data Commons Attribution License v1.0 (ODC-By 1.0)", False),
        ("CC0 1.0 Universal (public domain dedication)", False),
        ("BSD-3-Clause", False),
        ("unknown - the archive ships no LICENSE file", None),
        (None, None),
        ("", None),
    ],
)
def test_noncommercial_detection(text, expected):
    assert is_noncommercial_text(text) is expected


def test_nc_detection_does_not_fire_on_incidental_letters():
    """A bare substring search for 'nc' matches 'Inc.' and 'Encoding'."""
    assert is_noncommercial_text("Licensed by Example Inc.") is False
    assert is_noncommercial_text("BSD-3-Clause; encoding by ffmpeg") is False


def test_share_alike_is_detected_independently_of_nc():
    """CC-BY-NC-SA is both; reporting only NC understates the obligation."""
    assert is_share_alike_text("CC-BY-NC-SA-4.0") is True
    assert is_share_alike_text("CC BY-NC 4.0") is False
    assert is_share_alike_text("unknown") is None


# ======================================================================
# union arithmetic
# ======================================================================
def test_union_takes_the_most_restrictive_term():
    u = union_of([ODC_BY, CC0, HANSEN])
    assert u.noncommercial_by_inheritance is True
    assert u.share_alike_by_inheritance is True
    assert u.inheritance_sources == ("anatomical_prior",)


def test_unknown_never_becomes_permissive():
    """An unlicensed source must not read as 'commercial use permitted'."""
    u = union_of([ODC_BY, CC0, MNE_UNKNOWN])
    assert u.noncommercial_by_inheritance is None, "unknown must not collapse to False"
    assert "mne-sample" in u.unknown_sources
    assert "UNKNOWN" in u.summary()
    assert "not permissive" in u.summary()


def test_a_real_nc_source_beats_an_unknown():
    """True wins over None: a known restriction is not softened by an unknown."""
    u = union_of([MNE_UNKNOWN, HANSEN])
    assert u.noncommercial_by_inheritance is True


def test_not_nc_is_not_rendered_as_unrestricted():
    """Attribution obligations survive into the summary of a permissive union."""
    u = union_of([ODC_BY, CC0])
    assert u.noncommercial_effective is False
    assert "attribution_required" in u.obligations
    s = u.summary()
    assert "permissive" not in s.lower()
    assert "attribution: required" in s


def test_redistribution_takes_the_most_restrictive_class():
    blocked = LicenceTerm(
        source_id="ukbiobank", name="UK Biobank MTA", noncommercial=None,
        redistribution="none", provenance="card", verified=True,
    )
    assert union_of([ODC_BY, blocked]).redistribution == "none"
    assert union_of([ODC_BY, MNE_UNKNOWN]).redistribution == "unknown"
    assert union_of([ODC_BY, CC0]).redistribution == "full"


# ======================================================================
# inheritance vs policy — the decision that must be recorded, not inherited
# ======================================================================
def test_inheritance_and_policy_are_separate_fields():
    """NC forced by a source and NC chosen by an owner are different facts."""
    inherited = union_of([ODC_BY, HANSEN])
    chosen = union_of([ODC_BY, CC0], policy={"noncommercial": "owner directive"})

    assert inherited.noncommercial_by_inheritance is True
    assert inherited.noncommercial_by_policy is False

    assert chosen.noncommercial_by_inheritance is False
    assert chosen.noncommercial_by_policy is True

    # both are non-commercial in effect...
    assert inherited.noncommercial_effective is True
    assert chosen.noncommercial_effective is True
    # ...but only one of them can be revoked by the person who set it
    assert inherited.noncommercial_is_removable is False
    assert chosen.noncommercial_is_removable is True


def test_policy_nc_over_an_already_inherited_nc_is_not_removable():
    """Policy cannot make an inherited restriction disappear."""
    u = union_of([HANSEN], policy={"noncommercial": "owner directive"})
    assert u.noncommercial_effective is True
    assert u.noncommercial_is_removable is False


def test_no_policy_means_licence_is_exactly_what_sources_force():
    """With TRIBE absent and no policy, nothing imposes NC."""
    u = union_of([ODC_BY, CC0])
    assert u.noncommercial_effective is False
    assert u.as_dict()["by_policy"]["noncommercial"] is False
    assert u.as_dict()["by_policy"]["terms"] == {}


def test_unverified_licences_are_flagged_and_stay_flagged():
    """TRIBE's NC is asserted in a brief, not recorded in this repository."""
    u = union_of([ODC_BY, TRIBE])
    assert u.noncommercial_by_inheritance is True
    assert "tribe_v2_teacher" in u.unverified_sources
    d = u.as_dict()
    tribe = [t for t in d["terms"] if t["source_id"] == "tribe_v2_teacher"][0]
    assert tribe["verified"] is False
    assert tribe["provenance"] == "declared:brief"


def test_unknown_term_is_not_permissive_by_construction():
    assert UNKNOWN_TERM.noncommercial is None
    assert UNKNOWN_TERM.redistribution == "unknown"
    assert UNKNOWN_TERM.verified is False


# ======================================================================
# real cards on disk
# ======================================================================
def test_real_dataset_cards_classify_as_expected():
    """Regenerate from source: read the actual cards, do not trust a table."""
    eeg = term_from_dataset_card("scwbd/sources/cards/eegmmidb.yaml")
    assert eeg.noncommercial is False
    assert eeg.attribution is True
    assert eeg.verified is True
    assert "attribution_required" in eeg.obligations

    ds = term_from_dataset_card("scwbd/sources/cards/ds000117.yaml")
    assert ds.noncommercial is False
    assert ds.redistribution == "full"

    mne = term_from_dataset_card("scwbd/sources/cards/mne-sample.yaml")
    assert mne.noncommercial is None, "mne-sample ships no licence; must be unknown"
    assert mne.verified is False


def test_hansen_receptor_atlas_is_noncommercial_in_the_asset_registry():
    """The NC that reaches ``-raw`` and ``-with-simulation`` comes from here."""
    from scwbd.anatomy.sources import SRC

    assert is_noncommercial_text(SRC["hansen_receptors"]["license"]) is True
    assert is_share_alike_text(SRC["hansen_receptors"]["license"]) is True
