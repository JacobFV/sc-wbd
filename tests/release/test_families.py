"""Vocabulary and modality axis, checked against the real repository.

Two kinds of test live here:

* **anti-drift** — this package mirrors constants owned elsewhere
  (``scwbd.foundation.mixture.ROLES``). A copy without a test is a future
  contradiction, so the copy is asserted equal to the original.
* **regenerate, don't audit** — the taxonomy is exercised against the actual
  cards on disk rather than a fixture that agrees with it by construction.
"""

from __future__ import annotations

from scwbd.release.datasets import (
    MEASURED_MODALITIES,
    link_sources_to_datasets,
    load_dataset_cards,
)
from scwbd.release.families import (
    ALL_FAMILIES,
    FAMILY_TIER,
    FAMILY_TO_D12_BUCKET,
    ROLES,
    TIER_GAP_REASON,
    TIER_MEASUREMENT,
    TIER_SIMULATION,
    TIER_TEACHER_PREDICTION,
    UNKNOWN,
    VARIANT_FAMILIES,
    load_source_records,
)
from scwbd.release.manifest import build_manifest
from scwbd.release.tags import VARIANT_ORDER


# ======================================================================
# anti-drift
# ======================================================================
def test_role_vocabulary_matches_foundation():
    """This package's mirrored ``ROLES`` must equal the one the trainer uses.

    ``scwbd.release`` deliberately avoids importing torch, so it keeps a copy.
    This test is what makes the copy a fact instead of a divergence waiting to
    happen; it imports the real definition.
    """
    from scwbd.foundation.mixture import ROLES as FOUNDATION_ROLES

    assert set(ROLES) == set(FOUNDATION_ROLES), (
        "scwbd.release.families.ROLES has drifted from "
        "scwbd.foundation.mixture.ROLES"
    )


def test_every_role_maps_somewhere_or_is_deliberately_unmapped():
    """No role may be silently absent from the D12 bucket table."""
    from scwbd.release.families import ROLE_TO_D12_BUCKET

    unmapped = set(ROLES) - set(ROLE_TO_D12_BUCKET)
    # 'prior' is intentionally unmapped: only is_simulated can resolve it.
    assert unmapped == {"prior"}, f"unexpectedly unmapped roles: {unmapped}"


def test_every_variant_has_a_family_set_and_vice_versa():
    assert set(VARIANT_FAMILIES) == set(VARIANT_ORDER)


def test_every_family_has_a_tier_entry_and_gaps_have_reasons():
    """A ``None`` tier must be accompanied by a reason. Absence writes something."""
    assert set(FAMILY_TIER) == set(ALL_FAMILIES)
    for fam, tier in FAMILY_TIER.items():
        if tier is None:
            assert fam in TIER_GAP_REASON and TIER_GAP_REASON[fam], (
                f"family {fam!r} has no tier and no reason for not having one"
            )


def test_tier_boundary_is_provenance_not_modality():
    """Measurement is tier 1 whatever the instrument; simulation and teacher differ."""
    assert FAMILY_TIER["real"] == TIER_MEASUREMENT == 1
    assert FAMILY_TIER["simulation"] == TIER_SIMULATION == 4
    assert FAMILY_TIER["synthetic"] == TIER_TEACHER_PREDICTION == 5
    # tier 2 was never specified to this module and must not be invented
    assert 2 not in [t for t in FAMILY_TIER.values() if t is not None]


def test_every_family_has_a_d12_bucket():
    assert set(FAMILY_TO_D12_BUCKET) == set(ALL_FAMILIES)


# ======================================================================
# real cards on disk
# ======================================================================
def test_real_mixture_cards_all_classify():
    """Every card in ``configs/source_cards`` gets a family, unknown included."""
    records = load_source_records("configs/source_cards")
    ids = {r.id for r in records}
    assert {"eegmmidb_real", "sim_wholebrain", "tribe_v2_teacher"} <= ids
    for r in records:
        assert r.family in ALL_FAMILIES


def test_tribe_is_disabled_in_the_real_mixture():
    """The synthetic family contributes nothing today; the collapse path is the likely one."""
    records = {r.id: r for r in load_source_records("configs/source_cards")}
    assert records["tribe_v2_teacher"].enabled is False
    assert records["tribe_v2_teacher"].contributes_gradient is False


def test_anatomical_prior_is_unknown_family_in_the_real_mixture():
    """A non-simulated prior has no tag-axis name; it must not be guessed into one."""
    records = {r.id: r for r in load_source_records("configs/source_cards")}
    assert records["anatomical_prior"].family == UNKNOWN


def test_live_config_manifest_matches_the_variant_being_trained():
    """The run under ``configs/scwbd_001_beta.yaml`` is a ``-with-simulation`` artifact."""
    m = build_manifest(config="configs/scwbd_001_beta.yaml", anatomy_is_biological=True)
    assert m.best_variant() == "with-simulation"
    m.validate_tag("scwbd-001-beta-with-simulation-20260806T114623Z")
    assert not m.validates("scwbd-001-beta-raw-20260806T114623Z")


def test_live_run_inherits_noncommercial_from_anatomy_not_from_tribe():
    """With the real anatomical prior, NC arrives via Hansen — TRIBE is disabled.

    This is the finding that makes the inheritance/policy split load-bearing:
    ``-with-simulation`` is non-commercial *and* share-alike even though no
    teacher source contributed.
    """
    m = build_manifest(config="configs/scwbd_001_beta.yaml", anatomy_is_biological=True)
    lic = m.licence()
    assert lic.noncommercial_by_inheritance is True
    assert lic.share_alike_by_inheritance is True
    assert "anatomical_prior" in lic.inheritance_sources
    assert "tribe_v2_teacher" not in lic.inheritance_sources
    assert lic.noncommercial_by_policy is False
    assert lic.noncommercial_is_removable is False


def test_synthetic_fallback_anatomy_does_not_inherit_noncommercial():
    """The same cards with the fallback connectome carry no Hansen terms."""
    m = build_manifest(config="configs/scwbd_001_beta.yaml", anatomy_is_biological=False)
    lic = m.licence()
    assert lic.inheritance_sources == ()


def test_unrecorded_anatomy_provenance_yields_unknown_not_permissive():
    m = build_manifest(config="configs/scwbd_001_beta.yaml", anatomy_is_biological=None)
    lic = m.licence()
    assert lic.noncommercial_by_inheritance is None
    assert "anatomical_prior" in lic.unknown_sources


# ======================================================================
# modality axis
# ======================================================================
def test_measured_modalities_span_more_than_eeg():
    """``-raw`` is all measured observation, not EEG only."""
    for m in ("eeg", "meg", "fmri", "mri", "dwi", "ieeg", "emg", "eog", "ecg"):
        assert m in MEASURED_MODALITIES


def test_dataset_cards_expose_multimodal_ground_truth():
    """Verified against the cards, not against a relayed list."""
    cards = load_dataset_cards()
    assert "fmri" in cards["ds000117"].modalities
    assert "meg" in cards["ds000117"].modalities
    assert "dwi" in cards["ds004024"].modalities
    assert cards["eegmmidb"].modalities == ("eeg",)


def test_unavailable_datasets_are_marked_unavailable():
    """A source that is not on disk trained nothing, whatever a table says."""
    cards = load_dataset_cards()
    assert cards["adni"].is_available is False
    assert cards["ukbiobank-brain-imaging"].is_available is False
    assert cards["eegmmidb"].is_available is True


def test_manifest_records_modalities_of_contributing_sources():
    m = build_manifest(config="configs/scwbd_001_beta.yaml", anatomy_is_biological=True)
    assert "eeg" in m.measured_modalities


def test_sources_without_a_dataset_card_are_listed_not_ignored():
    """A simulator has no dataset card; that fact is recorded explicitly."""
    m = build_manifest(config="configs/scwbd_001_beta.yaml", anatomy_is_biological=True)
    assert "sim_wholebrain" in m.sources_without_dataset_card


def test_link_returns_none_rather_than_a_wrong_card():
    """A bad guess is worse than a recorded gap."""
    links = link_sources_to_datasets(["sim_wholebrain", "eegmmidb_real"])
    assert links["sim_wholebrain"] is None
    assert links["eegmmidb_real"] is not None
    assert links["eegmmidb_real"].dataset_id == "eegmmidb"
