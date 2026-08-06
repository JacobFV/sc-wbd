"""Tier derivation: from the cards, and failing on demand."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scwbd.curriculum.tiers import (
    REFUSAL_UNDECLARED_PROVENANCE,
    TIER_NONE,
    load_mixture_cards,
    tier_of,
)

REPO = Path(__file__).resolve().parents[2]
CORRECTED = REPO / "configs/curriculum/source_cards"
LEGACY = REPO / "configs/source_cards"


def test_corrected_cards_tier_as_designed() -> None:
    cards = load_mixture_cards(CORRECTED)
    got = {sid: tier_of(c).tier for sid, c in cards.items()}
    assert got == {
        "eegmmidb_real": 1,
        "sleepedf_real": 1,
        "montage_calibration": 2,
        "anatomical_prior": 3,
        "sim_wholebrain": 4,
        "tribe_v2_teacher": 5,
        "negative_control_shuffled": TIER_NONE,
    }


def test_simulated_prior_is_tier_4_not_tier_3() -> None:
    """The discriminator is ``is_simulated``, not ``role``.

    Both the population prior and the simulated corpus carry ``role: prior`` --
    the simulator has to, because ``SourceSpec`` refuses a simulated likelihood.
    Reading the role alone would collapse tier 4 into tier 3, which is the whole
    ordering error in one line.
    """
    cards = load_mixture_cards(CORRECTED)
    assert cards["sim_wholebrain"].spec.role == "prior"
    assert cards["anatomical_prior"].spec.role == "prior"
    assert tier_of(cards["sim_wholebrain"]).tier == 4
    assert tier_of(cards["anatomical_prior"]).tier == 3


def test_omitted_provenance_is_refused_not_defaulted(tmp_path: Path) -> None:
    """The guard fires: a ``role: prior`` card with no ``is_simulated`` is refused.

    ``SourceSpec.is_simulated`` defaults to ``False``, so an omission would be
    read as "measured" and promote a simulated corpus from tier 4 to tier 3.
    Absence must not read as a declaration.
    """
    d = tmp_path / "cards"
    d.mkdir()
    (d / "mystery.yaml").write_text(
        yaml.safe_dump(
            {"id": "mystery", "role": "prior", "losses": ["prior"], "gradient_permission": ["local.*"]}
        )
    )
    card = load_mixture_cards(d)["mystery"]
    a = tier_of(card)
    assert a.tier is None
    assert a.refusal == REFUSAL_UNDECLARED_PROVENANCE

    # ...and reads differently once the field is present.
    (d / "mystery.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "mystery",
                "role": "prior",
                "is_simulated": False,
                "losses": ["prior"],
                "gradient_permission": ["local.*"],
            }
        )
    )
    a2 = tier_of(load_mixture_cards(d)["mystery"])
    assert a2.refusal is None and a2.tier == 3


def test_current_mixture_has_an_untiered_card() -> None:
    """The shipped mixture really does contain the omission the guard catches."""
    a = tier_of(load_mixture_cards(LEGACY)["anatomical_prior"])
    assert a.refusal == REFUSAL_UNDECLARED_PROVENANCE


def test_sidecar_orphan_is_an_error(tmp_path: Path) -> None:
    """Metadata for a card that does not exist must not sit unnoticed."""
    d = tmp_path / "source_cards"
    d.mkdir()
    (d / "a.yaml").write_text(yaml.safe_dump({"id": "a", "role": "likelihood"}))
    (tmp_path / "card_metadata.yaml").write_text(
        yaml.safe_dump({"sources": {"a": {"observes": ["eeg"]}, "ghost": {"observes": ["eeg"]}}})
    )
    with pytest.raises(KeyError, match="ghost"):
        load_mixture_cards(d)


def test_sidecar_is_visible_to_declares(tmp_path: Path) -> None:
    d = tmp_path / "source_cards"
    d.mkdir()
    (d / "a.yaml").write_text(yaml.safe_dump({"id": "a", "role": "likelihood"}))
    (tmp_path / "card_metadata.yaml").write_text(
        yaml.safe_dump({"sources": {"a": {"observes": ["eeg"]}}})
    )
    card = load_mixture_cards(d)["a"]
    assert card.declares("observes") and card.raw["observes"] == ["eeg"]


def test_corrected_cards_load_through_the_trainer() -> None:
    """A card the trainer cannot load is worse than a split record.

    This is why ``observes:`` lives in ``card_metadata.yaml``: ``load_dir``
    constructs ``SourceSpec(**yaml)`` and raises on any undeclared key.
    """
    from scwbd.foundation.mixture import SourceSpec

    specs = SourceSpec.load_dir(CORRECTED)
    assert set(specs) == set(load_mixture_cards(CORRECTED))
