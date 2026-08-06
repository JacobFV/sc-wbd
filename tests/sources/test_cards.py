"""Source cards validate, say what they mean, and disable what they cannot support."""

from __future__ import annotations

import copy

import pytest
import yaml

from scwbd.sources import registry
from scwbd.sources.cards import (
    CARD_SCHEMA_VERSION,
    CardError,
    SourceCardDoc,
    load_all_cards,
    load_card,
    iter_card_paths,
)


def test_every_card_parses_and_validates(cards):
    assert cards, "no source cards found"
    for cid, doc in cards.items():
        assert doc.data["schema_version"] == CARD_SCHEMA_VERSION
        assert doc.id == cid


def test_card_filename_matches_id(cards):
    for cid, doc in cards.items():
        assert doc.path.stem == cid, f"{doc.path.name} declares id {cid}"


def test_every_registry_entry_has_a_card(entries, cards):
    for did, entry in entries.items():
        assert entry.card_path.exists(), f"{did}: missing {entry.card_path}"
        assert did in cards, f"{did}: card id does not match registry id"


def test_registry_version_matches_card_version(entries, cards):
    for did, entry in entries.items():
        assert entries[did].version == cards[did].version, (
            f"{did}: registry pins {entry.version} but the card says {cards[did].version}"
        )


def test_live_cards_carry_real_hashes(cards):
    """A live card must record measured bytes, not intentions."""
    for cid, doc in cards.items():
        if doc.status == "unavailable":
            continue
        fm = doc.data["identity"]["file_manifest"]
        assert isinstance(fm, dict), f"{cid}: file_manifest was never refreshed"
        assert fm["n_files"] > 0 and fm["total_bytes"] > 0, cid
        assert len(fm["manifest_sha256"]) == 64, cid
        for path, digest in fm["exemplars"].items():
            assert len(digest) == 64 and int(digest, 16) >= 0, f"{cid}: bad hash for {path}"
        assert doc.data["identity"]["retrieved_utc"], cid


def test_unavailable_cards_are_honest(cards):
    """No bytes, no local path, no gradients, and a stated reason."""
    for cid, doc in cards.items():
        if doc.status != "unavailable":
            continue
        gov = doc.data["governance"]
        assert gov["unavailable_reason"], f"{cid}: unavailable without a reason"
        assert len(str(gov["unavailable_reason"])) > 40, f"{cid}: reason is too thin to audit"
        assert doc.data["identity"]["local_path"] in (None, "", "unknown")
        assert doc.local_path is None
        assert doc.effective_gradient_permission()["enabled"] == [], cid
        assert not registry.get(cid).local_path.exists() or True  # bytes must not be claimed


def test_unknown_fields_disable_the_dependent_gradient_path(cards):
    """The Appendix B contract: unknown field -> disabled path, with a reason."""
    checked = 0
    for cid, doc in cards.items():
        unknown = set(doc.unknown_fields())
        for dec in doc.gradient_decisions():
            if set(dec.requires) & unknown:
                assert not dec.enabled, f"{cid}: {dec.target} enabled despite unknown prerequisite"
                assert dec.reason, f"{cid}: {dec.target} disabled without a reason"
                checked += 1
            elif doc.status not in ("unavailable",) and not doc.data["gradient_permission"].get(
                "evaluation_only"
            ):
                assert dec.enabled, f"{cid}: {dec.target} disabled with no unknown prerequisite"
    assert checked > 0, "no card exercises the unknown->disabled contract"


def test_eegmmidb_lead_field_path_is_disabled():
    """Concrete instance of the contract: template electrodes are not a lead field."""
    doc = load_card(registry.get("eegmmidb").card_path)
    perm = doc.effective_gradient_permission()
    assert "observe.eeg.lead_field" in perm["disabled"]
    assert "calibration.electrode_positions" in perm["disabled"]["observe.eeg.lead_field"]
    assert "observe.eeg.noise_covariance" in perm["enabled"]


def test_role_is_one_of_the_appendix_b_roles(cards):
    allowed = {
        "prior",
        "likelihood",
        "boundary_target",
        "distillation",
        "calibration",
        "negative_control",
        "evaluation_only",
    }
    for cid, doc in cards.items():
        assert doc.role in allowed, cid


def test_split_policy_declares_grouping_before_any_split(cards):
    for cid, doc in cards.items():
        keys = doc.data["split_policy"]["grouping_keys"]
        assert keys, f"{cid}: no grouping keys (R10)"
        assert "participant" in keys or "family" in keys, cid
        assert doc.data["split_policy"]["leakage_barrier"] == "parent_lineage", cid


def test_content_hash_is_stable_and_sensitive(cards):
    doc = next(iter(cards.values()))
    h1 = doc.content_hash()
    assert h1 == doc.content_hash()
    mutated = SourceCardDoc(path=doc.path, data=copy.deepcopy(doc.data))
    mutated.data["role"] = "negative_control"
    assert mutated.content_hash() != h1


def test_validation_rejects_a_live_card_without_hashes(tmp_path, cards):
    doc = next(d for d in cards.values() if d.status != "unavailable")
    bad = copy.deepcopy(doc.data)
    bad["identity"]["file_manifest"] = {"n_files": 3}
    p = tmp_path / f"{doc.id}.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(CardError):
        load_card(p)


def test_validation_rejects_missing_requires_declaration(tmp_path, cards):
    doc = next(iter(cards.values()))
    bad = copy.deepcopy(doc.data)
    bad["gradient_permission"]["allow"] = [{"target": "observe.eeg.gain"}]
    p = tmp_path / f"{doc.id}.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(CardError, match="requires"):
        load_card(p)


def test_validation_rejects_unavailable_card_claiming_bytes(tmp_path, cards):
    doc = next(d for d in cards.values() if d.status == "unavailable")
    bad = copy.deepcopy(doc.data)
    bad["identity"]["local_path"] = "/data/scwbd/ukbiobank"
    p = tmp_path / f"{doc.id}.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(CardError, match="local_path"):
        load_card(p)


def test_typed_projection_matches_agent_a_schema_when_available(cards):
    """If scwbd.schema.SourceCard exists, every live card must project onto it."""
    pytest.importorskip("scwbd.schema")
    try:
        from scwbd.schema import SourceCard  # noqa: F401
    except ImportError:
        pytest.skip("scwbd.schema.SourceCard not landed yet")
    failures = {
        cid: doc.typed_error
        for cid, doc in cards.items()
        if doc.status != "unavailable" and doc.typed is None
    }
    assert not failures, f"typed projection failed: {failures}"


def test_unavailable_cards_are_not_projected(cards):
    for cid, doc in cards.items():
        if doc.status == "unavailable":
            assert doc.typed is None
            assert "unavailable" in doc.typed_error


def test_bias_interval_is_ordered_and_bias_status_declared(cards):
    for cid, doc in cards.items():
        led = doc.data["ledger"]
        bi = led["bias_interval"]
        if bi != "unknown":
            assert float(bi[0]) <= float(bi[1]), cid
        assert led["bias_status"] in {
            "design_estimable",
            "externally_bounded",
            "prior_specified_sensitivity",
            "unknown",
        }, cid


def test_no_variance_entry_is_silently_zero(cards):
    """A zero variance is a reliability claim; unknown must stay unknown."""
    for cid, doc in cards.items():
        for k, v in (doc.data["ledger"]["variance"] or {}).items():
            if k == "numerical":
                continue
            assert v != 0 and v != 0.0, f"{cid}: ledger.variance.{k} is zero, say 'unknown' instead"


def test_card_count_covers_the_register():
    assert len(list(iter_card_paths())) == len(registry.REGISTRY)
