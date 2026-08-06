"""The tag is checked against the manifest, never trusted.

The central test in this file is
:func:`test_raw_tag_with_simulated_source_fails_validation`: a checkpoint whose
name claims real data only, but whose cards show a simulated source that could
move weights, must be refused. It is constructed on purpose from cards written
for the test, because the defect it catches is one that would otherwise reach a
release.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
import yaml

from scwbd.release.families import (
    REAL,
    SIMULATION,
    SYNTHETIC,
    UNKNOWN,
    SourceRecord,
    classify,
)
from scwbd.release.manifest import (
    ProvenanceBlock,
    ProvenanceMismatch,
    build_manifest,
    config_hash,
    sha256_file,
)
from scwbd.release.tags import CheckpointTag

UTC = timezone.utc
WHEN = datetime(2026, 8, 6, 11, 46, 23, tzinfo=UTC)


# ----------------------------------------------------------------------
# card fixtures
# ----------------------------------------------------------------------
def _card(sid, role, **kw):
    d = {
        "id": sid,
        "role": role,
        "gradient_permission": kw.pop("gradient_permission", ["local.*"]),
    }
    d.update(kw)
    return d


def _write_cards(tmp_path, cards):
    d = tmp_path / "source_cards"
    d.mkdir(exist_ok=True)
    for c in cards:
        (d / f"{c['id']}.yaml").write_text(yaml.safe_dump(c))
    return d


REAL_CARD = _card("eegmmidb_real", "likelihood", is_simulated=False, losses=["likelihood"])
SIM_CARD = _card("sim_wholebrain", "prior", is_simulated=True, losses=["prior"])
TEACHER_CARD = _card(
    "tribe_v2_teacher", "distillation", is_teacher=True, losses=["distill"]
)


def _manifest(tmp_path, cards, **kw):
    return build_manifest(card_dir=_write_cards(tmp_path, cards), **kw)


# ======================================================================
# the load-bearing refusal
# ======================================================================
def test_raw_tag_with_simulated_source_fails_validation(tmp_path):
    """A ``-raw`` checkpoint whose manifest shows simulation is a defect.

    This is the requirement the whole package exists to enforce: the tag is a
    claim about provenance and must be verifiable against the artifact.
    """
    m = _manifest(tmp_path, [REAL_CARD, SIM_CARD])
    assert m.contributing_families == {REAL, SIMULATION}

    with pytest.raises(ProvenanceMismatch) as exc:
        m.validate_tag("scwbd-001-beta-raw-20260806T114623Z")

    msg = str(exc.value)
    assert "understates" in msg
    assert "simulation" in msg
    assert "sim_wholebrain" in msg, "the refusal must name the offending source"


def test_raw_tag_with_teacher_source_fails_validation(tmp_path):
    """Teacher-derived data in a ``-raw`` artifact is the same defect."""
    m = _manifest(tmp_path, [REAL_CARD, TEACHER_CARD])
    assert SYNTHETIC in m.contributing_families
    with pytest.raises(ProvenanceMismatch, match="synthetic"):
        m.validate_tag("scwbd-001-beta-raw-20260806T114623Z")


def test_raw_tag_passes_when_manifest_really_is_real_only(tmp_path):
    """The positive control: the same guard must also *accept*."""
    m = _manifest(tmp_path, [REAL_CARD])
    m.validate_tag("scwbd-001-beta-raw-20260806T114623Z")  # must not raise
    assert m.best_variant() == "raw"


def test_overclaiming_tag_is_also_refused(tmp_path):
    """Claiming simulation that never contributed overstates the artifact."""
    m = _manifest(tmp_path, [REAL_CARD])
    with pytest.raises(ProvenanceMismatch, match="overstates"):
        m.validate_tag("scwbd-001-beta-with-simulation-20260806T114623Z")


def test_disabled_source_does_not_count_as_contributing(tmp_path):
    """``enabled: false`` means the family did not train the artifact.

    TRIBE v2 is disabled in the real mixture, so a checkpoint built with that
    card present is legitimately not ``-with-...-synthetic``.
    """
    disabled = dict(TEACHER_CARD, enabled=False)
    m = _manifest(tmp_path, [REAL_CARD, SIM_CARD, disabled])
    assert SYNTHETIC not in m.contributing_families
    m.validate_tag("scwbd-001-beta-with-simulation-20260806T114623Z")
    assert ("tribe_v2_teacher", "card sets enabled: false") in m.excluded


def test_empty_gradient_permission_does_not_count_as_contributing(tmp_path):
    """A source with no ``A_k`` cannot have moved a weight, whatever its role."""
    inert = dict(SIM_CARD, gradient_permission=[])
    m = _manifest(tmp_path, [REAL_CARD, inert])
    assert SIMULATION not in m.contributing_families
    reasons = dict(m.excluded)
    assert "gradient permission" in reasons["sim_wholebrain"]


def test_every_excluded_source_carries_a_reason(tmp_path):
    """Absence must write something: no source vanishes without an explanation."""
    cards = [
        REAL_CARD,
        dict(TEACHER_CARD, enabled=False),
        _card("negative_control_shuffled", "negative_control", gradient_permission=[], losses=[]),
    ]
    m = _manifest(tmp_path, cards)
    excluded = dict(m.excluded)
    assert set(excluded) == {"tribe_v2_teacher", "negative_control_shuffled"}
    assert all(v and "unrecorded" not in v for v in excluded.values())


def test_unknown_family_source_is_recorded_not_dropped(tmp_path):
    """A non-simulated ``prior`` has no family name; it is recorded as unknown."""
    m = _manifest(tmp_path, [REAL_CARD, _card("anatomical_prior", "prior", losses=["prior"])])
    assert "anatomical_prior" in m.unknown_sources
    assert UNKNOWN in m.contributing_families
    # ...and it does not silently become a tag-axis family
    assert m.tag_families == {REAL}
    m.validate_tag("scwbd-001-beta-raw-20260806T114623Z")


def test_validation_ignores_the_filename_entirely(tmp_path):
    """Two identical manifests validate identically regardless of the name given."""
    m = _manifest(tmp_path, [REAL_CARD, SIM_CARD])
    assert m.validates("scwbd-001-beta-with-simulation-20260806T114623Z")
    assert not m.validates("scwbd-001-beta-raw-20260806T114623Z")
    assert not m.validates("scwbd-001-beta-20260806T114623Z")  # alias -> combined


# ======================================================================
# classification
# ======================================================================
@pytest.mark.parametrize(
    "kw,expected",
    [
        ({"role": "likelihood"}, REAL),
        ({"role": "prior", "is_simulated": True}, SIMULATION),
        ({"role": "distillation", "is_teacher": True}, SYNTHETIC),
        ({"role": "prior"}, UNKNOWN),
        ({"role": "calibration"}, "calibration"),
        ({"role": "boundary_target"}, "boundary"),
        ({"role": "evaluation_only"}, "evaluation_only"),
        ({"role": "negative_control"}, "negative_control"),
    ],
)
def test_classification_is_by_card_fields(kw, expected):
    assert classify(SourceRecord(id="x", **kw)) == expected


def test_teacher_flag_beats_simulated_flag():
    """A simulated teacher is synthetic; order of checks is load-bearing."""
    r = SourceRecord(id="x", role="distillation", is_teacher=True, is_simulated=True)
    assert classify(r) == SYNTHETIC


# ======================================================================
# provenance block
# ======================================================================
def test_provenance_block_carries_every_required_field(tmp_path):
    m = _manifest(tmp_path, [REAL_CARD, SIM_CARD])
    weights = tmp_path / "last.pt"
    weights.write_bytes(b"not really weights")
    block = ProvenanceBlock(
        tag=CheckpointTag.mint("with-simulation", WHEN),
        manifest=m,
        weights_sha256=sha256_file(weights),
        git_sha="deadbeef",
        config_hash=config_hash({"a": 1}),
        created_utc="20260806T114623Z",
    )
    d = block.as_dict()
    for key in (
        "tag", "timestamp_utc", "source_family_manifest", "effective_licence",
        "git_sha", "config_hash", "weights_sha256", "claim_boundary",
    ):
        assert d[key], f"provenance block is missing {key}"
    assert d["claim_boundary"] == "reports/CLAIM_BOUNDARY.md"
    assert d["source_family_manifest"]["contributing_families"] == ["real", "simulation"]
    # the licence split is present and separable
    lic = d["effective_licence"]
    assert "by_inheritance" in lic and "by_policy" in lic


def test_config_hash_is_key_order_independent():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


def test_provenance_block_round_trips_to_disk(tmp_path):
    import json

    m = _manifest(tmp_path, [REAL_CARD])
    out = tmp_path / "provenance.json"
    ProvenanceBlock(
        tag=CheckpointTag.mint("raw", WHEN),
        manifest=m,
        weights_sha256="0" * 64,
        git_sha="abc",
        config_hash="def",
        created_utc="20260806T114623Z",
    ).save(out)
    loaded = json.loads(out.read_text())
    assert loaded["tag"] == "scwbd-001-beta-raw-20260806T114623Z"
    assert loaded["variant"] == "raw"


# ======================================================================
# D12 interface
# ======================================================================
def test_d12_families_are_keyed_by_role_bucket(tmp_path):
    """Popper's D12 takes role buckets, not this package's family names."""
    m = _manifest(tmp_path, [REAL_CARD, SIM_CARD, _card("montage_calibration", "calibration")])
    fams = m.d12_families()
    assert fams["empirical"] == ["eegmmidb_real"]
    assert fams["synthetic"] == ["sim_wholebrain"]
    assert fams["calibration"] == ["montage_calibration"]
    assert set(m.d12_roles()) == set(fams)


def test_d12_unknown_bucket_is_surfaced_not_folded(tmp_path):
    """An unclassifiable source becomes an ``unknown`` bucket D12 can report."""
    m = _manifest(tmp_path, [REAL_CARD, _card("anatomical_prior", "prior", losses=["prior"])])
    assert "unknown" in m.d12_families()
    assert m.d12_families()["unknown"] == ["anatomical_prior"]
