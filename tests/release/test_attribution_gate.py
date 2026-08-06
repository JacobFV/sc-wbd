"""The release path refuses to write an unattributable artifact.

`ARCHITECTURE.md` §7a makes inherited data attribution and permissions the
compliance surface. For `scwbd.anatomy.sources.SRC["tian2020"]` citation *is*
the licence condition, and ODC-By carries attribution as its only obligation.
So a checkpoint that records "you inherited an attribution obligation" without
recording *what to attribute* cannot be used in compliance with it.

`scwbd.sources.attribution` built the block. This file is about the other
half — that the block is **called**, and that calling it can stop a release.
A mechanism that exists and is never invoked is the exact category
`reports/decorative_guards.md` names, so every test here verifies by
execution: the artifact is written, or it is not, and the filesystem is
checked either way.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scwbd.release.manifest import ProvenanceBlock, build_manifest
from scwbd.release.tags import CheckpointTag
from scwbd.sources.attribution import AttributionError

CONFIG = "configs/scwbd_001_beta.yaml"


def _block(manifest=None) -> ProvenanceBlock:
    m = manifest or build_manifest(config=CONFIG, anatomy_is_biological=False)
    return ProvenanceBlock(
        tag=CheckpointTag(variant="raw", timestamp=datetime.now(timezone.utc)),
        manifest=m,
        weights_sha256="0" * 64,
        git_sha="deadbeef",
        config_hash="c0ffee",
        created_utc="2026-08-06T00:00:00Z",
    )


class _Rec:
    def __init__(self, i: str) -> None:
        self.id = i


class _BrokenManifest:
    """A manifest with a contributing source that links to no dataset card.

    This is not hypothetical: a combined `haemodynamic_real` mixture card
    produced exactly this state, because `link_sources_to_datasets` maps one
    mixture id to at most one dataset card. The licence union bound it to
    UNKNOWN_TERM and the checkpoint would have carried "unknown" instead of the
    CC0 terms of the two datasets it trained on.
    """

    records = ()
    contributing = (_Rec("eegmmidb_real"), _Rec("two_dataset_card_real"))
    dataset_links = {
        "eegmmidb_real": type("I", (), {"dataset_id": "eegmmidb"})(),
        "two_dataset_card_real": None,
    }
    policy: dict = {}

    def licence(self):
        from scwbd.release.licence import union_of

        return union_of(())

    def as_dict(self):
        return {"stub": True}


# ---------------------------------------------------------------------------
# the artifact carries its citations
# ---------------------------------------------------------------------------
def test_provenance_block_carries_an_attribution_section():
    d = _block().as_dict()
    assert "attribution" in d and "attribution_text" in d
    assert d["attribution"]["ok"] is True
    assert d["attribution"]["citations"], "an artifact with no citations has no provenance"


def test_attribution_and_licence_are_computed_over_the_same_sources():
    """They must not be able to drift apart.

    Both derive from ``dataset_links`` over ``contributing``. If a future
    change lets the licence union see a source the citation set does not, an
    artifact could inherit an obligation it does not name.
    """
    m = build_manifest(config=CONFIG, anatomy_is_biological=False)
    block = _block(m)
    attributed = {e["key"] for e in block.as_dict()["attribution"]["entries"]}
    licensed = {
        info.dataset_id
        for sid, info in m.dataset_links.items()
        if info is not None and sid in {r.id for r in m.contributing}
    }
    assert attributed == licensed, (attributed, licensed)


def test_saved_record_contains_the_citation_text(tmp_path):
    out = tmp_path / "provenance.json"
    _block().save(out)
    payload = json.loads(out.read_text())
    assert "Schalk" in payload["attribution_text"], "the citation did not reach the file"
    assert payload["attribution"]["ok"] is True


# ---------------------------------------------------------------------------
# the gate: verified by execution, not by reading the code
# ---------------------------------------------------------------------------
def test_an_unattributable_artifact_cannot_be_written(tmp_path):
    """The load-bearing test. Refusal, and NO FILE ON DISK."""
    out = tmp_path / "provenance.json"
    block = _block(_BrokenManifest())
    with pytest.raises(AttributionError, match="cannot be attributed"):
        block.save(out)
    assert not out.exists(), (
        "save() refused but still wrote a file; a half-written provenance record "
        "is worse than none, because downstream reads one that exists and is short"
    )


def test_the_refusal_names_the_offending_source(tmp_path):
    block = _block(_BrokenManifest())
    with pytest.raises(AttributionError) as exc:
        block.save(tmp_path / "p.json")
    assert "two_dataset_card_real" in str(exc.value)
    # and it says why it matters, not just that it happened
    assert "tian2020" in str(exc.value)


def test_the_refusal_survives_a_source_that_merely_looks_fine(tmp_path):
    """The defect was invisible in the card: spdx and citation were correct.

    `haemodynamic_real` had a valid `license_spdx` and a valid citation. What
    was wrong was that nothing linked it to a dataset. So the gate must key on
    the LINK, not on the card's contents — otherwise it passes exactly the case
    it exists for.
    """
    block = _block(_BrokenManifest())
    att = block.attribution()
    assert not att.ok
    assert any(k == "two_dataset_card_real" for k, _ in att.unattributable)
    # the source that DOES link is still attributed; one hole does not blank the block
    assert [e.key for e in att.entries] == ["eegmmidb"]


def test_the_escape_hatch_is_explicit_and_still_records_the_hole(tmp_path):
    """`require_attribution=False` exists for debugging a broken registry.

    It must not launder the artifact: the written record still says the
    attribution is incomplete, so a file produced this way is identifiable
    afterwards rather than indistinguishable from a compliant one.
    """
    out = tmp_path / "provenance.json"
    _block(_BrokenManifest()).save(out, require_attribution=False)
    payload = json.loads(out.read_text())
    assert payload["attribution"]["ok"] is False
    assert payload["attribution"]["unattributable"]
    assert "NOT COMPLIANT" in payload["attribution_text"]


def test_the_gate_is_on_by_default():
    """A control that must be opted into is not a control."""
    import inspect

    sig = inspect.signature(ProvenanceBlock.save)
    assert sig.parameters["require_attribution"].default is True
