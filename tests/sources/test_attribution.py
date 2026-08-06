"""The attribution block, and proof it fails when attribution is impossible.

`ARCHITECTURE.md` §7a makes inherited attribution a licence obligation, and for
`scwbd.anatomy.sources.SRC["tian2020"]` citation *is* the licence condition. So
a checkpoint that cannot state what it was built from is out of compliance, and
an attribution block that renders no matter what is a decorative control.

Every check here therefore comes in a pair: the block is correct on the real
registries, **and** it refuses when a contributing source cannot be cited.
"""

from __future__ import annotations

import pytest
import yaml

from scwbd.sources import registry
from scwbd.sources.attribution import (
    AttributionError,
    attribution_for_anatomy,
    attribution_for_checkpoint,
    attribution_for_datasets,
    attribution_from_manifest,
)


# ---------------------------------------------------------------------------
# derived, not restated
# ---------------------------------------------------------------------------
def test_every_dataset_card_yields_an_attribution():
    """No registered dataset may be unattributable."""
    ids = sorted(registry.REGISTRY)
    block = attribution_for_datasets(ids)
    assert block.ok, block.unattributable
    assert len(block.entries) == len(ids)
    for e in block.entries:
        assert len(e.citation) > 30, f"{e.key}: citation is too thin to be one"
        assert e.provenance.endswith(".yaml")


def test_citations_come_from_the_card_not_from_this_module():
    """The rendered citation must be byte-identical to the card's own text.

    A citation typed into the attribution module would drift from the card
    silently. This is the check that would catch that drift.
    """
    for did in ("eegmmidb", "ds002336", "ds000113"):
        card = yaml.safe_load(registry.get(did).card_path.read_text())
        want = " ".join(str(card["identity"]["citation"]).split())
        got = attribution_for_datasets([did]).entries[0].citation
        assert got == want, did


def test_licence_spdx_travels_with_the_citation():
    e = {a.key: a for a in attribution_for_datasets(["ds002336", "ds000113"]).entries}
    assert e["ds002336"].licence_spdx == "CC0-1.0"
    assert e["ds000113"].licence_spdx == "UNKNOWN"


def test_anatomy_attribution_is_read_from_the_anatomy_registry():
    block = attribution_for_anatomy(["schaefer2018", "tian2020", "hansen_receptors"])
    assert block.ok, block.unattributable
    by = {e.key: e for e in block.entries}
    # Tian's licence condition IS citation -- the reason this module exists.
    assert "cites Tian" in by["tian2020"].licence or "Attribution" in by["tian2020"].licence
    assert "Tian Y." in by["tian2020"].citation
    # Hansen is the NC-SA input; its term must appear verbatim, not paraphrased.
    assert "NC" in by["hansen_receptors"].licence


def test_rendered_block_names_every_input():
    block = attribution_for_checkpoint(
        dataset_ids=["eegmmidb", "ds002336"],
        anatomy_keys=["schaefer2018", "tian2020"],
        tag="scwbd-001-beta-raw",
    )
    text = block.render()
    for key in ("eegmmidb", "ds002336", "schaefer2018", "tian2020"):
        assert key in text
    assert "scwbd-001-beta-raw" in text
    assert block.ok


# ---------------------------------------------------------------------------
# negative controls: the block must be able to fail
# ---------------------------------------------------------------------------
def test_a_dataset_with_no_card_is_unattributable_not_omitted():
    block = attribution_for_datasets(["eegmmidb", "no-such-dataset"])
    assert not block.ok
    assert any(k == "no-such-dataset" for k, _ in block.unattributable)
    # and it must be visible in the rendered text, not just the dict
    assert "NOT COMPLIANT" in block.render()


def test_a_card_without_a_citation_is_unattributable(tmp_path):
    card = yaml.safe_load(registry.get("eegmmidb").card_path.read_text())
    card["identity"].pop("citation", None)
    (tmp_path / "eegmmidb.yaml").write_text(yaml.safe_dump(card, sort_keys=False))
    block = attribution_for_datasets(["eegmmidb"], card_dir=tmp_path)
    assert not block.ok
    assert "no identity.citation" in block.unattributable[0][1]


def test_a_placeholder_citation_does_not_count(tmp_path):
    """'unknown' is not a citation. It renders like one, which is the danger."""
    card = yaml.safe_load(registry.get("eegmmidb").card_path.read_text())
    card["identity"]["citation"] = "unknown - not established"
    (tmp_path / "eegmmidb.yaml").write_text(yaml.safe_dump(card, sort_keys=False))
    block = attribution_for_datasets(["eegmmidb"], card_dir=tmp_path)
    assert not block.ok


def test_an_unknown_anatomy_key_is_unattributable():
    block = attribution_for_anatomy(["schaefer2018", "harvard_oxford_wrong_key"])
    assert not block.ok
    assert any("harvard_oxford_wrong_key" in k for k, _ in block.unattributable)


def test_require_complete_raises_and_names_the_licence_condition():
    block = attribution_for_datasets(["no-such-dataset"])
    with pytest.raises(AttributionError, match="tian2020"):
        block.require_complete()


def test_require_complete_passes_on_a_real_set():
    attribution_for_checkpoint(
        dataset_ids=["eegmmidb", "ds000117"], anatomy_keys=["schaefer2018"]
    ).require_complete()


def test_an_artifact_with_no_inputs_is_not_compliant():
    """A checkpoint that names no source has no provenance, not a clean one."""
    block = attribution_for_checkpoint(dataset_ids=[], anatomy_keys=[])
    assert not block.ok
    with pytest.raises(AttributionError):
        block.require_complete()


# ---------------------------------------------------------------------------
# the manifest route: attribution and licence must see the same sources
# ---------------------------------------------------------------------------
def _manifest():
    from scwbd.release.manifest import build_manifest

    return build_manifest(
        config="configs/scwbd_001_beta.yaml", anatomy_is_biological=False
    )


def test_attribution_is_derived_from_the_same_links_as_the_licence():
    """The attribution set and the licence set must not be able to drift.

    Both are computed from ``manifest.dataset_links`` over
    ``manifest.contributing``. If a future change lets one see a source the
    other does not, this fails.
    """
    m = _manifest()
    block = attribution_from_manifest(m, tag="scwbd-001-beta-raw")
    attributed = {e.key for e in block.entries}
    licensed = {
        info.dataset_id
        for sid, info in m.dataset_links.items()
        if info is not None and sid in {r.id for r in m.contributing}
    }
    assert attributed == licensed, (attributed, licensed)


def test_run1_checkpoint_attributes_exactly_one_dataset_and_it_is_eeg():
    """The `-raw` EEG-only finding, printed by the artifact about itself.

    Recorded as a test so that when a haemodynamic source is enabled this
    fails, loudly, and has to be updated deliberately.
    """
    m = _manifest()
    block = attribution_from_manifest(m)
    assert block.ok, block.unattributable
    assert [e.key for e in block.entries] == ["eegmmidb"], (
        "run 1's contributing dataset set changed; if a haemodynamic source was "
        "enabled, update reports/sources/inventory.md §3 with it"
    )


def test_a_contributing_source_with_no_dataset_link_is_reported():
    """The defect that a combined `haemodynamic_real` card actually produced.

    One mixture id links to at most one dataset card, matched by normalising
    the id (`<x>_real` -> `<x>`). A card naming two datasets matched neither,
    so the checkpoint would have carried UNKNOWN instead of the obligations of
    the datasets it trained on. Found by calling ``build_manifest`` and reading
    the link, and pinned here with a stand-in manifest because the real one is
    frozen.
    """
    class _Rec:
        def __init__(self, i):
            self.id = i

    class _Stub:
        records = ()
        contributing = (_Rec("eegmmidb_real"), _Rec("invented_source_real"))
        dataset_links = {
            "eegmmidb_real": registry.get("eegmmidb"),
            "invented_source_real": None,
        }

    stub = _Stub()
    # dataset_links values must expose .dataset_id, as DatasetInfo does.
    stub.dataset_links["eegmmidb_real"] = type("I", (), {"dataset_id": "eegmmidb"})()
    block = attribution_from_manifest(stub)
    assert not block.ok
    assert any(k == "invented_source_real" for k, _ in block.unattributable)
    # the source that DOES link is still attributed -- one hole does not erase
    # the rest of the block
    assert [e.key for e in block.entries] == ["eegmmidb"]


def test_non_dataset_sources_are_not_reported_as_holes():
    """The simulator and the prior are not datasets and never will be.

    Reporting them as unattributable would make the block cry wolf on every
    run, and a control that always fails is as useless as one that never does.
    """
    m = _manifest()
    block = attribution_from_manifest(m)
    holes = {k for k, _ in block.unattributable}
    for known in ("sim_wholebrain", "anatomical_prior", "montage_calibration",
                  "negative_control_shuffled"):
        assert known not in holes
