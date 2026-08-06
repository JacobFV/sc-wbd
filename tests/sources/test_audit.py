"""The card-versus-disk audit, and proof that each of its checks can fire.

``reports/decorative_guards.md`` catalogues ~26 guards in this codebase that
looked green and could not fire.  A guard that has never been observed to fail
is indistinguishable from ``pass``.  So every check in
:mod:`scwbd.sources.audit` gets two tests here:

* a **positive** test — the real cards, against the real bytes on disk, pass;
* a **negative control** — a deliberately false card, against a real tree,
  produces exactly that check's finding.

The negative controls are the load-bearing half.  Without them this file would
be one more decorative guard.
"""

from __future__ import annotations

import copy
import shutil

import pytest
import yaml

from scwbd.sources import registry
from scwbd.sources.audit import audit_all, audit_card
from scwbd.sources.cards import load_card


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _write_card(tmp_path, base_id: str, mutate) -> "object":
    """Copy a real card, mutate its data, and reload it as a card doc."""
    src = registry.get(base_id).card_path
    data = yaml.safe_load(src.read_text())
    mutate(data)
    out = tmp_path / f"{data['identity']['id']}.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False))
    return load_card(out, validate=False, typed=False)


def _real_root(dataset_id: str):
    entry = registry.get(dataset_id)
    if not entry.local_path.exists():
        pytest.skip(f"{dataset_id} not on disk at {entry.local_path}")
    return entry.local_path


def _checks(audit) -> set[str]:
    return {f.check for f in audit.findings}


# ---------------------------------------------------------------------------
# positive: the shipped cards agree with the disk
# ---------------------------------------------------------------------------
def test_every_card_agrees_with_the_filesystem():
    """The audit over the real register. Findings are printed, not summarised.

    This is the check that would have caught ds004024 declaring ``fmri`` and
    ``dwi`` with two T1w volumes and nothing else on disk.
    """
    audits = audit_all()
    bad = {cid: [str(f) for f in a.findings] for cid, a in audits.items() if not a.ok}
    assert not bad, "\n".join(f"{cid}: {msgs}" for cid, msgs in bad.items())


def test_the_audit_actually_ran_its_checks():
    """A pass over zero checks is not a pass.

    ``audit_all`` returning ``ok`` for a card it never examined is precisely
    the decorative-guard failure, so the count of executed checks is asserted
    rather than assumed.
    """
    audits = audit_all()
    assert audits, "no cards audited"
    for cid, a in audits.items():
        assert a.checks_run, f"{cid}: audit ran no checks at all"
        assert "A5_licence" in a.checks_run, cid


@pytest.mark.parametrize("dataset_id", ["ds000117", "ds004024", "eegmmidb", "sleep-edfx"])
def test_available_cards_run_the_full_check_set(dataset_id):
    _real_root(dataset_id)
    a = audit_card(load_card(registry.get(dataset_id).card_path))
    for check in ("A1_status", "A2_file_manifest", "A3_participants", "A4_modality_evidence"):
        assert check in a.checks_run, f"{dataset_id}: {check} never ran"


# ---------------------------------------------------------------------------
# negative controls: one per check
# ---------------------------------------------------------------------------
def test_A1_fires_when_a_live_card_has_no_bytes(tmp_path):
    """`status: live` with an absent root is a finding, never a skip.

    The pre-existing tests skip when a dataset is missing, which means a card
    could declare live coverage of a tree that was deleted and every test
    stayed green.
    """
    doc = _write_card(tmp_path, "eegmmidb", lambda d: None)
    a = audit_card(doc, root=tmp_path / "definitely-not-here")
    assert not a.ok
    assert "A1_status" in _checks(a)


def test_A1_fires_on_an_empty_directory(tmp_path):
    doc = _write_card(tmp_path, "eegmmidb", lambda d: None)
    empty = tmp_path / "empty"
    empty.mkdir()
    a = audit_card(doc, root=empty)
    assert "A1_status" in _checks(a)


def test_A2_fires_when_the_declared_file_count_is_wrong(tmp_path):
    root = _real_root("mne-somato")

    def mutate(d):
        d["identity"]["file_manifest"]["n_files"] += 7

    a = audit_card(_write_card(tmp_path, "mne-somato", mutate), root=root)
    assert "A2_file_manifest" in _checks(a)


def test_A2_fires_when_a_named_exemplar_is_absent(tmp_path):
    """A card naming a file that is not there is the headline failure mode."""
    root = _real_root("mne-somato")

    def mutate(d):
        d["identity"]["file_manifest"]["exemplars"]["no/such/file.fif"] = "0" * 64

    a = audit_card(_write_card(tmp_path, "mne-somato", mutate), root=root)
    assert any(f.check == "A2_file_manifest" and "no/such/file.fif" in f.claim
               for f in a.findings), [str(f) for f in a.findings]


def test_A3_fires_when_a_declared_participant_is_missing(tmp_path):
    root = _real_root("ds000117")

    def mutate(d):
        d["population"]["participant_ids"] = ["sub-01", "sub-02", "sub-03"]

    a = audit_card(_write_card(tmp_path, "ds000117", mutate), root=root)
    assert "A3_participants" in _checks(a)
    assert any("sub-03" in f.detail for f in a.findings)


def test_A3_fires_when_n_participants_overstates_the_tree(tmp_path):
    root = _real_root("ds000117")

    def mutate(d):
        d["population"]["n_participants"] = 16  # the upstream release, not our subset

    a = audit_card(_write_card(tmp_path, "ds000117", mutate), root=root)
    assert "A3_participants" in _checks(a)


def test_A4_fires_on_a_modality_with_no_file_behind_it(tmp_path):
    """The ds004024 defect, reconstructed: declare fMRI, hold no BOLD."""
    root = _real_root("ds004024")

    def mutate(d):
        d["signal"]["modalities"] = list(d["signal"]["modalities"]) + ["fmri"]
        d["signal"]["modality_evidence"]["fmri"] = ["sub-*/ses-mri/func/*_bold.nii.gz"]

    a = audit_card(_write_card(tmp_path, "ds004024", mutate), root=root)
    assert "A4_modality_evidence" in _checks(a)
    assert any("fmri" in f.claim and "0 matching files" in f.on_disk for f in a.findings)


def test_A4_fires_on_a_modality_declared_without_any_evidence(tmp_path):
    """Silence is not a pass: no evidence entry is itself the finding."""
    root = _real_root("ds004024")

    def mutate(d):
        d["signal"]["modalities"] = list(d["signal"]["modalities"]) + ["pet"]

    a = audit_card(_write_card(tmp_path, "ds004024", mutate), root=root)
    assert any(f.check == "A4_modality_evidence" and "pet" in f.claim for f in a.findings)


def test_A4_fires_when_the_evidence_file_lacks_the_claimed_channel_type(tmp_path):
    """A glob proves a file exists; ``channel_type`` proves what is in it."""
    root = _real_root("ds004024")

    def mutate(d):
        d["signal"]["modalities"] = list(d["signal"]["modalities"]) + ["ieeg"]
        d["signal"]["modality_evidence"]["ieeg"] = {
            "globs": ["sub-*/ses-*/eeg/*_channels.tsv"],
            "channel_type": "SEEG",
        }

    a = audit_card(_write_card(tmp_path, "ds004024", mutate), root=root)
    assert any(f.check == "A4_modality_evidence" and "ieeg" in f.claim for f in a.findings)


def test_A4_fires_when_the_claimed_channel_name_is_absent(tmp_path):
    root = _real_root("sleep-edfx")

    def mutate(d):
        d["signal"]["modality_evidence"]["resp"] = {
            "globs": ["sleep-cassette/*PSG.edf"],
            "channel_name": "Airflow thermistor",  # real signal, wrong label
        }

    a = audit_card(_write_card(tmp_path, "sleep-edfx", mutate), root=root)
    assert any(f.check == "A4_modality_evidence" and "resp" in f.claim for f in a.findings)


def test_A4_fires_on_evidence_for_an_undeclared_modality(tmp_path):
    root = _real_root("eegmmidb")

    def mutate(d):
        d["signal"]["modality_evidence"]["fmri"] = ["S*/*.edf"]

    a = audit_card(_write_card(tmp_path, "eegmmidb", mutate), root=root)
    assert any(f.check == "A4_modality_evidence" and "undeclared" in f.on_disk
               for f in a.findings)


def test_A5_fires_when_spdx_and_free_text_disagree(tmp_path):
    """Two independent statements of the licence; a mis-parse must surface.

    The classifier is a regex over prose (``scwbd.release.licence``). This is
    the check that notices when it is wrong, instead of routing an NC source
    into a permissive checkpoint tier in silence.
    """
    def mutate(d):
        d["governance"]["license_spdx"] = "CC-BY-NC-SA-4.0"  # card says NC ...

    a = audit_card(_write_card(tmp_path, "eegmmidb", mutate), root=tmp_path)
    # ... while governance.license still reads ODC-By, which is not NC.
    assert any(f.check == "A5_licence" and "noncommercial" in f.claim for f in a.findings)


def test_A5_fires_when_the_spdx_field_is_absent(tmp_path):
    def mutate(d):
        d["governance"].pop("license_spdx", None)

    a = audit_card(_write_card(tmp_path, "eegmmidb", mutate), root=tmp_path)
    assert "A5_licence" in _checks(a)


def test_A0_fires_when_an_unavailable_card_has_bytes(tmp_path):
    """An `unavailable` card must not be sitting on a populated tree."""
    root = tmp_path / "bytes"
    root.mkdir()
    (root / "something.bin").write_bytes(b"x" * 16)

    def mutate(d):
        d["governance"]["status"] = "unavailable"
        d["governance"]["unavailable_reason"] = (
            "negative control for the audit: this card asserts no bytes were downloaded"
        )

    a = audit_card(_write_card(tmp_path, "eegmmidb", mutate), root=root)
    assert "A0_unavailable_claims_nothing" in _checks(a)


# ---------------------------------------------------------------------------
# the licence routing the checkpoint policy depends on
# ---------------------------------------------------------------------------
def test_no_card_routes_an_unknown_licence_as_permissive(cards):
    """`unknown` must never collapse to "commercial use permitted"."""
    from scwbd.release.licence import term_from_dataset_card

    for cid, doc in cards.items():
        spdx = str(doc.data["governance"].get("license_spdx", ""))
        term = term_from_dataset_card(doc.path)
        if spdx.upper() in ("UNKNOWN", "NONE") or spdx.upper().startswith("DUA-"):
            assert term.noncommercial is None, (
                f"{cid}: spdx {spdx!r} is not a licence, but the classifier "
                f"resolved noncommercial={term.noncommercial}"
            )


def test_adding_haemodynamics_to_raw_does_not_add_an_nc_clause():
    """The routing decision the checkpoint policy actually depends on.

    ``-raw`` is specified as measured data only and explicitly not EEG-only,
    and checkpoints before the synthetic-data stage must not carry the NC
    clause.  Those two requirements are only compatible if the haemodynamic
    sources we added are themselves NC-free.  This asserts it through the
    release module's own union, not by reading the cards.
    """
    from scwbd.release.licence import term_from_dataset_card, union_of

    def union(ids):
        return union_of([term_from_dataset_card(registry.get(i).card_path) for i in ids])

    eeg_only = union(["eegmmidb"])
    with_haemo = union(["eegmmidb", "ds000117", "ds002336"])
    assert eeg_only.effective("noncommercial") is False
    assert with_haemo.effective("noncommercial") is False, (
        "adding measured haemodynamics moved the -raw arm to non-commercial; "
        f"forced by {with_haemo.sources_forcing('noncommercial')}"
    )
    assert with_haemo.effective("share_alike") is False


def test_an_unresolved_licence_poisons_the_union_to_unknown_not_to_permissive():
    """ds000113 has no licence in its bytes. That must propagate as unknown.

    The failure this guards against is the quiet one: an unresolved term read
    as "no restriction", producing a checkpoint that claims commercial use is
    permitted on the authority of a file that says nothing.
    """
    from scwbd.release.licence import term_from_dataset_card, union_of

    ids = ["eegmmidb", "ds000117", "ds002336", "ds000113"]
    u = union_of([term_from_dataset_card(registry.get(i).card_path) for i in ids])
    assert u.effective("noncommercial") is None
    assert u.effective("share_alike") is None


def test_every_card_declares_a_routable_licence_field(cards):
    for cid, doc in cards.items():
        assert doc.data["governance"].get("license_spdx"), (
            f"{cid}: no governance.license_spdx; the checkpoint policy would have "
            f"only free text to route on"
        )


# ---------------------------------------------------------------------------
# A6: a slab may not be described as a brain
# ---------------------------------------------------------------------------
def test_A6_runs_on_every_source_that_has_bold():
    audits = audit_all()
    for did in ("ds002336", "ds000113", "ds000117"):
        if not registry.get(did).local_path.exists():
            continue
        assert "A6_spatial_coverage" in audits[did].checks_run


def test_A6_fires_on_an_affirmative_whole_brain_claim(tmp_path):
    """The defect this check exists for, reconstructed.

    Both cards I wrote for the new haemodynamic sources originally said
    "whole brain". Every functional acquisition we hold is a slab.
    """
    root = _real_root("ds002336")

    def mutate(d):
        d["spatial"]["extent"] = "whole brain coverage in every subject"

    a = audit_card(_write_card(tmp_path, "ds002336", mutate), root=root)
    assert "A6_spatial_coverage" in _checks(a)


def test_A6_does_not_fire_on_a_negated_mention(tmp_path):
    """"NOT established as whole-brain" is a disclaimer, not a claim.

    `scwbd/release/licence.py` records being burned by exactly this shape --
    a helpful field ("no non-commercial term") classified as asserting the
    thing it denied. This check hit the same trap on its first run, against
    the very cards it was written to fix.
    """
    root = _real_root("ds002336")

    def mutate(d):
        d["spatial"]["extent"] = "a 128 mm slab, NOT established as whole-brain"

    a = audit_card(_write_card(tmp_path, "ds002336", mutate), root=root)
    assert "A6_spatial_coverage" not in _checks(a)


def test_A6_fires_when_the_declared_slab_contradicts_the_header(tmp_path):
    root = _real_root("ds002336")

    def mutate(d):
        d["spatial"]["coverage"]["slab_mm"] = 180        # a whole head

    a = audit_card(_write_card(tmp_path, "ds002336", mutate), root=root)
    assert any(f.check == "A6_spatial_coverage" and "slab_mm" in f.claim for f in a.findings)


def test_A6_fires_on_whole_brain_true_without_evidence(tmp_path):
    root = _real_root("ds002336")

    def mutate(d):
        d["spatial"]["coverage"]["whole_brain"] = True
        d["spatial"]["coverage"]["evidence"] = ""

    a = audit_card(_write_card(tmp_path, "ds002336", mutate), root=root)
    assert any(f.check == "A6_spatial_coverage" and "whole_brain" in f.claim
               for f in a.findings)


def test_A6_fires_when_a_volumetric_source_states_no_coverage_at_all(tmp_path):
    """Silence reads as whole-brain downstream, so silence is a finding."""
    root = _real_root("ds000117")

    def mutate(d):
        d["spatial"].pop("coverage", None)

    a = audit_card(_write_card(tmp_path, "ds000117", mutate), root=root)
    assert any(f.check == "A6_spatial_coverage" and "no spatial.coverage" in f.on_disk
               for f in a.findings)


def test_the_haemodynamic_gradient_path_is_disabled_while_coverage_is_unknown():
    """Appendix B enforcing N-4a, instead of a comment in a config file.

    `spatial.coverage.whole_brain: unknown` is named in the `requires` of every
    BOLD-facing gradient target, so the path disables itself. This is what
    stops the haemodynamic likelihood contributing a term for parcels that may
    have no observation operator at all.
    """
    for cid in ("ds002336", "ds000113"):
        doc = load_card(registry.get(cid).card_path)
        bold = [d for d in doc.gradient_decisions()
                if "bold" in d.target or "retinotop" in d.target]
        assert bold, f"{cid}: no BOLD-facing gradient target to check"
        for dec in bold:
            assert not dec.enabled, f"{cid}: {dec.target} is enabled despite unknown coverage"
            assert "spatial.coverage.whole_brain" in dec.reason
