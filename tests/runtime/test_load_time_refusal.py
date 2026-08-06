"""The gate fires at ``load``, not at first use, and not in a docstring.

Run 1 shipped and was demonstrated with no such check.  What made that possible
is reconstructed here as a regression test: ``discover_checkpoint`` did not
recognise the filename the trainer writes (``last.pt``), so it reported
``found=False`` for a directory containing a real checkpoint, and every
downstream guard keyed on ``weights_status`` passed because its input never
arrived.

These tests use a synthesised checkpoint directory so they run anywhere.  The
last class additionally reads the **real** run-1 artifact when it is present on
this machine, and is skipped when it is not -- a skipped test is honest; a test
that silently passes on a missing fixture is the pattern this repository keeps
finding in itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scwbd.runtime.admission import CheckpointRefused
from scwbd.runtime.provenance import ProvenanceExpectation, ProvenanceMismatch
from scwbd.runtime.serving import ServedModel, discover_checkpoint

from scwbd.runtime.ports import PortContract

from test_ports import RUN1_LAYOUT, RUN2_LAYOUT


#: The real run-1 artifact, written by agent Turing's training run.
REAL_RUN1 = Path("/home/brandonin/Documents/scwbd-wt/turing/checkpoints")


def write_checkpoint(root: Path, name: str, manifest: dict | None, *,
                     weights_name: str = "last.pt") -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / weights_name).write_bytes(b"not a real tensor, but a real file")
    if manifest is not None:
        (d / "claim_manifest.json").write_text(json.dumps(manifest))
    return d


def run1_manifest() -> dict:
    """What a truthful sidecar for SC-WBD-001-beta says."""
    return {
        "id": "scwbd-001-beta",
        "claim_class": "surrogate",
        "posterior_class": "pseudo",
        "is_control_arm": True,
        "arm": "equal_capacity_generic_operator_control (body.tex 11.4)",
        "anatomy": {"is_biological": False, "provenance": "synthetic_fallback"},
        "gates": {f"G{i}": "COULD_NOT_RUN" for i in range(1, 6)},
        "state_layout": RUN1_LAYOUT,
    }


def clean_manifest() -> dict:
    return {
        "id": "scwbd-002-treatment",
        "claim_class": "mechanistic",
        "posterior_class": "generalized",
        "is_control_arm": False,
        "arm": "treatment",
        "anatomy": {"is_biological": True, "provenance": "enigma_hcp_dki"},
        "gates": {f"G{i}": "PASS" for i in range(1, 6)},
        "state_layout": RUN2_LAYOUT,
    }


# ---------------------------------------------------------------------------
# the regression that made run 1 possible
# ---------------------------------------------------------------------------

class TestCheckpointDiscoveryIsNotBlindToTheTrainersFilenames:
    @pytest.mark.parametrize(
        "weights_name",
        ["last.pt", "weights.pt", "stage_V_individual.pt", "stage_I_regional.pt"],
    )
    def test_the_names_the_trainer_writes_are_discovered(self, tmp_path, weights_name):
        write_checkpoint(tmp_path, "m", clean_manifest(), weights_name=weights_name)
        rec = discover_checkpoint("m", root=tmp_path)
        assert rec.found
        assert rec.weights_status == "trained"
        assert rec.weights_sha256

    def test_a_directory_with_no_recognised_weights_is_still_not_trained(self, tmp_path):
        d = tmp_path / "m"
        d.mkdir()
        (d / "notes.txt").write_text("x")
        rec = discover_checkpoint("m", root=tmp_path)
        assert not rec.found
        assert rec.weights_status == "analytic_backend"


# ---------------------------------------------------------------------------
# refusal at load
# ---------------------------------------------------------------------------

class TestLoadShipsTheArtifactAndLabelsIt:
    """Sec. 7a. Only an unreadable manifest blocks; quality is labelled."""

    @pytest.mark.parametrize("purpose", ["simulation", "research_offline",
                                         "live_hardware", "patient_directed"])
    def test_the_control_arm_loads_for_every_purpose_and_is_labelled(
        self, tmp_path, purpose
    ):
        write_checkpoint(tmp_path, "scwbd-001-beta", run1_manifest())
        served = ServedModel.load(
            "scwbd-001-beta", device="cpu", checkpoint_root=tmp_path, purpose=purpose,
        )
        assert served.admission.admitted
        assert set(served.admission.label_codes) == {"L1", "L2", "L3"}
        banner = served.provenance.notes["admission_banner"]
        assert "control arm" in banner
        assert "synthetic_fallback" in banner
        assert "COULD_NOT_RUN" in banner

    def test_a_checkpoint_with_no_manifest_refuses_because_it_cannot_be_labelled(
        self, tmp_path
    ):
        write_checkpoint(tmp_path, "bare", None)
        with pytest.raises(CheckpointRefused) as exc:
            ServedModel.load(
                "bare", device="cpu", checkpoint_root=tmp_path,
                purpose="research_offline",
            )
        assert exc.value.codes == ("A1",)
        assert "every label below would be a guess" in str(exc.value)

    def test_that_refusal_happens_before_any_service_object_exists(self, tmp_path):
        write_checkpoint(tmp_path, "bare", None)
        served = None
        try:
            served = ServedModel.load(
                "bare", device="cpu", checkpoint_root=tmp_path,
                purpose="patient_directed",
            )
        except CheckpointRefused:
            pass
        assert served is None

    def test_a_clean_checkpoint_loads_with_no_banner_at_all(self, tmp_path):
        """Paired with the above: the labelling discriminates."""
        write_checkpoint(tmp_path, "clean", clean_manifest())
        served = ServedModel.load(
            "clean", device="cpu", checkpoint_root=tmp_path, purpose="live_hardware",
        )
        assert served.admission.admitted
        assert served.admission.is_clean
        assert served.provenance.notes["admission_banner"] == ""
        assert served.provenance.notes["admission_flagged"] == []
        assert served.provenance.notes["consumer_standing_invariants"] == {
            "sim2real_ready": False,
            "promotion_eligible": False,
            "robot_command_authority": False,
        }

    def test_the_labels_reach_the_provenance_notes(self, tmp_path):
        write_checkpoint(tmp_path, "scwbd-001-beta", run1_manifest())
        served = ServedModel.load(
            "scwbd-001-beta", device="cpu", checkpoint_root=tmp_path,
            purpose="simulation",
        )
        labels = served.provenance.notes["admission_labels"]
        assert labels["L4"] == "clean"          # the fixture writes real bytes
        assert "control arm" in labels["L1"]
        assert served.provenance.notes["admission_hash"]


class TestThePortContractIsServedAndPinnable:
    def test_the_declared_ports_reach_the_consumer(self, tmp_path):
        write_checkpoint(tmp_path, "clean", clean_manifest())
        served = ServedModel.load(
            "clean", device="cpu", checkpoint_root=tmp_path,
            purpose="research_offline",
        )
        contract = served.port_contract()
        assert set(contract.families) == {
            "hippocampal", "cortical_visual", "brainstem"
        }
        assert served.provenance.port_contract_digest == contract.digest()
        assert "cortical_visual.retinotopic" in served.provenance.exported_ports

    def test_a_consumer_pinning_the_digest_survives_nothing_silently(self, tmp_path):
        """Run 1's layout served to a consumer pinned to run 2's must fail."""
        write_checkpoint(tmp_path, "one", run1_manifest())
        write_checkpoint(tmp_path, "two", clean_manifest())
        two = ServedModel.load("two", device="cpu", checkpoint_root=tmp_path,
                               purpose="research_offline")
        one = ServedModel.load("one", device="cpu", checkpoint_root=tmp_path,
                               purpose="simulation")

        pinned = ProvenanceExpectation(
            port_contract_digest=two.provenance.port_contract_digest,
            require_exported_ports=("cortical_visual.retinotopic",),
        )
        # the artifact it was written against: fine
        pinned.check(two.provenance)
        # the other one: a hard failure naming both problems
        with pytest.raises(ProvenanceMismatch) as exc:
            pinned.check(one.provenance)
        assert any("port_contract_digest" in m for m in exc.value.mismatches)
        assert any("exported_ports" in m for m in exc.value.mismatches)

    def test_a_checkpoint_declaring_no_layout_reports_no_ports(self, tmp_path):
        m = clean_manifest()
        m.pop("state_layout")
        write_checkpoint(tmp_path, "nolayout", m)
        served = ServedModel.load("nolayout", device="cpu", checkpoint_root=tmp_path,
                                  purpose="research_offline")
        assert served.provenance.port_contract_digest == ""
        assert served.provenance.exported_ports == ()
        # ...and a consumer that requires a port gets a hard failure, not silence
        with pytest.raises(ProvenanceMismatch):
            ProvenanceExpectation(
                require_exported_ports=("all_regions.rate_e",)
            ).check(served.provenance)


# ---------------------------------------------------------------------------
# the real artifact
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (REAL_RUN1 / "scwbd-001-beta" / "last.pt").is_file(),
    reason="the real run-1 checkpoint is not on this machine",
)
class TestAgainstTheRealRunOneArtifact:
    def test_it_is_now_discovered_as_trained(self):
        rec = discover_checkpoint("scwbd-001-beta", root=REAL_RUN1)
        assert rec.found, (
            "the real checkpoint directory must be recognised; reporting "
            "found=False for a directory containing last.pt is what let run 1 "
            "ship with every weights_status guard green"
        )
        assert rec.weights_status == "trained"

    def test_it_ships_no_claim_manifest_and_is_therefore_unlabellable(self):
        """Established by execution: the directory has provenance.json only."""
        rec = discover_checkpoint("scwbd-001-beta", root=REAL_RUN1)
        assert rec.manifest_path is None
        claims = rec.admission_claims()
        assert claims.manifest_readable is False
        assert claims.manifest_id == "absent"
        assert claims.weights_trained is True

    @pytest.mark.parametrize(
        "purpose", ["simulation", "research_offline", "live_hardware"]
    )
    def test_it_refuses_only_because_it_cannot_be_labelled(self, purpose):
        """Not because it is a control arm -- that would ship, labelled."""
        with pytest.raises(CheckpointRefused) as exc:
            ServedModel.load(
                "scwbd-001-beta", device="cpu", checkpoint_root=REAL_RUN1,
                purpose=purpose,
            )
        assert exc.value.codes == ("A1",)

    def test_writing_a_truthful_sidecar_is_all_it_takes_to_ship_it(self, tmp_path):
        """The remedy is one file, and then the artifact loads for anything."""
        from scwbd.runtime.admission import sidecar_from_checkpoint

        side = sidecar_from_checkpoint(
            REAL_RUN1 / "scwbd-001-beta" / "last.pt", trust_checkpoint_pickle=True,
        )
        d = tmp_path / "scwbd-001-beta"
        d.mkdir(parents=True)
        (d / "last.pt").symlink_to(REAL_RUN1 / "scwbd-001-beta" / "last.pt")
        (d / "claim_manifest.json").write_text(json.dumps(side))

        served = ServedModel.load(
            "scwbd-001-beta", device="cpu", checkpoint_root=tmp_path,
            purpose="patient_directed",
        )
        assert served.admission.admitted
        assert set(served.admission.label_codes) == {"L1", "L2", "L3"}
        assert served.provenance.notes["admission_banner"]


# ---------------------------------------------------------------------------

class TestTheSidecarHelpers:
    def test_read_sidecar_returns_the_worst_case_defaults_when_absent(self, tmp_path):
        from scwbd.runtime.admission import read_sidecar

        claims = read_sidecar(tmp_path / "nope.json")
        assert claims.manifest_readable is False
        assert claims.manifest_id == "absent"
        assert claims.is_control_arm is True
        assert claims.anatomy_is_biological is False

    def test_read_sidecar_reads_a_present_one(self, tmp_path):
        from scwbd.runtime.admission import read_sidecar

        p = tmp_path / "claim_manifest.json"
        p.write_text(json.dumps(clean_manifest()))
        claims = read_sidecar(p)
        assert claims.manifest_id == "scwbd-002-treatment"
        assert claims.is_control_arm is False
        assert claims.anatomy_is_biological is True

    def test_deriving_a_sidecar_requires_explicitly_trusting_the_pickle(self, tmp_path):
        """The serving path never does this; a person does it once, at emission."""
        from scwbd.runtime.admission import sidecar_from_checkpoint

        with pytest.raises(PermissionError) as exc:
            sidecar_from_checkpoint(tmp_path / "whatever.pt")
        assert "weights_only=False" in str(exc.value)
        assert "executes the pickle" in str(exc.value)


@pytest.mark.skipif(
    not (REAL_RUN1 / "scwbd-001-beta" / "last.pt").is_file(),
    reason="the real run-1 checkpoint is not on this machine",
)
class TestDerivingASidecarFromTheRealArtifact:
    def test_the_derived_sidecar_records_what_the_checkpoint_actually_says(self):
        """Regenerated from the checkpoint, not quoted from a report."""
        from scwbd.runtime.admission import (
            CheckpointClaims,
            admit,
            sidecar_from_checkpoint,
        )

        side = sidecar_from_checkpoint(
            REAL_RUN1 / "scwbd-001-beta" / "last.pt",
            trust_checkpoint_pickle=True,
        )
        assert side["id"] == "SC-WBD-001-beta"
        assert side["anatomy"]["is_biological"] is False
        assert side["anatomy"]["provenance"] == "synthetic_fallback"
        # no gate statuses live in a checkpoint -- under Sec. 7a that is a
        # flagged L3 label, not a blocked load
        assert side["gates"] == {}
        # the checkpoint's own state_layout becomes a readable port contract
        assert side["port_contract_digest"]
        contract = PortContract.from_state_layout(side["state_layout"])
        assert contract.is_uniform
        assert contract.width_of("all_regions") == 28
        assert {p.name for p in contract.exported_ports()} == {
            "rate_e", "rate_i", "spectral"
        }

        # and it ships for the most consequential purpose, fully labelled
        v = admit(CheckpointClaims.from_manifest(side), purpose="patient_directed")
        assert v.admitted
        assert set(v.label_codes) == {"L1", "L2", "L3"}

    def test_a_single_global_local_core_string_is_what_marks_the_control_arm(self):
        """Not truthiness: the config field that decides which arm this is."""
        import torch

        from scwbd.runtime.admission import sidecar_from_checkpoint

        ck = torch.load(
            str(REAL_RUN1 / "scwbd-001-beta" / "last.pt"),
            map_location="cpu", weights_only=False,
        )
        assert isinstance(ck["config"]["model"]["local_core"], str)
        assert ck["config"]["model"]["local_core"] == "learned"
        side = sidecar_from_checkpoint(
            REAL_RUN1 / "scwbd-001-beta" / "last.pt", trust_checkpoint_pickle=True
        )
        assert side["is_control_arm"] is True
        # an explicit declaration overrides the derivation in both directions
        assert sidecar_from_checkpoint(
            REAL_RUN1 / "scwbd-001-beta" / "last.pt",
            trust_checkpoint_pickle=True, is_control_arm=False,
        )["is_control_arm"] is False
