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

from scwbd.intervene.deployment import (
    PRELIMINARY_REVIEW_SCHEDULED,
    PreliminaryReviewRecord,
)
from scwbd.runtime.admission import CheckpointRefused
from scwbd.schema.authorization import epoch_seconds
from scwbd.runtime.provenance import ProvenanceExpectation, ProvenanceMismatch
from scwbd.runtime.serving import ServedModel, discover_checkpoint

from test_ports import RUN1_LAYOUT, RUN2_LAYOUT

REVIEW_DAY = PRELIMINARY_REVIEW_SCHEDULED.isoformat()
AS_OF = epoch_seconds(REVIEW_DAY) + 86400.0
#: Years past the scheduled review. Used to show the date is not an unlock.
LONG_AFTER_S = epoch_seconds("2027-12-31")

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


def approving() -> PreliminaryReviewRecord:
    """A *fictional* record that the review happened and passed."""
    return PreliminaryReviewRecord(
        review_body="Example University preliminary review panel",
        identifier="PRELIM-2026-0001",
        occurred_on=REVIEW_DAY,
        outcome="approved",
        covered_intervention_classes=("tms",),
        declared_by="test fixture",
    )


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

class TestLoadRefusesBeforeAServiceExists:
    def test_the_control_arm_is_refused_for_live_hardware(self, tmp_path, authorization):
        write_checkpoint(tmp_path, "scwbd-001-beta", run1_manifest())
        with pytest.raises(CheckpointRefused) as exc:
            ServedModel.load(
                "scwbd-001-beta",
                device="cpu",
                checkpoint_root=tmp_path,
                purpose="live_hardware",
                review=approving(),
            authorization=authorization,
                as_of=AS_OF,
            )
        assert set(exc.value.codes) == {"A2", "A3", "A4"}
        text = str(exc.value)
        assert "control arm" in text
        assert "synthetic_fallback" in text
        assert "COULD_NOT_RUN" in text

    def test_a_checkpoint_with_no_manifest_is_refused_for_research(self, tmp_path):
        write_checkpoint(tmp_path, "bare", None)
        with pytest.raises(CheckpointRefused) as exc:
            ServedModel.load(
                "bare", device="cpu", checkpoint_root=tmp_path,
                purpose="research_offline", as_of=AS_OF,
            )
        assert exc.value.codes == ("A1",)

    def test_live_use_is_refused_with_no_authorization_record(self, tmp_path):
        write_checkpoint(tmp_path, "clean", clean_manifest())
        with pytest.raises(CheckpointRefused) as exc:
            ServedModel.load(
                "clean", device="cpu", checkpoint_root=tmp_path,
                purpose="live_hardware", as_of=AS_OF,
            )
        assert exc.value.codes == ("A6",)

    def test_waiting_does_not_open_the_gate(self, tmp_path):
        """Ten years after the review date, with no record: still refused."""
        write_checkpoint(tmp_path, "clean", clean_manifest())
        with pytest.raises(CheckpointRefused) as exc:
            ServedModel.load(
                "clean", device="cpu", checkpoint_root=tmp_path,
                purpose="patient_directed",
                as_of=LONG_AFTER_S,
            )
        assert exc.value.codes == ("A6",)

    def test_simulation_still_loads_the_control_arm(self, tmp_path):
        """The gate governs purpose, not existence. Simulation is not gated."""
        write_checkpoint(tmp_path, "scwbd-001-beta", run1_manifest())
        served = ServedModel.load(
            "scwbd-001-beta", device="cpu", checkpoint_root=tmp_path,
            purpose="simulation", as_of=AS_OF,
        )
        assert served.admission.admitted
        assert served.admission.purpose == "simulation"

    def test_a_clean_checkpoint_with_a_record_loads_for_live_hardware(self, tmp_path, authorization):
        """Paired with every refusal above: the gate is not stuck closed."""
        write_checkpoint(tmp_path, "clean", clean_manifest())
        served = ServedModel.load(
            "clean", device="cpu", checkpoint_root=tmp_path,
            purpose="live_hardware",
            review=approving(),
            authorization=authorization,
            as_of=AS_OF,
        )
        assert served.admission.admitted
        assert served.provenance.notes["export_purpose"] == "live_hardware"
        assert served.provenance.notes["admission_hash"]
        assert served.provenance.notes["consumer_standing_invariants"] == {
            "sim2real_ready": False,
            "promotion_eligible": False,
            "robot_command_authority": False,
        }

    def test_the_refusal_happens_before_any_service_object_exists(self, tmp_path, authorization):
        """There is no window in which an inadmissible checkpoint is usable."""
        write_checkpoint(tmp_path, "scwbd-001-beta", run1_manifest())
        served = None
        try:
            served = ServedModel.load(
                "scwbd-001-beta", device="cpu", checkpoint_root=tmp_path,
                purpose="patient_directed", review=approving(),
                authorization=authorization,
                as_of=AS_OF,
            )
        except CheckpointRefused:
            pass
        assert served is None


# ---------------------------------------------------------------------------
# the port contract travels with the service
# ---------------------------------------------------------------------------

class TestThePortContractIsServedAndPinnable:
    def test_the_declared_ports_reach_the_consumer(self, tmp_path):
        write_checkpoint(tmp_path, "clean", clean_manifest())
        served = ServedModel.load(
            "clean", device="cpu", checkpoint_root=tmp_path,
            purpose="research_offline", as_of=AS_OF,
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
                               purpose="research_offline", as_of=AS_OF)
        one = ServedModel.load("one", device="cpu", checkpoint_root=tmp_path,
                               purpose="simulation", as_of=AS_OF)

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
                                  purpose="research_offline", as_of=AS_OF)
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

    def test_it_ships_no_claim_manifest_and_is_therefore_refused(self):
        """Established by execution, 2026-08-06: the directory has
        ``provenance.json``, not ``claim_manifest.json``."""
        rec = discover_checkpoint("scwbd-001-beta", root=REAL_RUN1)
        assert rec.manifest_path is None
        claims = rec.admission_claims()
        assert claims.manifest_id == "absent"
        # trained weights, but nothing that states what they are
        assert claims.weights_trained is True
        assert claims.is_control_arm is True
        assert claims.anatomy_is_biological is False

    @pytest.mark.parametrize(
        "purpose", ["research_offline", "live_hardware", "patient_directed"]
    )
    def test_no_purpose_beyond_simulation_admits_it(self, purpose, authorization):
        with pytest.raises(CheckpointRefused) as exc:
            ServedModel.load(
                "scwbd-001-beta", device="cpu", checkpoint_root=REAL_RUN1,
                purpose=purpose, review=approving(),
                authorization=authorization, as_of=AS_OF,
            )
        assert "A1" in exc.value.codes

    def test_the_g5_control_checkpoint_is_refused_too(self):
        with pytest.raises(CheckpointRefused):
            ServedModel.load(
                "scwbd-001-beta-g5control", device="cpu", checkpoint_root=REAL_RUN1,
                purpose="research_offline", as_of=AS_OF,
            )


# ---------------------------------------------------------------------------
# the unconditional refusal that could not discriminate
# ---------------------------------------------------------------------------

class TestProspectiveHumanIsGatedNotHardcoded:
    """``ModelProvenance.prospective_human`` used to raise regardless of any
    record, with the reason "out of scope for SC-WBD-001-beta (build-order item
    6)".  Both of its guards carried ``# pragma: no cover`` and neither was
    fired by any test.  An unconditional refusal cannot discriminate, and it
    made the authorization mechanism unreachable from the consumer -- the same
    defect as a stale permission string, pointed the safe way.

    It now routes through the one live-application gate, so it refuses with a
    reason that a record can satisfy.
    """

    def _verdict(self, authorization, **over):
        from scwbd.intervene.deployment import authorize_live_application

        kw = dict(
            mode="live", intervention_class="tms", at_time_s=AS_OF,
            review=approving(), authorization=authorization,
        )
        kw.update(over)
        return authorize_live_application(**kw).as_provenance()

    def test_it_refuses_with_no_verdict_at_all(self):
        from scwbd.runtime.provenance import ModelProvenance

        with pytest.raises(ValueError) as exc:
            ModelProvenance(prospective_human=True)
        assert "no live-application verdict was recorded" in str(exc.value)
        assert "This refusal is satisfiable" in str(exc.value)

    def test_it_refuses_on_a_refusing_verdict_and_names_the_failure(self, authorization):
        from scwbd.runtime.provenance import ModelProvenance

        with pytest.raises(ValueError) as exc:
            ModelProvenance(
                prospective_human=True,
                live_application=self._verdict(authorization, review=None),
            )
        assert "REVIEW_ABSENT" in str(exc.value)

    def test_a_computational_verdict_does_not_admit_a_prospective_human_claim(
        self, authorization
    ):
        from scwbd.runtime.provenance import ModelProvenance

        v = self._verdict(authorization, mode="computational")
        assert v["admitted"] is True  # it *is* admitted, as computational
        with pytest.raises(ValueError) as exc:
            ModelProvenance(prospective_human=True, live_application=v)
        assert "application_mode='computational'" in str(exc.value)

    def test_an_admitting_live_verdict_permits_it(self, authorization):
        """The pair that proves the guard discriminates rather than always fires."""
        from scwbd.runtime.provenance import ModelProvenance

        prov = ModelProvenance(
            prospective_human=True,
            live_application=self._verdict(authorization),
        )
        assert prov.prospective_human is True
        assert prov.live_application["admitted"] is True

    def test_human_use_authorized_stays_unconditional_and_now_fires(self):
        """The asymmetry is deliberate: this one no record can satisfy."""
        from scwbd.runtime.provenance import ModelProvenance

        with pytest.raises(ValueError) as exc:
            ModelProvenance(human_use_authorized=True)
        assert "this software does not issue authorization" in str(exc.value)
