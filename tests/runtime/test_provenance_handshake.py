"""A consumer must be able to assert what it is talking to, and fail if wrong.

Defect foreclosed: an analytic fallback or a randomly-initialised network being
consumed as if it were the trained ``SC-WBD-001-beta`` artifact.  ``tms-robotics``
keeps evidence classes strictly apart; handing it an unlabelled prediction would
break that discipline from outside its own repository.
"""

from __future__ import annotations

import json

import pytest

from scwbd.runtime import (
    MODEL_DESIGNATION,
    RUNTIME_API_VERSION,
    SCHEMA_VERSION,
    ModelProvenance,
    ProvenanceExpectation,
    ProvenanceMismatch,
    ServedModel,
    discover_checkpoint,
)


class TestTheHandshakeCatchesMismatches:
    def test_a_matching_expectation_passes(self, served):
        expectation = ProvenanceExpectation(
            model_designation=MODEL_DESIGNATION,
            schema_version=SCHEMA_VERSION,
            runtime_api_version=RUNTIME_API_VERSION,
        )
        assert served.handshake(expectation) is served.provenance

    def test_a_wrong_schema_version_is_caught(self, served):
        with pytest.raises(ProvenanceMismatch) as exc:
            served.handshake(ProvenanceExpectation(schema_version="scwbd-schema/9.9.9"))
        assert any("schema_version" in m for m in exc.value.mismatches)

    def test_a_wrong_runtime_api_version_is_caught(self, served):
        with pytest.raises(ProvenanceMismatch) as exc:
            served.handshake(
                ProvenanceExpectation(runtime_api_version="scwbd-runtime/0.1.0")
            )
        assert any("runtime_api_version" in m for m in exc.value.mismatches)

    def test_a_wrong_model_designation_is_caught(self, served):
        with pytest.raises(ProvenanceMismatch):
            served.handshake(ProvenanceExpectation(model_designation="SC-WBD-002"))

    def test_demanding_a_trained_artifact_fails_on_the_analytic_fallback(self, served):
        """The single most important assertion a consumer can make."""
        with pytest.raises(ProvenanceMismatch) as exc:
            served.handshake(
                ProvenanceExpectation(accept_weights_status=("trained",))
            )
        assert any("not a trained SC-WBD-001-beta checkpoint" in m for m in exc.value.mismatches)
        assert exc.value.served.weights_status == "analytic_backend"

    def test_demanding_a_numerical_field_solver_fails_on_the_analytic_one(self, served):
        if served.provenance.efield_backend_class != "analytic":
            pytest.skip("agent G's numerical solver has landed")
        with pytest.raises(ProvenanceMismatch) as exc:
            served.handshake(
                ProvenanceExpectation(accept_efield_backend_class=("numerical_fem",))
            )
        assert any("efield_backend_class" in m for m in exc.value.mismatches)

    def test_a_wrong_checkpoint_hash_is_caught(self, served):
        with pytest.raises(ProvenanceMismatch):
            served.handshake(ProvenanceExpectation(checkpoint_sha256="0" * 64))

    def test_every_mismatch_is_reported_not_just_the_first(self, served):
        with pytest.raises(ProvenanceMismatch) as exc:
            served.handshake(
                ProvenanceExpectation(
                    model_designation="wrong",
                    schema_version="wrong",
                    runtime_api_version="wrong",
                )
            )
        assert len(exc.value.mismatches) == 3

    def test_a_consumer_can_require_enough_candidate_models(self, served):
        served.handshake(ProvenanceExpectation(min_response_models=2))
        with pytest.raises(ProvenanceMismatch):
            served.handshake(ProvenanceExpectation(min_response_models=99))

    def test_a_consumer_can_require_cited_safety_limits(self, served):
        served.handshake(ProvenanceExpectation(require_a_safe_citations=True))
        assert served.provenance.a_safe_citations


class TestProvenanceTellsTheTruthAboutWhatWasLoaded:
    def test_no_checkpoint_is_reported_as_an_analytic_backend(self, served):
        assert served.provenance.weights_status == "analytic_backend"
        assert served.provenance.is_trained_artifact is False
        assert served.provenance.checkpoint_sha256 is None
        assert "must not be reported as outputs of a trained whole-brain model" in (
            served.provenance.notes["untrained_warning"]
        )

    def test_the_claim_class_of_an_untrained_service_is_not_mechanistic(self, served):
        assert served.provenance.claim_class in ("surrogate", "provenance_only")
        assert served.provenance.posterior_class == "pseudo"

    def test_provenance_is_hashable_and_stable(self, served):
        first = served.provenance.content_hash()
        assert first == served.provenance.content_hash()
        assert json.loads(json.dumps(served.provenance.canonical()))

    def test_provenance_refuses_to_claim_human_authorization(self):
        with pytest.raises(ValueError):
            ModelProvenance(human_use_authorized=True)
        with pytest.raises(ValueError):
            ModelProvenance(prospective_human=True)

    def test_the_evaluation_carries_the_same_provenance_object(
        self, served, head, nominal_pose
    ):
        evaluation = served.targeting.evaluate_pose(head, nominal_pose)
        assert evaluation.provenance.content_hash() == served.provenance.content_hash()


class TestCheckpointDiscovery:
    def test_a_missing_checkpoint_is_not_an_error_by_default(self, tmp_path):
        record = discover_checkpoint("scwbd-001-beta", root=tmp_path)
        assert record.found is False
        assert record.weights_status == "analytic_backend"

    def test_a_caller_can_demand_a_checkpoint_and_get_a_hard_failure(self, tmp_path):
        from scwbd.runtime import CheckpointNotFound

        with pytest.raises(CheckpointNotFound):
            discover_checkpoint("scwbd-001-beta", root=tmp_path, require=True)

    def test_a_found_checkpoint_is_hashed_and_its_manifest_read(self, tmp_path):
        base = tmp_path / "scwbd-001-beta"
        base.mkdir()
        (base / "weights.pt").write_bytes(b"not real weights, but hashed")
        (base / "claim_manifest.json").write_text(
            json.dumps({"id": "beta_claim", "claim_class": "functional"})
        )
        record = discover_checkpoint("scwbd-001-beta", root=tmp_path)
        assert record.found and record.weights_status == "trained"
        assert record.weights_sha256 and len(record.weights_sha256) == 64
        assert record.claim_fields()["claim_manifest_id"] == "beta_claim"

        served = ServedModel.load(device="cpu", checkpoint_root=tmp_path)
        assert served.provenance.weights_status == "trained"
        served.handshake(
            ProvenanceExpectation(
                accept_weights_status=("trained",),
                checkpoint_sha256=record.weights_sha256,
            )
        )


class TestWarmUpAndBatching:
    def test_warm_up_returns_the_evaluation_it_ran(self, served):
        evaluation = served.warm_up()
        assert evaluation.label == "warm_up"
        assert evaluation.transform_chain
        assert evaluation.ledger.variance

    def test_warm_up_is_deterministic(self, served):
        a = served.warm_up()
        b = served.warm_up()
        assert a.field_accuracy.peak_v_per_m == b.field_accuracy.peak_v_per_m
        assert a.target_engagement.mean() == b.target_engagement.mean()

    def test_each_pose_in_a_batch_gets_its_own_decision(
        self, served, head, nominal_pose
    ):
        from dataclasses import replace

        from scwbd.transforms.uncertainty import PoseUncertainty

        poses = [
            replace(nominal_pose, label="tight", uncertainty=PoseUncertainty.isotropic(5e-4, 5e-3)),
            replace(nominal_pose, label="loose", uncertainty=PoseUncertainty.isotropic(2e-2, 3.5e-1)),
        ]
        results = served.evaluate_batch(head, poses)
        assert [r.label for r in results] == ["tight", "loose"]
        assert {type(r.decision).__name__ for r in results} != {"Recommend"}
