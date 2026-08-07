"""What still refuses when the authorization is completely valid.

Governance being unblocked is not capability being established.  These tests
hold a fully valid declaration constant and show that every substantive
refusal path in the runtime still fires:

* **no trained model.**  ``weights_status="analytic_backend"``, so a
  protocol-bound targeting claim is refused ``R11`` naming the missing
  checkpoint.  This release has no trained SC-WBD-001-beta artifact, and an
  approval does not supply one.
* **uncertainty dominating the benefit difference** -> ``Defer`` with the next
  measurement named (``thesis_contract.tex`` Sec. 0.5 step 6).
* **outside the validated field-solver envelope** -> ``Defer``; the solver's
  own refusal, not a number without an error bound.
* **underspecified pose, undeclared frame, undeclared pose uncertainty** ->
  raised refusals, exactly as without a record.
* **outside** :math:`\\mathcal A_{\\rm safe}` -> ``Refuse(code="R11")`` naming
  the axis.  The negative control.

Nothing in this file authorises anything; the record is the fictional fixture
from ``tests/conftest.py``.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from scwbd.runtime import (
    Defer,
    PoseRequest,
    Recommend,
    Refuse,
    TargetingService,
    UndeclaredTransform,
    UnderspecifiedPose,
    spherical_phantom,
)
from scwbd.runtime.backends import ChargeBEMEField, MagnitudeThresholdResponse
from scwbd.runtime.serving import coil_pose_over_region
from scwbd.runtime.types import LedgerIncomplete
from scwbd.schema.authorization import epoch_seconds
from scwbd.schema.refusals import CompilerRefusal as SchemaRefusal
from scwbd.transforms.se3 import Pose
from scwbd.transforms.uncertainty import PoseUncertainty

_DT = torch.float64
WITHIN_WINDOW_S = epoch_seconds("2026-08-05")
AFTER_WINDOW_S = epoch_seconds("2027-06-01")


@pytest.fixture
def authorized_service(authorization) -> TargetingService:
    """A targeting service constructed under the valid fixture declaration."""
    return TargetingService(
        authorization=authorization, request_time_s=WITHIN_WINDOW_S
    )


def _displaced(head, base, radial_offset_m: float, label: str):
    direction = base.pose.t - head.centre
    direction = direction / torch.linalg.norm(direction)
    t = head.centre + direction * (head.scalp_radius + radial_offset_m)
    pose = Pose.from_Rt(
        base.pose.R, t, head.frame, "coil", provenance={"method": "radial_offset"}
    )
    return replace(base, pose=pose, label=label)


# ---------------------------------------------------------------------------
# the record is recorded, and it changes what is served
# ---------------------------------------------------------------------------


class TestTheServiceRecordsTheAuthorization:
    def test_the_provenance_names_the_protocol_and_pins_the_record(
        self, authorized_service, authorization
    ):
        prov = authorized_service.provenance
        assert prov.claim_scope == "protocol:EX-TMS-DLPFC-01@3.2"
        assert prov.is_protocol_bound
        assert prov.authorization_hash == authorization.content_hash()
        assert "protocol:EX-TMS-DLPFC-01@3.2" in prov.label

    def test_a_service_with_no_record_is_simulation_only(self, service):
        assert service.provenance.claim_scope == "simulation_only"
        assert not service.is_protocol_bound
        assert service.provenance.authorization is None

    def test_the_two_services_are_distinguishable_downstream(
        self, service, authorized_service
    ):
        assert service.provenance.content_hash() != (
            authorized_service.provenance.content_hash()
        )

    def test_a_consumer_can_assert_the_scope_it_expects(
        self, service, authorized_service
    ):
        from scwbd.runtime.provenance import ProvenanceExpectation, ProvenanceMismatch

        simulation_only = ProvenanceExpectation(require_claim_scope="simulation_only")
        simulation_only.check(service.provenance)
        with pytest.raises(ProvenanceMismatch, match="claim_scope"):
            simulation_only.check(authorized_service.provenance)

    def test_the_software_still_does_not_authorise_human_use(self, authorized_service):
        """Recording somebody else's approval is not issuing one."""
        assert authorized_service.provenance.human_use_authorized is False
        assert authorized_service.provenance.prospective_human is False

    def test_an_evaluation_carries_the_scope(
        self, authorized_service, head, nominal_pose
    ):
        evaluation = authorized_service.evaluate_pose(head, nominal_pose)
        assert evaluation.provenance.claim_scope == "protocol:EX-TMS-DLPFC-01@3.2"
        assert "DECLARATION, NOT VERIFICATION" in (
            evaluation.provenance.authorization["notice"]
        )


class TestAnInvalidRecordCannotEvenConstructTheService:
    def test_an_expired_record_refuses_at_construction(self, authorization):
        with pytest.raises(SchemaRefusal) as excinfo:
            TargetingService(authorization=authorization, request_time_s=AFTER_WINDOW_S)
        codes = [f["code"] for f in excinfo.value.evidence["authorization_failures"]]
        assert codes == ["AUTH_EXPIRED"]

    def test_an_undated_request_refuses_at_construction(self, authorization):
        with pytest.raises(SchemaRefusal) as excinfo:
            TargetingService(authorization=authorization, request_time_s=None)
        codes = [f["code"] for f in excinfo.value.evidence["authorization_failures"]]
        assert "AUTH_TIME_UNDECLARED" in codes

    def test_a_protocol_that_does_not_bound_the_checked_axes_refuses(
        self, make_authorization, authorization
    ):
        thin = make_authorization(
            a_safe=authorization.a_safe.model_copy(
                update={"constraint_axes": ("tms.peak_efield_v_per_m",)}
            )
        )
        with pytest.raises(SchemaRefusal) as excinfo:
            TargetingService(authorization=thin, request_time_s=WITHIN_WINDOW_S)
        assert "AUTH_ASAFE_NOT_TRACEABLE" in [
            f["code"] for f in excinfo.value.evidence["authorization_failures"]
        ]


# ---------------------------------------------------------------------------
# what still refuses under a valid authorization
# ---------------------------------------------------------------------------


class TestNoTrainedModelRefusesTheTargetingClaim:
    def test_a_confident_pose_refuses_for_want_of_a_checkpoint(
        self, service, authorized_service, head, nominal_pose
    ):
        """The same pose that a simulation-only service would recommend."""
        unauthorized = service.evaluate_pose(head, nominal_pose)
        assert isinstance(unauthorized.decision, Recommend), (
            "fixture must be a case that would otherwise recommend"
        )

        decision = authorized_service.evaluate_pose(head, nominal_pose).decision
        assert isinstance(decision, Refuse)
        assert decision.code == "R11"
        assert "weights_status=analytic_backend" in decision.violations
        assert "does not supply a model" in decision.reason

    def test_the_refusal_names_the_protocol_it_was_requested_under(
        self, authorized_service, head, nominal_pose
    ):
        decision = authorized_service.evaluate_pose(head, nominal_pose).decision
        assert "EX-TMS-DLPFC-01@3.2" in decision.reason

    def test_the_remedy_is_a_checkpoint_not_a_looser_gate(
        self, authorized_service, head, nominal_pose
    ):
        decision = authorized_service.evaluate_pose(head, nominal_pose).decision
        assert "train and validate a checkpoint" in decision.remedy


class TestUncertaintyStillDefersUnderAuthorization:
    def test_transform_uncertainty_dominating_defers_to_calibration(
        self, authorized_service, head, nominal_pose
    ):
        decision = authorized_service.evaluate_pose(
            head,
            replace(
                nominal_pose,
                uncertainty=PoseUncertainty.isotropic(0.020, 0.35),
                label="high_transform_uncertainty",
            ),
        ).decision
        assert isinstance(decision, Defer)
        assert decision.suggested_action == "additional_calibration_measurement"
        assert decision.detail["transform_sd"] > decision.detail["model_disagreement"]

    def test_a_single_response_model_still_defers(self, authorization, head, nominal_pose):
        lone = TargetingService(
            response_operators=(MagnitudeThresholdResponse(),),
            authorization=authorization,
            request_time_s=WITHIN_WINDOW_S,
        )
        decision = lone.evaluate_pose(head, nominal_pose).decision
        assert isinstance(decision, Defer)
        assert "single response model" in decision.reason


class TestOutsideTheSolverEnvelopeStillDefers:
    def test_an_unresolved_field_defers_and_returns_no_number(self, authorization):
        """Faraday's ``assert_resolves_sources`` refusal, under a valid approval."""
        small_head = spherical_phantom(
            subject_id="phantom-bem-auth", n_vertices=120, n_parcels=12
        )
        pose = PoseRequest(
            pose=coil_pose_over_region(small_head, method="bem_auth_fixture"),
            frame=small_head.frame,
            label="coarse_bem",
            uncertainty=PoseUncertainty.isotropic(1e-3, 1e-2),
        )
        evaluation = TargetingService(
            efield_backend=ChargeBEMEField(uniform_subdiv=1),
            authorization=authorization,
            request_time_s=WITHIN_WINDOW_S,
        ).evaluate_pose(small_head, pose)

        assert isinstance(evaluation.decision, Defer)
        assert evaluation.decision.suggested_action == "no_action"
        assert evaluation.unresolved_quantities()
        assert evaluation.provenance.is_protocol_bound


class TestASafeStillRefusesUnderAuthorization:
    def test_a_pose_outside_a_safe_refuses_r11_naming_the_axis(
        self, authorized_service, head, nominal_pose
    ):
        """The negative control at the runtime level."""
        far = _displaced(head, nominal_pose, 0.090, "far_from_scalp")
        decision = authorized_service.evaluate_pose(head, far).decision
        assert isinstance(decision, Refuse)
        assert decision.code == "R11"
        assert any("coil_scalp_distance_mm" in v for v in decision.violations)
        # the A_safe refusal wins over the missing-checkpoint refusal: it is
        # the more specific statement about this particular pose
        assert "weights_status" not in decision.reason

    def test_the_verdict_is_identical_to_the_unauthorized_one(
        self, service, authorized_service, head, nominal_pose
    ):
        far = _displaced(head, nominal_pose, 0.090, "far_from_scalp")
        plain = service.evaluate_pose(head, far).decision
        gated = authorized_service.evaluate_pose(head, far).decision
        assert plain.code == gated.code == "R11"
        assert plain.violations == gated.violations

    def test_an_out_of_envelope_protocol_still_refuses(self, authorization, head, nominal_pose):
        from scwbd.runtime.targeting import SessionProtocol

        service = TargetingService(
            authorization=authorization, request_time_s=WITHIN_WINDOW_S
        ).with_config(protocol=SessionProtocol(frequency_hz=120.0))
        decision = service.evaluate_pose(head, nominal_pose).decision
        assert isinstance(decision, Refuse)
        assert decision.code == "R11"
        assert any("frequency_hz" in v for v in decision.violations)


class TestPoseSpecificationStillRefusesUnderAuthorization:
    def test_a_scalar_offset_description_is_still_not_a_pose(
        self, authorized_service, head
    ):
        with pytest.raises(UnderspecifiedPose) as excinfo:
            authorized_service.evaluate_pose(head, {"description": "5 cm anterior"})
        assert excinfo.value.code == "R01"

    def test_an_undeclared_frame_is_still_refused_not_assumed(
        self, authorized_service, head
    ):
        pose = Pose.from_Rt(
            torch.eye(3, dtype=_DT),
            [0.0, 0.0, 0.10],
            "consumer_head_ALS",
            "coil",
            provenance={"method": "declared_by_consumer"},
        )
        with pytest.raises(UndeclaredTransform):
            authorized_service.evaluate_pose(
                head,
                PoseRequest(
                    pose=pose,
                    frame="consumer_head_ALS",
                    uncertainty=PoseUncertainty.isotropic(0.001, 0.01),
                ),
            )

    def test_an_undeclared_pose_uncertainty_is_still_refused(
        self, authorized_service, head, nominal_pose
    ):
        with pytest.raises(LedgerIncomplete):
            authorized_service.evaluate_pose(
                head, replace(nominal_pose, uncertainty=None)
            )

    def test_an_expired_calibration_is_still_refused(self):
        """Calibration validity is upstream of governance and stays that way.

        ``scwbd.transforms.calibration`` never sees an AuthorizationRecord, so
        an approval cannot extend a calibration -- which is the point: an
        interval cannot vouch for a device outside it, whoever approved the
        study.
        """
        from scwbd.transforms.calibration import CalibrationRecord, ExpiryPolicy
        from scwbd.transforms.errors import CalibrationExpiredError
        from scwbd.transforms.se3 import ValidityInterval

        record = CalibrationRecord(
            method="fiducial_lsq",
            validity=ValidityInterval(0.0, 3600.0),
        )
        record.check(1800.0, policy=ExpiryPolicy.REFUSE)
        with pytest.raises(CalibrationExpiredError):
            record.check(7200.0, policy=ExpiryPolicy.REFUSE)


# ---------------------------------------------------------------------------
# the surface stays a read/refuse surface
# ---------------------------------------------------------------------------


def test_authorization_creates_no_command_path(authorized_service):
    forbidden = ("command", "actuate", "trajectory", "execute", "deliver", "dose")
    names = [n for n in dir(authorized_service) if not n.startswith("_")]
    assert not [n for n in names if any(t in n.lower() for t in forbidden)]
    assert authorized_service.read("stimulator_output_percent").missing == (
        "stimulator_output_percent",
    )
