"""A pose is a 6-DoF transform in a named frame, or it is not a pose.

Defect foreclosed: ``thesis_contract.tex`` Sec. 0.5 step 2 -- *"A pose expressed
only as '5 cm anterior' is rejected."*  Every shape of underspecification a
caller can plausibly hand us must reach the same R01 refusal, because the
induced cortical field depends on orientation relative to the local cortical
normal and a scalp label does not determine it (``body.tex`` Sec. 2.8, 7.2).
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from scwbd.runtime import PoseRequest, UndeclaredTransform, UnderspecifiedPose
from scwbd.runtime.types import LedgerIncomplete
from scwbd.transforms.se3 import Pose
from scwbd.transforms.uncertainty import PoseUncertainty


class TestUnderspecifiedPosesAreRejected:
    def test_a_scalar_offset_description_is_not_a_pose(self, service, head):
        with pytest.raises(UnderspecifiedPose) as exc:
            service.evaluate_pose(head, {"description": "5 cm anterior"})
        assert exc.value.code == "R01"
        assert "6-DoF" in str(exc.value)

    def test_an_empty_request_is_not_a_pose(self, service, head):
        with pytest.raises(UnderspecifiedPose):
            service.evaluate_pose(head, {})

    def test_a_scalp_label_alone_is_not_a_pose(self, service, head):
        with pytest.raises(UnderspecifiedPose):
            service.evaluate_pose(head, PoseRequest(description="F3", label="F3"))

    def test_a_frame_declaration_that_contradicts_the_transform_is_rejected(
        self, service, head, nominal_pose
    ):
        wrong = replace(nominal_pose, frame="some_other_frame")
        with pytest.raises(UnderspecifiedPose) as exc:
            service.evaluate_pose(head, wrong)
        assert "will not guess" in str(exc.value)

    def test_an_epoch_declaration_that_contradicts_the_transform_is_rejected(
        self, head, nominal_pose
    ):
        pose = Pose(
            nominal_pose.pose.matrix,
            head.frame,
            "coil",
            epoch="ses-01",
            provenance={"method": "test"},
        )
        with pytest.raises(UnderspecifiedPose):
            replace(nominal_pose, pose=pose, epoch="ses-02").resolve()

    def test_unknown_keys_are_not_silently_ignored(self, service, head, nominal_pose):
        with pytest.raises(UnderspecifiedPose) as exc:
            service.evaluate_pose(
                head,
                {"pose": nominal_pose.pose, "five_cm_anterior": True},
            )
        assert "unrecognised keys" in str(exc.value)


class TestUndeclaredTransformsAreRefusedNotAssumed:
    def test_a_foreign_frame_with_no_declared_route_refuses(self, service, head):
        pose = Pose.from_Rt(
            torch.eye(3, dtype=torch.float64),
            [0.0, 0.0, 0.10],
            "consumer_head_ALS",
            "coil",
            provenance={"method": "declared_by_consumer"},
        )
        with pytest.raises(UndeclaredTransform) as exc:
            service.evaluate_pose(
                head,
                PoseRequest(
                    pose=pose,
                    frame="consumer_head_ALS",
                    uncertainty=PoseUncertainty.isotropic(0.001, 0.01),
                ),
            )
        assert exc.value.code == "R01"
        assert "will not insert an identity" in str(
            exc.value
        ) or "appears in no declared edge" in str(exc.value)

    def test_a_declared_route_is_used_and_reported(
        self, service, head, head_with_chain
    ):
        pose = Pose.from_Rt(
            torch.eye(3, dtype=torch.float64),
            [0.0, 0.0, 0.10],
            "consumer_head_ALS",
            "coil",
            provenance={"method": "declared_by_consumer"},
        )
        evaluation = service.evaluate_pose(
            head_with_chain,
            PoseRequest(
                pose=pose,
                frame="consumer_head_ALS",
                label="via_declared_chain",
                uncertainty=PoseUncertainty.isotropic(0.001, 0.01),
            ),
        )
        assert evaluation.requested_frame == "consumer_head_ALS"
        assert evaluation.pose.parent == head.frame
        # the declared edge, its provenance and its method are all reported
        assert len(evaluation.transform_chain) == 1
        assert "measured:landmark_axes_fit_with_residual" in evaluation.transform_chain[0]

    def test_the_evaluation_always_names_the_chain_it_used(
        self, service, head, nominal_pose
    ):
        evaluation = service.evaluate_pose(head, nominal_pose)
        assert evaluation.transform_chain
        assert head.frame in evaluation.transform_chain[0]


class TestPoseUncertaintyMustBeDeclared:
    def test_a_pose_with_no_declared_uncertainty_refuses(self, service, head, nominal_pose):
        naked = replace(nominal_pose, uncertainty=None)
        with pytest.raises(LedgerIncomplete) as exc:
            service.evaluate_pose(head, naked)
        assert exc.value.code == "R08"
        assert "rather than letting it default to zero" in str(exc.value)

    def test_a_zero_covariance_pose_refuses(self, service, head, nominal_pose):
        zero = replace(
            nominal_pose,
            uncertainty=PoseUncertainty(cov=torch.zeros(6, 6, dtype=torch.float64)),
        )
        with pytest.raises(LedgerIncomplete) as exc:
            service.evaluate_pose(head, zero)
        assert "known exactly is a claim" in str(exc.value)
