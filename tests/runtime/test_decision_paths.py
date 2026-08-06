"""Defer and Refuse are the paths that matter; they are tested first.

Defects foreclosed:

* ``thesis_contract.tex`` Sec. 0.5 step 6 -- an optimizer that answers with a
  recommendation when model disagreement or transform uncertainty dominates the
  estimated benefit difference.  Here that is a real branch with a real
  threshold read from ``limits/a_safe.toml``, and this module *constructs* the
  case that forces it.
* Refusal R11 -- intervention optimization outside an independently validated
  feasible set.  A pose that leaves :math:`\\mathcal A_{\\rm safe}` gets a
  ``Refuse(code="R11")`` naming the axis, and is never repaired by projection.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from scwbd.runtime import Defer, Recommend, Refuse, TargetingService
from scwbd.runtime.backends import MagnitudeThresholdResponse
from scwbd.transforms.se3 import Pose
from scwbd.transforms.uncertainty import PoseUncertainty

_DT = torch.float64


def _displaced(head, base, radial_offset_m: float, label: str):
    """Move the coil radially away from the fitted scalp sphere."""
    direction = base.pose.t - head.centre
    direction = direction / torch.linalg.norm(direction)
    t = head.centre + direction * (head.scalp_radius + radial_offset_m)
    pose = Pose.from_Rt(
        base.pose.R, t, head.frame, "coil", provenance={"method": "radial_offset"}
    )
    return replace(base, pose=pose, label=label)


class TestDeferWhenUncertaintyDominates:
    def test_high_transform_uncertainty_forces_defer(self, service, head, nominal_pose):
        """The constructed case: inflate the declared pose covariance only."""
        confident = service.evaluate_pose(
            head, replace(nominal_pose, uncertainty=PoseUncertainty.isotropic(0.0005, 0.005))
        )
        uncertain = service.evaluate_pose(
            head,
            replace(
                nominal_pose,
                uncertainty=PoseUncertainty.isotropic(0.020, 0.35),
                label="high_transform_uncertainty",
            ),
        )
        # the *prediction* is unchanged; only what we are entitled to say is
        assert confident.field_accuracy.peak_v_per_m == pytest.approx(
            uncertain.field_accuracy.peak_v_per_m
        )
        assert isinstance(uncertain.decision, Defer)

    def test_the_defer_names_the_next_measurement(self, service, head, nominal_pose):
        evaluation = service.evaluate_pose(
            head,
            replace(nominal_pose, uncertainty=PoseUncertainty.isotropic(0.020, 0.35)),
        )
        decision = evaluation.decision
        assert isinstance(decision, Defer)
        assert decision.suggested_action in (
            "additional_calibration_measurement",
            "reversible_probe",
        )
        # transform uncertainty dominates here, so the answer is calibration
        assert decision.suggested_action == "additional_calibration_measurement"
        assert decision.detail["transform_sd"] > decision.detail["model_disagreement"]

    def test_the_defer_reports_both_competing_terms(self, service, head, nominal_pose):
        decision = service.evaluate_pose(
            head,
            replace(nominal_pose, uncertainty=PoseUncertainty.isotropic(0.020, 0.35)),
        ).decision
        for key in ("benefit_margin", "epistemic", "transform_sd", "model_disagreement"):
            assert key in decision.detail

    def test_a_single_response_model_always_defers(self, head, nominal_pose):
        """The mechanism is unresolved; one operator is never a comparison."""
        lone = TargetingService(response_operators=(MagnitudeThresholdResponse(),))
        decision = lone.evaluate_pose(head, nominal_pose).decision
        assert isinstance(decision, Defer)
        assert "single response model" in decision.reason
        assert decision.suggested_action == "additional_calibration_measurement"

    def test_model_disagreement_alone_can_also_defer(self, head, nominal_pose):
        """A tiny pose error but wildly disagreeing operators still defers."""
        from scwbd.runtime.backends import (
            NormalComponentResponse,
            TangentialDirectionResponse,
        )

        service = TargetingService(
            response_operators=(
                MagnitudeThresholdResponse(threshold_v_per_m=10.0, width_v_per_m=5.0),
                NormalComponentResponse(threshold_v_per_m=90.0, width_v_per_m=2.0),
                TangentialDirectionResponse(threshold_v_per_m=95.0, width_v_per_m=2.0),
            )
        )
        decision = service.evaluate_pose(
            head,
            replace(nominal_pose, uncertainty=PoseUncertainty.isotropic(1e-5, 1e-4)),
        ).decision
        assert isinstance(decision, Defer)
        assert decision.detail["model_disagreement"] > decision.detail["transform_sd"]
        assert decision.suggested_action == "reversible_probe"


class TestRefuseOutsideASafe:
    def test_a_pose_far_from_the_scalp_refuses_with_R11(self, service, head, nominal_pose):
        far = _displaced(head, nominal_pose, 0.090, "far_from_scalp")
        evaluation = service.evaluate_pose(head, far)
        assert isinstance(evaluation.decision, Refuse)
        assert evaluation.decision.code == "R11"
        assert any("coil_scalp_distance_mm" in v for v in evaluation.decision.violations)

    def test_the_refusal_cites_the_limit_it_enforced(self, service, head, nominal_pose):
        evaluation = service.evaluate_pose(
            head, _displaced(head, nominal_pose, 0.090, "far")
        )
        assert "Caulfield" in " ".join(evaluation.decision.violations)

    def test_an_out_of_envelope_protocol_refuses(self, service, head, nominal_pose):
        """Exposure axes other than the pose can also leave the set."""
        loud = service.with_config(
            protocol=replace(service.config.protocol, frequency_hz=120.0)
        )
        evaluation = loud.evaluate_pose(head, nominal_pose)
        assert isinstance(evaluation.decision, Refuse)
        assert any("frequency_hz" in v for v in evaluation.decision.violations)

    def test_a_refusal_still_returns_the_field_and_the_ledger(
        self, service, head, nominal_pose
    ):
        """The consumer must see *why*, so the evidence is not withheld."""
        evaluation = service.evaluate_pose(
            head, _displaced(head, nominal_pose, 0.090, "far")
        )
        assert evaluation.efield.n_elements == head.n_vertices
        assert evaluation.ledger.variance
        assert evaluation.utility.estimable is False

    def test_refusals_are_falsey_so_a_careless_branch_fails_closed(
        self, service, head, nominal_pose
    ):
        evaluation = service.evaluate_pose(
            head, _displaced(head, nominal_pose, 0.090, "far")
        )
        assert not evaluation.decision
        assert evaluation.refused is True
        assert evaluation.recommended is False

    def test_a_safe_reports_the_axes_it_could_not_check(
        self, service, head, nominal_pose
    ):
        evaluation = service.evaluate_pose(head, nominal_pose)
        assert "tms.peak_efield_v_per_m" in evaluation.safety_axes_checked
        # whatever was declared but not supplied is named, not ignored
        assert isinstance(evaluation.safety_axes_unchecked, tuple)


class TestRecommendIsReachableAndBounded:
    def test_a_confident_well_placed_pose_can_recommend(
        self, head, nominal_pose, as_if_trained
    ):
        service = as_if_trained(TargetingService(
            response_operators=(
                MagnitudeThresholdResponse(threshold_v_per_m=45.0, width_v_per_m=6.0),
                MagnitudeThresholdResponse(
                    threshold_v_per_m=46.0,
                    width_v_per_m=6.0,
                    name="efield_magnitude_threshold_b",
                ),
            )
        ))
        evaluation = service.evaluate_pose(
            head,
            replace(nominal_pose, uncertainty=PoseUncertainty.isotropic(0.0002, 0.002)),
        )
        assert isinstance(evaluation.decision, Recommend)
        assert evaluation.decision.benefit_margin > evaluation.decision.epistemic_uncertainty

