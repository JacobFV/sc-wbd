"""The ledger is always populated and the four quantities never collapse.

Defects foreclosed:

* thesis Sec. 2.7 / refusal R08 -- an object that reports one confidence number,
  or a bias point estimate with no estimator behind it.  ``ARCHITECTURE.md``
  Sec. 6 says the ledger on a ``PoseEvaluation`` is populated *always*; here
  that is enforced at construction, so an incomplete ledger cannot exist.
* thesis Sec. 7.2 -- "Pose accuracy, field accuracy, target engagement, network
  change, symptom change, and comparative clinical utility are separate
  validation levels."  Nothing in the runtime fuses them, target engagement
  refuses ``float()``, and utility refuses to carry a value at all.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from scwbd.runtime import (
    EngagementDistribution,
    NetworkResponse,
    UtilityStatus,
)
from scwbd.runtime.types import LedgerIncomplete, full_ledger
from scwbd.schema.ledger import VARIANCE_COMPONENTS, UncertaintyLedger
from scwbd.transforms.uncertainty import PoseUncertainty


@pytest.fixture(scope="module")
def evaluation(service, head):
    from scwbd.runtime.serving import _default_warmup_pose

    base = replace(
        _default_warmup_pose(head),
        label="ledger_fixture",
        uncertainty=PoseUncertainty.isotropic(0.0015, 0.015),
    )
    return service.evaluate_pose(head, base)


class TestTheLedgerIsAlwaysPopulated:
    def test_every_variance_component_is_present(self, evaluation):
        assert set(evaluation.ledger.variance) == set(VARIANCE_COMPONENTS)

    def test_the_bias_is_an_interval_backed_by_its_declared_status(self, evaluation):
        lo, hi = evaluation.ledger.bias_interval
        assert hi > lo, "a degenerate bias interval is a point estimate (R08)"
        assert evaluation.ledger.has_estimator()

    def test_nested_objects_carry_their_own_complete_ledgers(self, evaluation):
        for obj in (
            evaluation.efield,
            evaluation.target_engagement,
            evaluation.network_response,
        ):
            assert set(obj.ledger.variance) == set(VARIANCE_COMPONENTS)
            assert obj.ledger.has_estimator()

    def test_a_refused_evaluation_still_carries_a_full_ledger(
        self, service, head, nominal_pose
    ):
        from scwbd.runtime import Refuse
        from scwbd.transforms.se3 import Pose

        direction = nominal_pose.pose.t - head.centre
        direction = direction / torch.linalg.norm(direction)
        pose = Pose.from_Rt(
            nominal_pose.pose.R,
            head.centre + direction * (head.scalp_radius + 0.09),
            head.frame,
            "coil",
            provenance={"method": "out_of_envelope"},
        )
        result = service.evaluate_pose(head, replace(nominal_pose, pose=pose))
        assert isinstance(result.decision, Refuse)
        assert set(result.ledger.variance) == set(VARIANCE_COMPONENTS)

    def test_an_incomplete_ledger_cannot_be_attached(self):
        bad = UncertaintyLedger(variance={"measurement": 1.0}, bias_interval=(-1.0, 1.0))
        with pytest.raises(LedgerIncomplete) as exc:
            from scwbd.runtime.types import check_ledger

            check_ledger(bad, what="test")
        assert exc.value.code == "R08"

    def test_a_bias_point_estimate_without_an_estimator_is_refused(self):
        with pytest.raises(LedgerIncomplete):
            full_ledger(
                units="V/m",
                measurement=1.0,
                within_session=0.0,
                between_session=0.0,
                parameter=0.0,
                model_class=0.0,
                numerical=0.0,
                bias_interval=(0.0, 0.0),
                bias_status="prior_specified_sensitivity",
            )

    def test_the_variance_split_traces_to_declared_registration_scopes(
        self, service, head_with_chain
    ):
        """A between-session registration edge lands in between_session."""
        from scwbd.transforms.se3 import Pose

        from scwbd.runtime import PoseRequest

        pose = Pose.from_Rt(
            torch.eye(3, dtype=torch.float64),
            [0.0, 0.0, 0.10],
            "consumer_head_ALS",
            "coil",
            provenance={"method": "declared_by_consumer"},
        )
        result = service.evaluate_pose(
            head_with_chain,
            PoseRequest(
                pose=pose,
                frame="consumer_head_ALS",
                uncertainty=PoseUncertainty.isotropic(0.001, 0.01),
                uncertainty_scope="within_session",
            ),
        )
        assert result.ledger.variance["between_session"] > 0.0
        assert result.ledger.variance["within_session"] > 0.0

    def test_the_validity_domain_records_that_this_is_a_phantom(self, evaluation):
        domain = evaluation.ledger.validity_domain
        assert domain["is_phantom"] is True
        assert domain["human_use_authorized"] is False


class TestTheFourQuantitiesNeverCollapse:
    def test_four_distinct_objects_are_reported(self, evaluation):
        quantities = evaluation.four_quantities()
        assert set(quantities) == {
            "field_accuracy",
            "target_engagement",
            "network_effect",
            "utility",
        }
        assert len({id(v) for v in quantities.values()}) == 4
        assert len({type(v) for v in quantities.values()}) == 4

    def test_target_engagement_refuses_to_become_a_number(self, evaluation):
        with pytest.raises(Exception) as exc:
            float(evaluation.target_engagement)
        assert "distribution" in str(exc.value)

    def test_target_engagement_is_a_distribution_over_named_operators(self, evaluation):
        engagement = evaluation.target_engagement
        assert len(engagement.response_models) >= 2
        assert engagement.samples.shape[0] == len(engagement.response_models)
        assert engagement.samples.shape[1] > 1
        assert engagement.sd() > 0.0
        assert engagement.model_disagreement() >= 0.0
        # every number is attributable to a named operator
        assert all(name for name in engagement.response_models)
        assert len(engagement.mechanistic_status) == len(engagement.response_models)

    def test_network_response_keeps_model_classes_apart(self, evaluation):
        response = evaluation.network_response
        assert len(response.model_classes) >= 2
        assert response.per_model.shape[0] == len(response.model_classes)
        assert response.model_class_disagreement() >= 0.0
        with pytest.raises(Exception):
            float(response)

    def test_utility_is_reported_and_reports_itself_as_not_estimable(self, evaluation):
        assert isinstance(evaluation.utility, UtilityStatus)
        assert evaluation.utility.estimable is False
        assert evaluation.utility.value is None
        with pytest.raises(Exception):
            evaluation.utility.require_value()
        assert evaluation.utility.as_unresolved().missing == ("clinical_utility",)

    def test_utility_cannot_be_declared_estimable(self):
        with pytest.raises(Exception):
            UtilityStatus(estimable=True)

    def test_no_object_exposes_a_fused_score(self, evaluation):
        forbidden = {"score", "total_score", "combined", "overall", "fused", "utility_score"}
        for obj in (
            evaluation,
            evaluation.field_accuracy,
            evaluation.target_engagement,
            evaluation.network_response,
            evaluation.utility,
        ):
            names = {n for n in dir(obj) if not n.startswith("_")}
            assert forbidden.isdisjoint(names), f"{type(obj).__name__} exposes {names & forbidden}"

    def test_a_physical_dose_still_refuses_to_become_a_neural_effect(self, evaluation):
        dose = evaluation.efield.as_physical_dose()
        with pytest.raises(Exception) as exc:
            dose.as_neural_effect()
        assert "R04" in str(exc.value)

    def test_field_accuracy_states_what_it_was_validated_against(self, evaluation):
        accuracy = evaluation.field_accuracy
        assert accuracy.validation_status in (
            "unvalidated",
            "solver_refinement_only",
            "phantom",
            "cross_solver",
            "in_vivo",
        )
        assert accuracy.validation_status != "in_vivo"
        assert accuracy.peak_sd_v_per_m > 0.0


class TestTheEFieldCarriesItsCovariance:
    def test_covariance_is_per_vertex_and_positive_semidefinite(self, evaluation):
        cov = evaluation.efield.covariance
        assert cov.shape == (evaluation.efield.n_elements, 3, 3)
        eigenvalues = torch.linalg.eigvalsh(cov)
        assert float(eigenvalues.min()) > -1e-9

    def test_inflating_the_pose_covariance_inflates_the_field_covariance(
        self, service, head, nominal_pose
    ):
        tight = service.evaluate_pose(
            head, replace(nominal_pose, uncertainty=PoseUncertainty.isotropic(1e-4, 1e-3))
        )
        loose = service.evaluate_pose(
            head, replace(nominal_pose, uncertainty=PoseUncertainty.isotropic(1e-2, 1e-1))
        )
        assert loose.efield.peak_sd() > tight.efield.peak_sd()
        assert loose.field_accuracy.transform_sd_v_per_m > tight.field_accuracy.transform_sd_v_per_m

    def test_the_analytic_backend_declares_that_conductivity_does_not_enter(
        self, evaluation
    ):
        domain = evaluation.efield.ledger.validity_domain
        if evaluation.efield.backend_class == "analytic":
            assert domain["conductivity_enters_solution"] is False


def _unused(*_):  # pragma: no cover
    return EngagementDistribution, NetworkResponse
