"""The runtime consumes the gated field solvers, their bounds, and their refusals.

Defects foreclosed:

* **a constant standing in for a measurement.**  The numerical term on the field
  path used to be a hardcoded 2 %.  It now comes from
  ``bem_error_envelope(panel_to_standoff)`` evaluated on the mesh the solver
  actually used -- a step function over gate N7/N8's refinement table -- and
  this module asserts the two are the same number, and that refining the mesh
  changes it.
* **an unvalidated field reaching a consumer.**
  ``ChargeBEM.assert_resolves_sources`` refuses outside the validated envelope,
  where gate N7/N8 measured 16 % error with *non-monotonic* refinement -- a
  coarser mesh scores better, so a user watching the error converge would be
  watching nothing.  The runtime turns that refusal into ``Defer``, never into
  an exception escaping to the bridge and never into a number.
* **blanket deferral of the contact regime.**  N8 passes at
  ``a/R_c = 0.955`` with 0.73 % mean relative error against an independent
  reference, so contact-regime targeting proceeds *inside the declared
  envelope* and defers only outside it.
* **a coil buried in the scalp.**  The coil frame's ``+z`` points away from the
  head.  Getting that backwards puts every winding ``winding_height`` deep in
  the conductor, where the interior solution's denominator passes through zero
  and returns a large, smooth, fictitious number.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from scwbd.runtime import (
    ChargeBEMEField,
    CoilSpec,
    Defer,
    GatedAnalyticSphereEField,
    PoseRequest,
    ProvenanceExpectation,
    ProvenanceMismatch,
    Recommend,
    Refuse,
    TargetingService,
    Unresolved,
    coil_pose_over_region,
    spherical_phantom,
)
from scwbd.runtime.backends import (
    AnalyticSphericalEField,
    ImpossiblePlacement,
    _dipoles_in_head_frame,
)
from scwbd.schema.ledger import VARIANCE_COMPONENTS
from scwbd.transforms.se3 import Pose
from scwbd.transforms.uncertainty import PoseUncertainty

efield = pytest.importorskip("scwbd.intervene.tms.efield")

_DT = torch.float64


@pytest.fixture(scope="module")
def small_head():
    """A smaller phantom: the dense BEM solve is cubic in the panel count."""
    return spherical_phantom(
        subject_id="phantom-bem-001", n_vertices=120, n_parcels=12
    )


@pytest.fixture(scope="module")
def bem_pose(small_head):
    return PoseRequest(
        pose=coil_pose_over_region(small_head, method="bem_fixture"),
        frame=small_head.frame,
        label="bem_fixture",
        uncertainty=PoseUncertainty.isotropic(1e-3, 1e-2),
    )


@pytest.fixture(scope="module")
def bem_evaluation(small_head, bem_pose):
    service = TargetingService(efield_backend=ChargeBEMEField())
    return service.evaluate_pose(small_head, bem_pose)


class TestTheGatedSolverIsWhatRuns:
    def test_the_default_backend_is_the_gated_one(self, service):
        assert isinstance(service.efield_backend, GatedAnalyticSphereEField)
        assert service.provenance.efield_backend.startswith("scwbd.intervene.tms")

    def test_the_provenance_names_the_gates_the_field_passed(self, service):
        assert "N6_induced_efield" in service.provenance.efield_gate_evidence

    def test_the_runtimes_fallback_overestimates_and_says_so(self, head):
        """Recorded because wiring the gated solver *changed* the answer.

        The fallback returns the tangential projection of the primary field.
        That is not the Sarvas / Heller--van Hulsteyn interior solution: the
        secondary field carries a tangential component too, and dropping it
        inflates the magnitude by ~1.5x here while leaving the direction alone.
        The fallback exists to keep the runtime's structure exercisable without
        agent G, and it is labelled an approximation everywhere it appears.
        """
        pose = coil_pose_over_region(head)
        coil = CoilSpec.figure_eight()
        gated = GatedAnalyticSphereEField().solve(head, pose, coil)
        local = AnalyticSphericalEField().solve(head, pose, coil)

        gated_mag = torch.linalg.norm(gated, dim=-1)
        local_mag = torch.linalg.norm(local, dim=-1)
        peak = int(gated_mag.argmax())
        cosine = float(
            (gated[peak] @ local[peak])
            / (gated_mag[peak] * local_mag[peak]).clamp_min(1e-30)
        )
        assert cosine > 0.999, "the approximation is parallel to the reference"
        ratio = float(local_mag.max() / gated_mag.max())
        assert 1.4 < ratio < 1.7, f"measured overestimate {ratio:.3f}"
        # and the fallback declares a discrepancy prior wide enough to contain it
        lo, hi = AnalyticSphericalEField().discrepancy_fraction
        assert hi >= ratio - 1.0
        assert AnalyticSphericalEField().gate_evidence == ()

    def test_a_non_spherical_head_is_refused_not_approximated(self, head):
        dented = replace(
            head,
            subject_id="dented",
            cortex_vertices=head.cortex_vertices * torch.tensor(
                [1.0, 1.0, 1.06], dtype=_DT
            ),
        )
        with pytest.raises(ImpossiblePlacement) as exc:
            GatedAnalyticSphereEField().solve(
                dented, coil_pose_over_region(head), CoilSpec.figure_eight()
            )
        assert "spherically symmetric" in str(exc.value)


class TestTheCoilFrameConvention:
    def test_plus_z_points_away_from_the_head(self, head):
        pose = coil_pose_over_region(head)
        outward = (pose.t - head.centre) / torch.linalg.norm(pose.t - head.centre)
        assert float(pose.R[:, 2] @ outward) > 0.99

    def test_the_windings_clear_the_scalp(self, head):
        """The finding N8's handover records, reproduced as a regression test.

        A figure-eight coil is flat and a head is curved, so the nearest
        *winding* stands much further off the scalp than the coil face does.
        At a 4 mm face standoff the nearest winding is ~9 mm out, which puts a
        clinical placement at ``a/R_c ~ 0.90`` -- **easier** than the 0.955
        contact case N8 validated, not at its edge.
        """
        pose = coil_pose_over_region(head, standoff_m=0.004)
        positions, _ = _dipoles_in_head_frame(pose, CoilSpec.figure_eight())
        source_radius = float(torch.linalg.norm(positions - head.centre, dim=-1).min())
        winding_standoff_mm = (source_radius - head.scalp_radius) * 1e3
        a_over_rc = head.scalp_radius / source_radius
        assert winding_standoff_mm > 4.0, "the nearest winding is closer than the face"
        assert 8.0 < winding_standoff_mm < 12.0
        assert a_over_rc < 0.955, "a clinical placement is inside N8's contact case"

    def test_the_inverted_convention_is_refused_as_an_impossible_placement(
        self, service, head
    ):
        """Point +z into the head and every winding lands inside the scalp."""
        good = coil_pose_over_region(head, standoff_m=0.004)
        flipped_R = good.R.clone()
        flipped_R[:, 1] = -flipped_R[:, 1]
        flipped_R[:, 2] = -flipped_R[:, 2]
        bad = Pose.from_Rt(
            flipped_R, good.t, head.frame, "coil", provenance={"method": "inverted"}
        )
        evaluation = service.evaluate_pose(
            head,
            PoseRequest(
                pose=bad,
                frame=head.frame,
                label="inverted_coil",
                uncertainty=PoseUncertainty.isotropic(1e-3, 1e-2),
            ),
        )
        assert isinstance(evaluation.decision, Refuse)
        assert evaluation.decision.code == "R06"
        assert evaluation.unresolved_quantities()


class TestTheCalibratedBoundIsConsumed:
    def test_the_bem_path_runs_and_reports_a_measured_resolution(self, bem_evaluation):
        accuracy = bem_evaluation.field_accuracy
        assert not isinstance(accuracy, Unresolved)
        assert accuracy.peak_v_per_m > 0.0
        resolution = accuracy.near_source_resolution
        assert {"standoff_m", "panel_edge_m", "panel_to_standoff"} <= set(resolution)
        assert resolution["panel_to_standoff"] <= efield.MAX_PANEL_TO_STANDOFF

    def test_the_bound_is_the_gate_table_not_a_constant(self, bem_evaluation):
        accuracy = bem_evaluation.field_accuracy
        expected = efield.bem_error_envelope(
            accuracy.near_source_resolution["panel_to_standoff"]
        )
        assert accuracy.solver_relative_error_bound == pytest.approx(expected)
        # and it is not the 2 % that used to be hardcoded on this path
        assert accuracy.solver_relative_error_bound != 0.02 or expected == 0.02

    def test_refining_the_mesh_changes_the_bound(self, small_head, bem_pose):
        """A constant would not move; a calibrated envelope does."""
        coarse = TargetingService(
            efield_backend=ChargeBEMEField(uniform_subdiv=3)
        ).evaluate_pose(small_head, bem_pose)
        fine = TargetingService(
            efield_backend=ChargeBEMEField(base_subdiv=2, grading_levels=2)
        ).evaluate_pose(small_head, bem_pose)
        coarse_ratio = coarse.field_accuracy.near_source_resolution["panel_to_standoff"]
        fine_ratio = fine.field_accuracy.near_source_resolution["panel_to_standoff"]
        assert fine_ratio < coarse_ratio
        assert (
            fine.field_accuracy.solver_relative_error_bound
            < coarse.field_accuracy.solver_relative_error_bound
        )

    def test_the_solver_bound_reaches_the_field_ledger(self, bem_evaluation):
        domain = bem_evaluation.efield.ledger.validity_domain
        assert domain["solver_relative_error_bound"] == pytest.approx(
            bem_evaluation.field_accuracy.solver_relative_error_bound
        )
        # the numerical term is the sum of two *measured* discretisations
        assert (
            bem_evaluation.efield.ledger.variance["numerical"]
            >= domain["solver_numerical_variance"]
        )
        assert domain["coil_discretisation_numerical_sd_v_per_m"] >= 0.0

    def test_the_contact_regime_is_not_a_blanket_deferral(self, bem_evaluation):
        """N8 passes, so a resolved contact-regime pose gets a real decision."""
        assert not isinstance(bem_evaluation.decision, Refuse)
        assert bem_evaluation.field_resolved
        assert isinstance(bem_evaluation.decision, (Recommend, Defer))

    def test_the_field_accuracy_status_names_the_independent_reference(
        self, bem_evaluation
    ):
        accuracy = bem_evaluation.field_accuracy
        assert accuracy.validation_status == "cross_solver"
        assert "N8_induced_efield_contact" in accuracy.validated_against
        assert accuracy.validation_status != "in_vivo"

    def test_a_numerical_gate_does_not_narrow_the_geometry_prior(self, bem_evaluation):
        """N8 validates the discretisation. It does not make a sphere a head."""
        lo, hi = bem_evaluation.efield.ledger.bias_interval
        peak = bem_evaluation.field_accuracy.peak_v_per_m
        assert (hi - lo) / peak == pytest.approx(0.8, rel=1e-6)


@pytest.fixture(scope="module")
def coarse_evaluation(small_head, bem_pose):
    """Deliberately too coarse: 80 panels over a 92 mm sphere, where gate
    N7/N8 measured 106 % error and non-monotonic refinement."""
    return TargetingService(
        efield_backend=ChargeBEMEField(uniform_subdiv=1)
    ).evaluate_pose(small_head, bem_pose)


class TestTheResolutionRefusalBecomesDefer:
    def test_it_defers_rather_than_raising(self, coarse_evaluation):
        assert isinstance(coarse_evaluation.decision, Defer)

    def test_the_defer_says_a_mesh_is_the_gap_not_a_measurement(self, coarse_evaluation):
        decision = coarse_evaluation.decision
        assert decision.suggested_action == "no_action"
        assert "does not resolve the near-source field" in decision.reason
        assert "non-monotonic" in decision.reason

    def test_the_defer_carries_the_measured_resolution(self, coarse_evaluation):
        detail = coarse_evaluation.decision.detail
        assert detail["panel_to_standoff"] > efield.MAX_PANEL_TO_STANDOFF

    def test_every_downstream_quantity_is_unresolved_never_zero(self, coarse_evaluation):
        assert set(coarse_evaluation.unresolved_quantities()) == {
            "efield",
            "field_accuracy",
            "target_engagement",
            "network_effect",
        }
        for value in coarse_evaluation.four_quantities().values():
            if isinstance(value, Unresolved):
                assert "resolve the near-source field" in value.reason
        assert coarse_evaluation.field_resolved is False

    def test_the_ledger_is_still_populated_from_what_is_known(self, coarse_evaluation):
        ledger = coarse_evaluation.ledger
        assert set(ledger.variance) == set(VARIANCE_COMPONENTS)
        assert ledger.has_estimator()
        assert ledger.units == "m", "with no field there is no readout to report in"
        assert ledger.validity_domain["terms_not_applicable"] == [
            "parameter",
            "model_class",
            "numerical",
        ]

    def test_the_utility_slot_is_still_occupied(self, coarse_evaluation):
        assert coarse_evaluation.utility.estimable is False

    def test_the_summary_reports_reasons_not_zeros(self, coarse_evaluation):
        summary = coarse_evaluation.summary()
        assert "field_peak_v_per_m" not in summary
        assert "resolve the near-source field" in summary["field_unresolved_reason"]
        assert summary["unresolved_quantities"]

    def test_an_unresolved_evaluation_can_never_be_a_recommendation(
        self, coarse_evaluation
    ):
        from dataclasses import replace as dc_replace

        with pytest.raises(Exception) as exc:
            dc_replace(
                coarse_evaluation,
                decision=Recommend(
                    label="x", rationale="", benefit_margin=1.0, epistemic_uncertainty=0.0
                ),
            )
        assert "resting on nothing" in str(exc.value)


class TestAConsumerCanDemandTheGates:
    def test_the_analytic_backend_does_not_claim_the_contact_gate(self, served):
        with pytest.raises(ProvenanceMismatch) as exc:
            served.handshake(
                ProvenanceExpectation(
                    require_efield_gates=("N8_induced_efield_contact",)
                )
            )
        assert any("efield_gate_evidence" in m for m in exc.value.mismatches)

    def test_the_analytic_backend_does_claim_the_induction_gate(self, served):
        served.handshake(
            ProvenanceExpectation(require_efield_gates=("N6_induced_efield",))
        )

    def test_the_bem_backend_claims_both(self):
        service = TargetingService(efield_backend=ChargeBEMEField())
        assert set(service.provenance.efield_gate_evidence) == {
            "N6_induced_efield",
            "N8_induced_efield_contact",
        }
        assert service.provenance.efield_backend_class == "numerical_bem"

