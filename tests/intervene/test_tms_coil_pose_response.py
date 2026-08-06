"""Coil waveforms, the pose chain and its refusals, and candidate responses.

SIMULATION ONLY.
"""

from __future__ import annotations

import math

import pytest
import torch

from scwbd.intervene.base import InterventionRefusal, PhysicalDose, TargetEngagement
from scwbd.intervene.tms.coil import (
    CircularCoil,
    FigureEightCoil,
    biphasic,
    halfsine,
    monophasic,
    pulse_waveform_spec,
)
from scwbd.intervene.tms.efield import efield_from_coil
from scwbd.intervene.tms.pose import (
    CoilPose,
    FrameTransform,
    PoseChain,
    UnderspecifiedPose,
    coil_pose_on_sphere,
    compose_chain,
    pose_field_sensitivity,
    propagate_pose_uncertainty,
    require_pose,
    se3_adjoint,
    se3_exp,
    se3_log,
    standard_fiducials,
)
from scwbd.intervene.tms.response import (
    ActivatingFunctionResponse,
    DirectionalTuningResponse,
    MagnitudeThresholdResponse,
    NormalComponentResponse,
    PopulationState,
    ResponseModelSet,
    TangentialMagnitudeResponse,
    default_candidate_set,
    local_cortical_frame,
)

_DT = torch.float64


# ---------------------------------------------------------------------------
# coil
# ---------------------------------------------------------------------------


def test_didt_is_the_analytic_derivative_of_the_current():
    for p in (monophasic(), biphasic(), halfsine()):
        t = torch.linspace(1e-7, p.duration - 1e-7, 5001, dtype=_DT)
        h = 1e-9
        fd = (p.current(t + h) - p.current(t - h)) / (2 * h)
        an = p.didt(t)
        assert float((fd - an).abs().max()) / float(an.abs().max()) < 1e-4, p.kind


def test_didt_peak_is_normalised_to_the_declared_value():
    p = biphasic(peak_didt=1.4e8)
    t = torch.linspace(0.0, p.duration, 50001, dtype=_DT)
    assert float(p.didt(t).abs().max()) == pytest.approx(1.4e8, rel=1e-3)


def test_monophasic_is_asymmetric_and_biphasic_is_not():
    assert monophasic().didt_asymmetry() > 5.0
    assert 0.7 < biphasic().didt_asymmetry() < 1.6


def test_figure_eight_has_zero_net_moment_but_a_circular_coil_does_not():
    assert abs(FigureEightCoil().net_moment_per_amp()) < 1e-12
    assert CircularCoil().total_moment_per_amp() > 1e-3


def test_closed_windings_have_zero_net_dl():
    for coil in (CircularCoil(), FigureEightCoil()):
        _, dl = coil.segments()
        assert float(dl.sum(0).norm()) < 1e-12


def test_waveform_spec_carries_the_drive_in_amps_per_second():
    spec = pulse_waveform_spec(biphasic())
    assert spec.units == "A/s"
    assert spec.name == "tms_biphasic"


# ---------------------------------------------------------------------------
# SE(3) and the chain
# ---------------------------------------------------------------------------


def test_se3_exp_log_round_trip():
    g = torch.Generator().manual_seed(11)
    for _ in range(20):
        xi = torch.randn(6, generator=g, dtype=_DT) * 0.4
        assert torch.allclose(se3_log(se3_exp(xi)), xi, atol=1e-9)


def test_chain_composition_retains_cross_covariance():
    a = FrameTransform("head", "tracker", se3_exp(torch.tensor([0.1, 0, 0, 0.02, 0, 0.05])),
                       cov=torch.eye(6, dtype=_DT) * 1e-5)
    b = FrameTransform("tracker", "coil", se3_exp(torch.tensor([0, 0.2, 0, 0, 0.01, 0])),
                       cov=torch.eye(6, dtype=_DT) * 2e-5)
    no_cross = PoseChain((a, b)).compose()
    C = 0.8 * torch.eye(6, dtype=_DT) * 1.4e-5
    with_cross = PoseChain((a, b), cross_cov={(0, 1): C}).compose()
    assert not torch.allclose(no_cross.cov, with_cross.cov)
    # shared session error inflates rather than averaging away
    assert float(with_cross.cov.trace()) > float(no_cross.cov.trace())
    # the composed transform itself is unchanged
    assert torch.allclose(no_cross.matrix, with_cross.matrix)


def test_chain_composition_equals_the_matrix_product():
    a = FrameTransform("head", "tracker", se3_exp(torch.tensor([0.1, 0, 0, 0.02, 0, 0.05])))
    b = FrameTransform("tracker", "coil", se3_exp(torch.tensor([0, 0.2, 0, 0, 0.01, 0])))
    assert torch.allclose(PoseChain((a, b)).compose().matrix, a.matrix @ b.matrix)


def test_broken_lineage_is_refused():
    a = FrameTransform("head", "tracker", torch.eye(4, dtype=_DT))
    b = FrameTransform("image", "coil", torch.eye(4, dtype=_DT))
    with pytest.raises(UnderspecifiedPose, match="broken transform lineage"):
        compose_chain(PoseChain((a, b)))


def test_non_rigid_warp_without_a_residual_is_refused():
    with pytest.raises(UnderspecifiedPose, match="non-rigid warp"):
        FrameTransform("atlas", "image", torch.eye(4, dtype=_DT), kind="nonrigid_warp")


def test_reflection_in_a_rigid_transform_is_refused():
    M = torch.eye(4, dtype=_DT)
    M[0, 0] = -1.0
    with pytest.raises(UnderspecifiedPose, match="handedness"):
        FrameTransform("head", "tracker", M)


def test_adjoint_is_consistent_with_conjugation():
    g = torch.Generator().manual_seed(5)
    T = se3_exp(torch.randn(6, generator=g, dtype=_DT) * 0.3)
    xi = torch.randn(6, generator=g, dtype=_DT) * 0.01
    lhs = T @ se3_exp(xi) @ torch.linalg.inv(T)
    rhs = se3_exp(se3_adjoint(T) @ xi)
    assert torch.allclose(lhs, rhs, atol=1e-7)


# ---------------------------------------------------------------------------
# the refusal the running case demands
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec",
    [
        "5 cm anterior",
        "5cm anterior to the motor hotspot",
        "10 mm forward",
    ],
)
def test_a_displacement_is_not_a_pose(spec):
    with pytest.raises(UnderspecifiedPose) as e:
        require_pose(spec)
    assert e.value.code == "R01"
    assert "displacement" in str(e.value)


@pytest.mark.parametrize("spec", ["F3", "left DLPFC", "vertex"])
def test_a_scalp_or_region_label_is_not_a_pose(spec):
    with pytest.raises(UnderspecifiedPose, match="label, not a pose"):
        require_pose(spec)


def test_a_position_without_orientation_is_not_a_pose():
    with pytest.raises(UnderspecifiedPose, match="orientation"):
        require_pose(torch.tensor([0.04, 0.0, 0.06], dtype=_DT))
    with pytest.raises(UnderspecifiedPose, match="orientation"):
        require_pose([0.04, 0.0, 0.06])


def test_a_pose_without_fiducials_is_refused(head):
    chain = coil_pose_on_sphere(head, [0.0, 0.0, 1.0]).chain
    with pytest.raises(UnderspecifiedPose, match="fiducials"):
        CoilPose(chain=chain, fiducials={"nasion": torch.zeros(3, dtype=_DT)})


def test_a_full_pose_is_accepted_and_round_trips(pose):
    assert require_pose(pose) is pose
    assert pose.matrix().shape == (4, 4)
    assert set(("nasion", "lpa", "rpa")).issubset(pose.fiducials)


def test_atlas_leg_is_refused_when_it_was_never_recorded(head):
    p = coil_pose_on_sphere(head, [0.0, 0.0, 1.0], with_atlas_leg=False)
    with pytest.raises(UnderspecifiedPose, match="atlas"):
        p.full_chain()


def test_full_chain_composes_all_four_factors_of_equation_three(pose):
    fc = pose.full_chain()
    assert len(fc.factors) == 4
    assert fc.dst == "atlas" and fc.src == "coil"
    composed = fc.compose()
    # the atlas leg adds uncertainty; it never removes any
    assert float(composed.cov.trace()) > float(pose.covariance().trace())


# ---------------------------------------------------------------------------
# pose -> field sensitivity, the Caulfield point
# ---------------------------------------------------------------------------


def test_pose_to_field_sensitivity_is_material_and_axis_specific(
    coil, pulse, pose, head, cortex
):
    pts, _ = cortex
    mag = efield_from_coil(coil, pulse, pose.matrix(), pts, head=head).value.norm(dim=-1)
    i = int(mag.argmax())
    rep = pose_field_sensitivity(coil, pulse, pose, pts, target_index=i)

    # scalp distance dominates: several percent per millimetre
    axial = rep.slope("axial_translation")
    assert -6.0 < axial < -2.0, axial
    assert rep.by_name("axial_translation").target_pct < -3.0
    five_mm = [r for r in rep.rows if r.name == "axial_translation" and r.magnitude == 5.0][0]
    assert five_mm.target_pct < -15.0

    # tilt about the handle axis lifts a wing; tilt about the wing axis does not
    tilt_handle = abs(rep.slope("tilt_about_handle_axis"))
    tilt_wing = abs(rep.slope("tilt_about_wing_axis"))
    assert tilt_handle > 3 * tilt_wing

    # rotating the coil about its own axis barely changes |E| but rotates the
    # field direction by the same angle: magnitude alone hides orientation
    rot10 = [
        r
        for r in rep.rows
        if r.name == "rotation_about_coil_axis" and r.magnitude == 10.0
    ][0]
    assert abs(rot10.target_pct) < 0.5
    assert rot10.target_direction_change_deg == pytest.approx(10.0, abs=0.5)


def test_pose_covariance_propagates_into_field_uncertainty(coil, pulse, pose, head, cortex):
    pts, _ = cortex
    mag = efield_from_coil(coil, pulse, pose.matrix(), pts, head=head).value.norm(dim=-1)
    i = int(mag.argmax())
    fu = propagate_pose_uncertainty(
        coil, pulse, pose, pts, n_samples=64, seed=0, target_index=i
    )
    assert fu.target_sd > 0.0
    assert 0.001 < fu.target_cv < 0.5
    # bias is carried separately from variance
    assert fu.ledger.bias_status == "design_estimable"
    assert "pose" in fu.ledger.variance
    # deterministic given the seed
    fu2 = propagate_pose_uncertainty(
        coil, pulse, pose, pts, n_samples=64, seed=0, target_index=i
    )
    assert fu.target_sd == pytest.approx(fu2.target_sd)


def test_a_systematic_twist_bias_shifts_the_field_and_is_reported_separately(
    coil, pulse, head, cortex
):
    pts, _ = cortex
    bias = torch.tensor([0.0, 0, 0, 0, 0, 0.003], dtype=_DT)  # 3 mm standoff bias
    p = coil_pose_on_sphere(head, [-0.55, 0.68, 0.48], bias_twist=bias)
    mag = efield_from_coil(coil, pulse, p.matrix(), pts, head=head).value.norm(dim=-1)
    i = int(mag.argmax())
    fu = propagate_pose_uncertainty(coil, pulse, p, pts, n_samples=16, seed=1, target_index=i)
    assert fu.bias_shift_v_per_m < -1.0  # 3 mm further away -> weaker field
    assert fu.ledger.bias_interval[0] < 0.0


# ---------------------------------------------------------------------------
# candidate response operators
# ---------------------------------------------------------------------------


def _dose_and_frame(coil, pulse, pose, head, n=162):
    pts, nrm = head.cortical_shell(n)
    dose = efield_from_coil(coil, pulse, pose.matrix(), pts, head=head)
    frame = local_cortical_frame(pts, nrm, fibre_direction=nrm)
    return dose, frame


def test_candidate_set_is_plural_and_none_claims_a_mechanism():
    cands = default_candidate_set()
    assert len(cands) >= 4
    assert all(c.mechanistic_status != "mechanistic" for c in cands)
    assert all(c.disabling_evidence for c in cands)


def test_a_single_candidate_is_not_an_admissible_model_set():
    with pytest.raises(ValueError, match="unresolved"):
        ResponseModelSet([NormalComponentResponse()])


def test_response_operators_produce_target_engagement_not_a_dose(coil, pulse, pose, head):
    dose, frame = _dose_and_frame(coil, pulse, pose, head)
    st = PopulationState.resting(frame.points.shape[0])
    te = NormalComponentResponse().engage(dose, frame, st, target="sim_dlpfc_tile")
    assert isinstance(te, TargetEngagement)
    assert not isinstance(te, PhysicalDose)
    assert te.response_model == "normal_component"
    assert te.ledger.model_discrepancy is not None
    assert te.ledger.validity_domain["mechanism_resolved"] is False


def test_orientation_dependence_normal_vs_tangential(coil, pulse, pose, head):
    dose, frame = _dose_and_frame(coil, pulse, pose, head)
    st = PopulationState.resting(frame.points.shape[0])
    en = NormalComponentResponse().engage(dose, frame, st).value
    et = TangentialMagnitudeResponse().engage(dose, frame, st).value
    # in a spherical conductor the field is purely tangential (theorem), so a
    # normal-component operator predicts ~nothing and a tangential one does not
    assert float(en.abs().max()) < 1e-9 * float(et.abs().max())
    assert float(et.abs().max()) > 0.0


def test_reversing_the_coil_flips_a_signed_operator_but_not_an_unsigned_one(
    coil, pulse, pose, head
):
    pts, nrm = head.cortical_shell(162)
    frame = local_cortical_frame(pts, nrm)
    st = PopulationState.resting(pts.shape[0])
    d1 = efield_from_coil(coil, pulse, pose.matrix(), pts, head=head)
    flipped = pose.matrix() @ se3_exp(torch.tensor([0.0, 0, math.pi, 0, 0, 0], dtype=_DT))
    d2 = efield_from_coil(coil, pulse, flipped, pts, head=head)

    tan1 = TangentialMagnitudeResponse().engage(d1, frame, st).value
    tan2 = TangentialMagnitudeResponse().engage(d2, frame, st).value
    assert torch.allclose(tan1, tan2, rtol=1e-6)

    dir1 = DirectionalTuningResponse().engage(d1, frame, st).value
    dir2 = DirectionalTuningResponse().engage(d2, frame, st).value
    assert not torch.allclose(dir1, dir2, rtol=1e-3)


def test_state_dependence_changes_the_predicted_engagement(coil, pulse, pose, head):
    dose, frame = _dose_and_frame(coil, pulse, pose, head)
    n = frame.points.shape[0]
    low = PopulationState(excitability=torch.full((n,), 0.6, dtype=_DT))
    high = PopulationState(excitability=torch.full((n,), 1.6, dtype=_DT))
    op = MagnitudeThresholdResponse()
    a = op.engage(dose, frame, low).value
    b = op.engage(dose, frame, high).value
    assert float(b.max()) > float(a.max())


def test_phase_dependence_is_available_and_off_by_default(coil, pulse, pose, head):
    dose, frame = _dose_and_frame(coil, pulse, pose, head)
    n = frame.points.shape[0]
    base = PopulationState.resting(n)
    phased = PopulationState(
        excitability=torch.ones(n, dtype=_DT),
        phase=torch.zeros(n, dtype=_DT),
        band_hz=10.0,
    )
    op = TangentialMagnitudeResponse()
    assert torch.allclose(op.engage(dose, frame, base).value, op.engage(dose, frame, phased).value)
    op.phase_preference = 0.0
    op.phase_depth = 0.5
    assert float(op.engage(dose, frame, phased).value.max()) > float(
        op.engage(dose, frame, base).value.max()
    )


def test_activating_function_refuses_to_substitute_the_normal_silently(
    coil, pulse, pose, head
):
    pts, nrm = head.cortical_shell(162)
    dose = efield_from_coil(coil, pulse, pose.matrix(), pts, head=head)
    frame = local_cortical_frame(pts, nrm)  # no fibre direction
    with pytest.raises(ValueError, match="fibre_direction"):
        ActivatingFunctionResponse().engage(dose, frame, PopulationState.resting(pts.shape[0]))


def test_model_comparison_recovers_the_generating_operator(coil, pulse, pose, head):
    dose, frame = _dose_and_frame(coil, pulse, pose, head, n=642)
    st = PopulationState.resting(frame.points.shape[0])
    mset = ResponseModelSet(
        [
            TangentialMagnitudeResponse(),
            MagnitudeThresholdResponse(),
            DirectionalTuningResponse(),
        ]
    )
    preds, names = mset.predict(dose, frame, st)
    truth = preds[2]  # directional tuning generated the data
    g = torch.Generator().manual_seed(4)
    obs = truth + 0.02 * truth.std() * torch.randn(
        truth.numel(), generator=g, dtype=_DT
    )
    cmp_ = mset.compare(preds, obs, names)
    assert cmp_.best() == "directional_tuning"
    assert set(cmp_.names) == set(names)
    # a near-perfect fit resolves the comparison, and a resolved posterior has
    # (by construction) little disagreement left to report
    assert cmp_.is_resolved()
    assert cmp_.disagreement < 0.05
    # ... whereas the flat prior it started from does not
    assert mset.disagreement(preds) > 0.1


def test_model_comparison_stays_unresolved_when_the_data_do_not_separate(
    coil, pulse, pose, head
):
    dose, frame = _dose_and_frame(coil, pulse, pose, head, n=642)
    st = PopulationState.resting(frame.points.shape[0])
    mset = ResponseModelSet(
        [TangentialMagnitudeResponse(), MagnitudeThresholdResponse(),
         DirectionalTuningResponse()]
    )
    preds, names = mset.predict(dose, frame, st)
    g = torch.Generator().manual_seed(9)
    # observation dominated by noise: nothing to choose between candidates
    obs = torch.randn(preds.shape[1], generator=g, dtype=_DT)
    cmp_ = mset.compare(preds, obs, names)
    assert not cmp_.is_resolved()
    assert cmp_.disagreement > 0.05


def test_disagreement_is_large_when_candidates_disagree(coil, pulse, pose, head):
    dose, frame = _dose_and_frame(coil, pulse, pose, head, n=642)
    st = PopulationState.resting(frame.points.shape[0])
    mset = ResponseModelSet(
        [TangentialMagnitudeResponse(), MagnitudeThresholdResponse(),
         DirectionalTuningResponse()]
    )
    preds, _ = mset.predict(dose, frame, st)
    assert mset.disagreement(preds) > 0.1
    mu = mset.to_mechanistic_uncertainty()
    assert not mu.resolved
    assert len(mu.candidates) == 3
