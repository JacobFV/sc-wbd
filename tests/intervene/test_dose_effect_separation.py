"""End-to-end: the four validation levels stay four different objects.

SIMULATION ONLY.  This walks the *structure* of the running DLPFC case
(``thesis_contract.tex`` Sec. 0.5) over simulated geometry and simulated
responses.  It stops where the contract stops: no protocol is emitted, no
parameter is recommended for a person, and step 7 (a prospective, causally
identified human comparison) is out of scope for SC-WBD-001-beta.
"""

from __future__ import annotations

import math

import pytest
import torch

from scwbd.intervene.base import (
    ClinicalUtility,
    InterventionRefusal,
    NetworkEffect,
    PhysicalDose,
    TargetEngagement,
)
from scwbd.intervene.safety import (
    CompilerRefusal,
    Defer,
    FeasibleSet,
    NoRecommendation,
    ProposedIntervention,
    RiskSensitiveController,
    SimulatedRanking,
)
from scwbd.intervene.tms import (
    FigureEightCoil,
    SphericalHeadModel,
    biphasic,
    coil_pose_on_sphere,
    efield_from_coil,
    pose_field_sensitivity,
    propagate_pose_uncertainty,
)
from scwbd.intervene.tms.response import (
    DirectionalTuningResponse,
    MagnitudeThresholdResponse,
    PopulationState,
    ResponseModelSet,
    TangentialMagnitudeResponse,
    local_cortical_frame,
)

_DT = torch.float64


def _candidate_poses(head: SphericalHeadModel):
    """A small preregistered set of simulated scalp contacts."""
    return {
        f"contact_{i}": coil_pose_on_sphere(
            head,
            d,
            standoff_m=0.004,
            handle_azimuth_rad=math.radians(az),
            target_label=f"simulated dorsolateral contact {i}",
        )
        for i, (d, az) in enumerate(
            [
                ([-0.55, 0.68, 0.48], 45.0),
                ([-0.62, 0.60, 0.50], 45.0),
                ([-0.55, 0.68, 0.48], 135.0),
            ]
        )
    }


def test_the_full_chain_keeps_dose_engagement_and_network_effect_distinct(head):
    coil, pulse = FigureEightCoil(), biphasic()
    pose = coil_pose_on_sphere(head, [-0.55, 0.68, 0.48], standoff_m=0.004)
    pts, nrm = head.cortical_shell(642)

    # level 1: physical dose
    dose = efield_from_coil(coil, pulse, pose.matrix(), pts, head=head)
    assert isinstance(dose, PhysicalDose)
    assert dose.units == "V/m"

    # level 2: target engagement -- only through a NAMED candidate operator
    frame = local_cortical_frame(pts, nrm)
    state = PopulationState.resting(pts.shape[0])
    eng = TangentialMagnitudeResponse().engage(dose, frame, state, target="sim_tile")
    assert isinstance(eng, TargetEngagement)
    assert eng.response_model == "tangential_magnitude"

    # level 3: network effect -- a different object again
    net = NetworkEffect(
        readout="simulated_eeg_gfp",
        units="dimensionless",
        value=eng.value.mean().reshape(1) * 0.5,
        horizon_s=0.2,
        ledger=eng.ledger,
    )
    assert not isinstance(net, TargetEngagement)

    # level 4: refused outright
    with pytest.raises(InterventionRefusal) as e:
        ClinicalUtility()
    assert e.value.code == "R11"

    # and there is no shortcut from level 1 to level 3
    with pytest.raises(InterventionRefusal):
        dose.as_neural_effect()


def test_candidate_poses_are_compared_and_the_comparison_defers_under_disagreement(head):
    coil, pulse = FigureEightCoil(), biphasic()
    poses = _candidate_poses(head)
    pts, nrm = head.cortical_shell(642)
    frame = local_cortical_frame(pts, nrm)
    state = PopulationState.resting(pts.shape[0])
    mset = ResponseModelSet(
        [TangentialMagnitudeResponse(), MagnitudeThresholdResponse(),
         DirectionalTuningResponse()]
    )
    # a declared target tile: the first 32 vertices nearest the first contact
    dose0 = efield_from_coil(coil, pulse, poses["contact_0"].matrix(), pts, head=head)
    tile = dose0.value.norm(dim=-1).topk(32).indices

    benefit_rows, labels, proposals = [], [], []
    for label, pose in poses.items():
        dose = efield_from_coil(coil, pulse, pose.matrix(), pts, head=head)
        preds, _ = mset.predict(dose, frame, state)
        benefit_rows.append(preds[:, tile].mean(dim=-1))  # [n_models]
        labels.append(label)
        proposals.append(
            ProposedIntervention(
                label=label,
                modality="tms",
                exposure={
                    "tms.peak_efield_v_per_m": dose.peak(),
                    "tms.pulses_per_session": 600.0,
                    "tms.coil_scalp_distance_mm": pose.scalp_distance(head) * 1e3,
                },
                pose_certified=True,
                reversible=True,
            )
        )
    benefit = torch.stack(benefit_rows, dim=1)  # [n_models, n_candidates]
    benefit = benefit / benefit.abs().max()

    out = RiskSensitiveController().decide(
        proposals,
        benefit=benefit,
        burden=torch.zeros(len(proposals), dtype=_DT),
        harm=torch.zeros(benefit.shape[0], len(proposals), dtype=_DT),
        model_log_weights=torch.zeros(benefit.shape[0], dtype=_DT),
    )
    # the mechanism is unresolved, so a bounded set of near-equivalent contacts
    # must not produce a confident ranking. Any of the three honest outcomes is
    # acceptable; what is refused is a confident answer with no gap.
    assert isinstance(out, (SimulatedRanking, Defer, NoRecommendation))
    if isinstance(out, SimulatedRanking):
        assert out.benefit_gap > out.epistemic_uncertainty
        assert out.human_use_authorized is False
    else:
        assert "SIMULATION ONLY" in out.notice


@pytest.mark.slow
def test_transform_uncertainty_alone_can_withhold_a_recommendation(head):
    """Sec. 0.5 step 6, driven by a measured pose-to-field uncertainty."""
    coil, pulse = FigureEightCoil(), biphasic()
    poses = _candidate_poses(head)
    pts, _ = head.cortical_shell(642)

    unc, proposals, peaks = [], [], []
    for label, pose in poses.items():
        dose = efield_from_coil(coil, pulse, pose.matrix(), pts, head=head)
        i = int(dose.value.norm(dim=-1).argmax())
        fu = propagate_pose_uncertainty(
            coil, pulse, pose, pts, n_samples=24, seed=0, target_index=i
        )
        unc.append(fu.target_cv)
        peaks.append(dose.peak())
        proposals.append(
            ProposedIntervention(
                label=label, modality="tms",
                exposure={"tms.peak_efield_v_per_m": dose.peak()},
                pose_certified=True, reversible=True,
            )
        )
    # measured pose-driven field CV is non-trivial
    assert all(u > 0.0 for u in unc)

    benefit = torch.tensor([peaks, peaks], dtype=_DT) / max(peaks)
    out = RiskSensitiveController().decide(
        proposals,
        benefit=benefit,
        burden=torch.zeros(3, dtype=_DT),
        harm=torch.zeros(2, 3, dtype=_DT),
        model_log_weights=torch.zeros(2, dtype=_DT),
        transform_uncertainty=torch.tensor(unc, dtype=_DT) * 10.0,
    )
    assert isinstance(out, Defer)
    assert out.detail["epistemic"] >= out.detail["benefit_gap"]


def test_a_field_optimizer_that_leaves_a_safe_is_blocked(head):
    """The optimizer wants a stronger field; A_safe stops it before scoring."""
    coil = FigureEightCoil()
    fs = FeasibleSet()
    pts, _ = head.cortical_shell(162)
    scale = {"k": 1.0}

    def optimizer() -> ProposedIntervention:
        scale["k"] *= 3.0
        p = biphasic(peak_didt=1e8 * scale["k"])
        pose = coil_pose_on_sphere(head, [-0.55, 0.68, 0.48], standoff_m=0.004)
        dose = efield_from_coil(coil, p, pose.matrix(), pts, head=head)
        return ProposedIntervention(
            label=f"gain_{scale['k']:g}", modality="tms",
            exposure={"tms.peak_efield_v_per_m": dose.peak()},
            pose_certified=True,
        )

    with pytest.raises(CompilerRefusal) as e:
        fs.guard_optimizer(optimizer, max_attempts=4)
    assert e.value.code == "R11"
    assert "tms.peak_efield_v_per_m" in str(e.value)


@pytest.mark.slow
def test_reported_pose_sensitivity_numbers(head, capsys):
    """Emit the measured pose-to-field sensitivity for the record."""
    coil, pulse = FigureEightCoil(), biphasic()
    pose = coil_pose_on_sphere(
        head, [-0.55, 0.68, 0.48], standoff_m=0.004,
        handle_azimuth_rad=math.radians(45.0),
    )
    pts, _ = head.cortical_shell(2562)
    dose = efield_from_coil(coil, pulse, pose.matrix(), pts, head=head)
    i = int(dose.value.norm(dim=-1).argmax())
    rep = pose_field_sensitivity(coil, pulse, pose, pts, target_index=i)
    fu = propagate_pose_uncertainty(
        coil, pulse, pose, pts, n_samples=64, seed=0, target_index=i
    )
    print("\n" + rep.as_table())
    print(
        f"  pose covariance -> field: target {fu.target_mean:.1f} +/- "
        f"{fu.target_sd:.1f} V/m (CV {fu.target_cv:.3f}) over {fu.n_samples} draws"
    )
    assert 30.0 < rep.baseline_peak_v_per_m < 400.0
    assert rep.slope("axial_translation") < -2.0
    assert fu.target_cv > 0.0
