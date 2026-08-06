"""Negative controls: every gate must FAIL where its null is literally true.

This is the file that makes the rest of ``scwbd.bench`` worth anything.  Each
world below is constructed so that the claim under test is *false by
construction* — the slow modality carries no information, the "anatomical"
graph is an unrelated random graph, there is nothing below the parcel, the
perturbation informs only the field model, subjects differ only by noise — and
the corresponding gate is required to say so.
"""

from __future__ import annotations

import numpy as np
import pytest

from scwbd.bench.gates import Thresholds, run_g1, run_g2, run_g3, run_g4, run_g5
from scwbd.bench.synthetic import (
    HallucinatingFineModel,
    HybridMechResidual,
    RidgeGaussian,
    SyntheticFisher,
    make_fusion_dataset,
    make_graph_dataset,
    make_individualization_dataset,
    make_multiscale_dataset,
)

from .conftest import FIXTURE_THRESHOLDS, decision_problem, g1_arms, g5_arms

RECOVERY_OK = {
    k: {"true": 1.0, "estimate": 1.02, "lo": 0.9, "hi": 1.15}
    for k in ("direction", "delay", "gain", "dose", "state_dependence")
}
EVIDENCE_OK = {
    "joint_native": {"m1": -1.00, "m2": -1.01},
    "joint_plus_impulse": {"m1": -0.90, "m2": -1.30},
}


# --------------------------------------------------------------------------
def test_g1_fails_when_the_second_modality_carries_no_information():
    d = make_fusion_dataset(seed=1, bold_informative=False)
    cand, base = g1_arms()
    rep = run_g1(
        train=d["train"], test=d["test"], candidate=cand, baselines=base,
        delay_true=3.0,
        delay_estimates={"candidate": [3.1, 2.9, 3.05],
                         "naive_resampling": [4.4, 4.1, 4.6],
                         "single_modality_eeg": [3.9, 4.2, 3.7]},
        intervention={"train": d["train"], "test": d["test"]},
        thresholds=FIXTURE_THRESHOLDS, seed=0,
    )
    assert rep.status == "FAIL"
    assert rep.consequence.startswith("Retain only the provenance/type system")
    names = {s.name for s in rep.subchecks if s.status == "FAIL"}
    assert "heldout_log_score" in names


def test_g1_fails_when_the_fusion_model_is_overconfident():
    from scwbd.bench.synthetic import OverconfidentModel

    d = make_fusion_dataset(seed=1, bold_informative=True)
    cand, base = g1_arms()
    over = OverconfidentModel(inner=cand, factor=0.3)
    over.name = "typed_fusion_overconfident"
    rep = run_g1(
        train=d["train"], test=d["test"], candidate=over, baselines=base,
        delay_true=3.0,
        delay_estimates={"candidate": [3.0], "naive_resampling": [4.4],
                         "single_modality_eeg": [3.9]},
        intervention={"train": d["train"], "test": d["test"]},
        thresholds=FIXTURE_THRESHOLDS, seed=0,
    )
    assert rep.status == "FAIL"
    failed = {s.name for s in rep.subchecks if s.status == "FAIL"}
    assert "calibration_not_degraded" in failed


# --------------------------------------------------------------------------
def test_g2_fails_when_anatomy_genuinely_does_not_help():
    """The headline negative control: the topology is an unrelated random graph."""
    d = make_graph_dataset(seed=2, anatomy_is_true=False, n_train=120, n_test=400)
    rep = run_g2(
        train=d["train"], test=d["test"], ood=d["ood"],
        model_for_graph=lambda A: RidgeGaussian(mask=A),
        anatomy=d["anatomy"], controls=d["controls"],
        causal_holdout={"train": d["train"], "test": d["ood"]},
        thresholds=FIXTURE_THRESHOLDS, seed=0,
    )
    assert rep.status == "FAIL"
    assert rep.consequence == (
        "Demote anatomy from compiled constraint to weak prior for the affected scale."
    )
    failed = {s.name for s in rep.subchecks if s.status == "FAIL"}
    assert "data_efficiency" in failed
    assert "beats_equal_capacity_controls" in failed


def test_g2_fails_when_the_residual_absorbs_the_topology_error():
    """Second thesis falsifier: correct anatomy, but a residual that repairs it.

    The graph is genuinely informative here; what fails is the *claim that the
    topology is load-bearing*, because corrupting half the edges costs nothing.
    """
    d = make_graph_dataset(seed=2, anatomy_is_true=True, n_train=400, n_test=400)
    rep = run_g2(
        train=d["train"], test=d["test"], ood=d["ood"],
        model_for_graph=lambda A: HybridMechResidual(mask=A, residual_strength=1.0,
                                                     alpha=0.01),
        anatomy=d["anatomy"], controls=d["controls"],
        causal_holdout={"train": d["train"], "test": d["ood"]},
        data_efficiency_sizes=[200, 400], n_efficiency_seeds=2,
        thresholds=FIXTURE_THRESHOLDS, seed=0,
    )
    assert rep.status == "FAIL"
    absorption = next(s for s in rep.subchecks if s.name == "residual_absorption")
    assert absorption.status == "FAIL"
    delta = rep.artifacts["absorption"]["delta"]
    assert abs(delta) < 1e-3, "corrupting the topology cost essentially nothing"


# --------------------------------------------------------------------------
def _g3(model, *, fine_structure: bool, thresholds: Thresholds = FIXTURE_THRESHOLDS):
    d = make_multiscale_dataset(seed=3, fine_structure=fine_structure)
    return run_g3(
        fine_train=d["fine_train"], fine_test=d["fine_test"],
        coarse_train=d["coarse_train"], coarse_test=d["coarse_test"],
        restriction=d["restriction"], multires_model=model,
        coarse_only_model=lambda: RidgeGaussian(name="coarse_only",
                                                blocks=["coarse_evidence"]),
        compute_full_fine=100.0, compute_adaptive=40.0,
        thresholds=thresholds, seed=0,
    )


def test_g3_fails_when_there_is_nothing_below_the_parcel():
    rep = _g3(lambda: RidgeGaussian(name="multires",
                                    blocks=["coarse_evidence", "fine_evidence"]),
              fine_structure=False)
    assert rep.status == "FAIL"
    assert rep.consequence.startswith("Disable the scale relation")
    failed = {s.name for s in rep.subchecks if s.status == "FAIL"}
    assert "native_scale_prediction" in failed


def test_g3_fails_on_high_frequency_hallucination():
    """Fine detail emitted with the fine evidence withheld, at unchanged confidence."""
    rep = _g3(lambda: HallucinatingFineModel(blocks=["coarse_evidence", "fine_evidence"]),
              fine_structure=True)
    assert rep.status == "FAIL"
    halluc = next(s for s in rep.subchecks if s.name == "high_frequency_hallucination")
    assert halluc.status == "FAIL"
    assert rep.artifacts["hallucination"]["index"] > 1.25


# --------------------------------------------------------------------------
def test_g4_fails_when_the_perturbation_only_informs_the_field_model():
    f = SyntheticFisher(nuisance_only_gain=True)
    rep = run_g4(fisher=f, theta_index=f.theta_index, nuisance_index=f.nuisance_index,
                 recovery=RECOVERY_OK, model_evidence=EVIDENCE_OK, seed=0)
    assert rep.status == "FAIL"
    assert rep.consequence.startswith("Narrow the identifiable parameter set")
    fisher_sub = next(s for s in rep.subchecks if s.name == "fisher_rank_and_eigenvalue")
    assert fisher_sub.status == "FAIL"
    art = rep.artifacts["fisher"]
    assert art["theta_min_eig_intervention"] == pytest.approx(art["theta_min_eig_base"])
    assert art["nuisance_min_eig_gain"] > 0  # the gain went entirely to the nuisance block


def test_g4_fails_when_the_intervention_does_not_separate_model_classes():
    f = SyntheticFisher()
    flat = {"joint_native": {"m1": -1.00, "m2": -1.01},
            "joint_plus_impulse": {"m1": -0.90, "m2": -0.91}}
    rep = run_g4(fisher=f, theta_index=f.theta_index, nuisance_index=f.nuisance_index,
                 recovery=RECOVERY_OK, model_evidence=flat, seed=0)
    assert rep.status == "FAIL"
    disc = next(s for s in rep.subchecks if s.name == "model_discrimination")
    assert disc.status == "FAIL"


def test_g4_fails_when_a_parameter_is_not_recovered():
    f = SyntheticFisher()
    bad = dict(RECOVERY_OK)
    bad["delay"] = {"true": 1.0, "estimate": 2.5, "lo": 2.2, "hi": 2.8}
    rep = run_g4(fisher=f, theta_index=f.theta_index, nuisance_index=f.nuisance_index,
                 recovery=bad, model_evidence=EVIDENCE_OK, seed=0)
    assert rep.status == "FAIL"
    rec = next(s for s in rep.subchecks if s.name == "prospective_recovery")
    assert rec.status == "FAIL"


# --------------------------------------------------------------------------
def _g5(*, individual_effect: bool, anatomy_predicts: bool = False):
    d = make_individualization_dataset(seed=5, individual_effect=individual_effect,
                                       anatomy_predicts=anatomy_predicts)
    cand, base = g5_arms()
    util = decision_problem(d["train"], d["new_session"], cand, base, seed=0)
    cand, base = g5_arms()
    return run_g5(train=d["train"], new_session=d["new_session"],
                  unseen_task=d["unseen_task"], candidate=cand, baselines=base,
                  utility=util, thresholds=FIXTURE_THRESHOLDS, seed=0)


def test_g5_fails_when_subjects_differ_only_by_noise():
    rep = _g5(individual_effect=False)
    assert rep.status == "FAIL"
    assert rep.consequence.startswith("Do not label the model an individual twin")
    failed = {s.name for s in rep.subchecks if s.status == "FAIL"}
    assert "incremental_log_score_new_session" in failed


def test_g5_fails_when_the_scan_is_doing_the_work():
    """'Including the person's scan is not itself individualization.'"""
    rep = _g5(individual_effect=False, anatomy_predicts=True)
    assert rep.status == "FAIL"
    scan = next(s for s in rep.subchecks if s.name == "scan_is_not_personalization")
    assert scan.status == "FAIL"
    assert "not itself individualization" in scan.falsified_by


# --------------------------------------------------------------------------
# G4 must name the modality-additivity identity rather than exploit it
# --------------------------------------------------------------------------
def test_g4_declares_the_modality_additivity_identity():
    """`I_{EEG+BOLD} = I_EEG + I_BOLD` is arithmetic, not a result."""
    f = SyntheticFisher()
    rep = run_g4(fisher=f, theta_index=f.theta_index, nuisance_index=f.nuisance_index,
                 recovery=RECOVERY_OK, model_evidence=EVIDENCE_OK,
                 single_modality_designs=("eeg", "fmri"), seed=0)
    decl = next(s for s in rep.subchecks if s.name == "modality_additivity_declaration")
    assert decl.mandatory is False, "an identity must never gate a claim"
    resid = next(m for m in decl.metrics
                 if m.name == "additivity.block_diagonal_residual")
    assert resid.value < 1e-9, "the fixture is block-diagonal by construction"
    assert "IDENTITY" in resid.note
    assert "NOT reported by this gate as evidence" in resid.note
    # and the gate's own verdict does not depend on it
    assert rep.status == "PASS"


def test_g4_reports_zero_non_additive_joint_information_as_zero():
    """A whitened map with no cross-covariance content must show excess = 0."""
    f = SyntheticFisher()
    rep = run_g4(fisher=f, fisher_whitened=f,           # identical -> no excess
                 theta_index=f.theta_index, nuisance_index=f.nuisance_index,
                 recovery=RECOVERY_OK, model_evidence=EVIDENCE_OK, seed=0)
    excess = next(m for s in rep.subchecks for m in s.metrics
                  if m.name == "additivity.joint_content_beyond_sum")
    assert excess.value == pytest.approx(0.0, abs=1e-12)
    assert excess.passed is False, \
        "zero non-additive information means typed fusion added nothing here"
    assert "falsifiable part of the fusion claim" in excess.note


def test_g4_records_non_additive_joint_information_when_it_exists():
    class _Whitened(SyntheticFisher):
        def __call__(self, design: str) -> np.ndarray:
            M = SyntheticFisher.__call__(self, design)
            if design.startswith("joint"):
                M = M.copy()
                M[0, 1] = M[1, 0] = 0.25    # cross-covariance content
            return M

    f = SyntheticFisher()
    w = _Whitened()
    rep = run_g4(fisher=f, fisher_whitened=w, theta_index=f.theta_index,
                 nuisance_index=f.nuisance_index, recovery=RECOVERY_OK,
                 model_evidence=EVIDENCE_OK, seed=0)
    excess = next(m for s in rep.subchecks for m in s.metrics
                  if m.name == "additivity.joint_content_beyond_sum")
    assert excess.value > 0.0 and excess.passed is True


def test_g4_states_its_basis_on_every_eigenvalue_metric():
    f = SyntheticFisher()
    rep = run_g4(fisher=f, theta_index=f.theta_index, nuisance_index=f.nuisance_index,
                 recovery=RECOVERY_OK, model_evidence=EVIDENCE_OK,
                 basis="prior_standardised", seed=0)
    assert rep.manifest.acceptance_thresholds["basis"] == "prior_standardised"
    assert rep.artifacts["fisher"]["basis"] == "prior_standardised"
    for name in ("fisher.theta_min_eigenvalue_gain",
                 "fisher.theta_condition_number_ratio"):
        m = next(mm for s in rep.subchecks for mm in s.metrics if mm.name == name)
        assert "basis=prior_standardised" in m.note
    md = rep.to_markdown()
    assert "prior_standardised" in md
    assert "IDENTITY, NOT RESULT" in md


def test_g4_falsifiable_comparison_is_the_design_contrast_not_the_modality_sum():
    """The gate fails on a null intervention even though joint>=single still holds."""
    f = SyntheticFisher(nuisance_only_gain=True)
    rep = run_g4(fisher=f, theta_index=f.theta_index, nuisance_index=f.nuisance_index,
                 recovery=RECOVERY_OK, model_evidence=EVIDENCE_OK, seed=0)
    assert rep.status == "FAIL"
    decl = next(s for s in rep.subchecks if s.name == "modality_additivity_declaration")
    assert decl.status == "PASS"      # the identity still holds...
    fisher_sub = next(s for s in rep.subchecks if s.name == "fisher_rank_and_eigenvalue")
    assert fisher_sub.status == "FAIL"   # ...and the gate still fails
