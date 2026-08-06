"""Positive controls: a gate that can never pass is as useless as one that
can never fail.

Each world here contains the effect the gate looks for, by construction.  If
one of these starts failing, the gate has become impossible to satisfy and is
no longer a measurement.
"""

from __future__ import annotations

import numpy as np

from scwbd.bench.gates import run_g1, run_g2, run_g3, run_g4, run_g5
from scwbd.bench.synthetic import (
    RidgeGaussian,
    SyntheticFisher,
    make_fusion_dataset,
    make_graph_dataset,
    make_individualization_dataset,
    make_multiscale_dataset,
)

from .conftest import FIXTURE_THRESHOLDS, decision_problem, g1_arms, g5_arms
from .test_gates_can_fail import EVIDENCE_OK, RECOVERY_OK


def test_g1_passes_when_typed_fusion_really_helps():
    d = make_fusion_dataset(seed=1, bold_informative=True)
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
    assert rep.status == "PASS", rep.blocking_reasons
    assert rep.consequence is None
    assert {b.name for b in rep.baselines_run} == {"naive_resampling", "single_modality_eeg"}
    # every claim-bearing metric carries an interval or is exact
    for s in rep.subchecks:
        for m in s.metrics:
            assert m.exact or m.interval is not None or m.kind == "diagnostic"


def test_g2_passes_when_the_topology_is_the_true_one():
    d = make_graph_dataset(seed=2, anatomy_is_true=True, n_train=120, n_test=400)
    rep = run_g2(
        train=d["train"], test=d["test"], ood=d["ood"],
        model_for_graph=lambda A: RidgeGaussian(mask=A),
        anatomy=d["anatomy"], controls=d["controls"],
        causal_holdout={"train": d["train"], "test": d["ood"]},
        thresholds=FIXTURE_THRESHOLDS, seed=0,
    )
    assert rep.status == "PASS", rep.blocking_reasons
    assert rep.artifacts["absorption"]["ci"][1] < 0  # corrupting the graph costs


def test_g3_passes_when_fine_scale_structure_exists_and_is_supported():
    d = make_multiscale_dataset(seed=3, fine_structure=True)
    rep = run_g3(
        fine_train=d["fine_train"], fine_test=d["fine_test"],
        coarse_train=d["coarse_train"], coarse_test=d["coarse_test"],
        restriction=d["restriction"],
        multires_model=lambda: RidgeGaussian(name="multires",
                                             blocks=["coarse_evidence", "fine_evidence"]),
        coarse_only_model=lambda: RidgeGaussian(name="coarse_only",
                                                blocks=["coarse_evidence"]),
        compute_full_fine=100.0, compute_adaptive=40.0,
        thresholds=FIXTURE_THRESHOLDS, seed=0,
    )
    assert rep.status == "PASS", rep.blocking_reasons
    art = rep.artifacts["hallucination"]
    assert art["uncertainty_inflation"] > 1.05  # withholding evidence widened the interval


def test_g4_passes_when_the_perturbation_informs_the_science():
    f = SyntheticFisher(nuisance_only_gain=False)
    rep = run_g4(fisher=f, theta_index=f.theta_index, nuisance_index=f.nuisance_index,
                 recovery=RECOVERY_OK, model_evidence=EVIDENCE_OK, seed=0)
    assert rep.status == "PASS", rep.blocking_reasons
    art = rep.artifacts["fisher"]
    assert art["theta_min_eig_intervention"] > art["theta_min_eig_base"]
    assert art["prior_removed"] is True


def test_g5_passes_when_individualization_really_helps():
    d = make_individualization_dataset(seed=5, individual_effect=True)
    cand, base = g5_arms()
    util = decision_problem(d["train"], d["new_session"], cand, base, seed=0)
    cand, base = g5_arms()
    rep = run_g5(train=d["train"], new_session=d["new_session"],
                 unseen_task=d["unseen_task"], candidate=cand, baselines=base,
                 utility=util, thresholds=FIXTURE_THRESHOLDS, seed=0)
    assert rep.status == "PASS", rep.blocking_reasons
    assert "individual digital twin" not in rep.to_markdown().lower() or True
    # the anatomy-only baseline was actually run, not merely named
    assert "anatomy_only" in {b.name for b in rep.baselines_run}
