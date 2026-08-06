"""§11.4 ablations: capacity matching, systematic error, and the smoothing rule."""

from __future__ import annotations

import numpy as np
import pytest

from scwbd.bench.ablations import ABLATIONS, AblationSpec, run_ablation, run_all_ablations
from scwbd.bench.gates import Thresholds
from scwbd.bench.synthetic import (
    RidgeGaussian,
    SmoothedModel,
    make_graph_dataset,
)

from .conftest import FIXTURE_THRESHOLDS


def test_every_11_4_bullet_has_a_registry_entry():
    clauses = " ".join(s.thesis_clause for s in ABLATIONS.values()).lower()
    for phrase in [
        "structured regional state",
        "hybrid local field",
        "single-resolution",
        "randomized, and distance-matched topology",
        "anatomically typed operators",
        "region-specific pretraining",
        "session-adapted",
        "language-only behavioural imitation",
        "teacher",
        "correlation fitting versus held-out perturbational prediction",
    ]:
        assert phrase in clauses, f"§11.4 bullet not represented: {phrase}"


def test_bare_run_is_all_could_not_run():
    reports = run_all_ablations()
    assert len(reports) == len(ABLATIONS)
    assert {r.status for r in reports} == {"COULD_NOT_RUN"}


def test_quarantined_teacher_ablation_is_off_by_default():
    rep = run_ablation(ABLATIONS["A9_teacher_quarantined"])
    assert rep.status == "COULD_NOT_RUN"
    assert any("off by default" in r for r in rep.blocking_reasons)


def test_topology_ablation_runs_and_reports_variance_and_systematic_error():
    d = make_graph_dataset(seed=11, anatomy_is_true=True, n_train=200, n_test=400)
    A, C = d["anatomy"], d["controls"]
    soft = np.clip(A + 0.4 * (C["distance_matched"] - A), 0, 1) > 0.3
    arms = {
        "hard": RidgeGaussian(name="hard", mask=A),
        "soft": RidgeGaussian(name="soft", mask=soft),
        "learned": RidgeGaussian(name="learned", mask=np.ones_like(A)),
        "randomized": RidgeGaussian(name="randomized", mask=C["randomized"]),
        "distance_matched": RidgeGaussian(name="distance_matched",
                                          mask=C["distance_matched"]),
    }
    rep = run_ablation("A4_topology", train=d["train"], test=d["test"], arms=arms,
                       thresholds=FIXTURE_THRESHOLDS, seed=0)
    names = {s.name for s in rep.subchecks}
    assert {"variance_reported", "systematic_error_reported", "smoothing_not_preferred",
            "candidate_beats_controls", "matched_capacity"} <= names
    assert rep.status in ("PASS", "FAIL")
    assert "per_arm" in rep.artifacts and "systematic_error" in rep.artifacts
    for arm, row in rep.artifacts["systematic_error"].items():
        assert row["status"] in ("design_estimable", "externally_bounded",
                                 "prior_specified_sensitivity")


def test_ablation_refuses_to_report_without_systematic_error():
    """No strata and no external bound -> systematic error is not estimable."""
    d = make_graph_dataset(seed=11, n_train=120, n_test=200)
    test = d["test"]
    test.strata = {}
    arms = {"typed_operators": RidgeGaussian(name="typed", mask=d["anatomy"]),
            "generic_equal_parameter": RidgeGaussian(name="generic", mask=d["anatomy"])}
    rep = run_ablation("A5_typed_operators", train=d["train"], test=test, arms=arms,
                       thresholds=FIXTURE_THRESHOLDS, seed=0)
    sub = next(s for s in rep.subchecks if s.name == "systematic_error_reported")
    assert sub.status == "COULD_NOT_RUN"
    assert "prior-specified sensitivity" in sub.reason
    assert rep.status != "PASS"


def test_smoothing_rule_fires_when_the_winning_arm_smoothed_away_the_effect():
    """§11.4: a lower-variance model is not preferred when it wins by attenuation."""
    d = make_graph_dataset(seed=7, n_train=20, n_test=300, n_regions=10, density=0.5,
                           noise=1.5)
    inner = RidgeGaussian(name="inner", alpha=0.05)
    arms = {
        "typed_operators": SmoothedModel(inner=inner, shrink=0.9, sd_scale=2.0),
        "generic_equal_parameter": RidgeGaussian(name="generic", alpha=0.05),
    }
    arms["typed_operators"].name = "typed_operators"
    rep = run_ablation("A5_typed_operators", train=d["train"], test=d["test"], arms=arms,
                       thresholds=FIXTURE_THRESHOLDS, seed=0)
    smoothing = next(s for s in rep.subchecks if s.name == "smoothing_not_preferred")
    assert smoothing.status == "FAIL", rep.artifacts["smoothing"]
    assert "REJECT-PREFERENCE" in smoothing.reason
    assert rep.status == "FAIL"
    # the winning arm won on raw score and is still refused
    per = rep.artifacts["per_arm"]
    assert per["typed_operators"]["log_score"] > per["generic_equal_parameter"]["log_score"]


def test_mechanistic_ablation_cannot_pass_without_a_mechanism_holdout():
    d = make_graph_dataset(seed=12, n_train=200, n_test=300)
    arms = {"typed_operators": RidgeGaussian(name="typed", mask=d["anatomy"]),
            "generic_equal_parameter": RidgeGaussian(name="generic", mask=d["anatomy"])}
    rep = run_ablation("A5_typed_operators", train=d["train"], test=d["test"], arms=arms,
                       thresholds=FIXTURE_THRESHOLDS, seed=0)
    sub = next(s for s in rep.subchecks if s.name == "mechanism_uniquely_supported")
    assert sub.status == "COULD_NOT_RUN"
    assert rep.status != "PASS"


def test_missing_arm_names_the_missing_arm():
    d = make_graph_dataset(seed=13, n_train=100, n_test=100)
    arms = {"hard": RidgeGaussian(mask=d["anatomy"])}
    rep = run_ablation("A4_topology", train=d["train"], test=d["test"], arms=arms, seed=0)
    assert rep.status == "COULD_NOT_RUN"
    joined = " ".join(rep.blocking_reasons)
    for missing in ("soft", "learned", "randomized", "distance_matched"):
        assert missing in joined


def test_ablation_consequences_are_concrete():
    for spec in ABLATIONS.values():
        assert len(spec.consequence) > 40
        assert spec.consequence.strip().endswith(".")
