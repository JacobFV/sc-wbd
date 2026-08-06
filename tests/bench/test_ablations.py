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


# ---------------------------------------------------------------------------
# A1 run-2 preregistration (reports/ablations/PREREG_A1_run2.md, 2026-08-06).
# Filed while A1_structured_state is COULD_NOT_RUN and no heterogeneous arm
# exists.  These tests exist because a preregistration that lives only in prose
# is a document, and documents get edited.
# ---------------------------------------------------------------------------


def _regionally_differentiated(n=64, t=40, c=8, seed=0):
    """Channels with genuinely different temporal dynamics (amplitudes 1..c)."""
    rng = np.random.default_rng(seed)
    amp = np.linspace(1.0, float(c), c)
    return rng.standard_normal((n, t, c)) * amp


def _collapsed_to_shared_dynamic(y):
    """A1's failure mode: every channel gets the SAME temporal std.

    Global dynamic range is preserved exactly -- each channel is rescaled to the
    pooled temporal std -- while between-channel differentiation is destroyed.
    """
    per_ch = y.std(axis=1, keepdims=True)  # (n, 1, c)
    shared = per_ch.mean(axis=2, keepdims=True)  # (n, 1, 1)
    return y / np.maximum(per_ch, 1e-12) * shared


def test_A1_EFFECT_reads_the_failure_that_default_effect_cannot():
    """The reason A1 declares its own effect, demonstrated rather than asserted."""
    from scwbd.bench.ablations import A1_EFFECT, default_effect

    truth = _regionally_differentiated()
    collapsed = _collapsed_to_shared_dynamic(truth)

    # default_effect is BLIND: the collapse preserves global dynamic range.
    assert default_effect(collapsed) == pytest.approx(default_effect(truth), rel=0.05)

    # A1_EFFECT SEES it: between-region dispersion is destroyed.
    assert A1_EFFECT(truth) > 0.0
    assert A1_EFFECT(collapsed) < 0.05 * A1_EFFECT(truth)


def test_A1_EFFECT_is_not_trivially_zero_or_constant():
    """An effect that returns the same number for everything cannot discriminate."""
    from scwbd.bench.ablations import A1_EFFECT

    a = A1_EFFECT(_regionally_differentiated(seed=0))
    b = A1_EFFECT(_regionally_differentiated(c=8, seed=1) * 0.0 + 1.0)  # flat
    assert a > 0.0
    assert b == pytest.approx(0.0, abs=1e-9)
    assert a != b


def test_A1_refuses_to_run_with_the_default_effect():
    """The guard fires: A1 with default_effect is COULD_NOT_RUN, not a green check."""
    from scwbd.bench.ablations import A1_EFFECT, default_effect

    d = make_graph_dataset(seed=0, n_train=120, n_test=200)
    rep = run_ablation(
        "A1_structured_state",
        train=d["train"],
        test=d["test"],
        arms={a: RidgeGaussian() for a in ABLATIONS["A1_structured_state"].required_arms},
        effect=default_effect,
        thresholds=FIXTURE_THRESHOLDS,
    )
    assert rep.status == "COULD_NOT_RUN"
    reasons = " ".join(rep.blocking_reasons).lower()
    assert "effect" in reasons and "a1_effect" in reasons
    # the effect refusal must not HIDE the missing arms; both are reported
    assert any(s.name.startswith("arms[") for s in rep.subchecks) or "structured_state" in reasons

    # ...and the refusal is specific to the wrong effect, not a blanket block:
    # with A1_EFFECT supplied -- or with nothing supplied, since the registry
    # then provides its own -- the run never blocks on `effect_of_interest`.
    rep2 = run_ablation(
        "A1_structured_state",
        train=d["train"],
        test=d["test"],
        arms={a: RidgeGaussian() for a in ABLATIONS["A1_structured_state"].required_arms},
        effect=A1_EFFECT,
        thresholds=FIXTURE_THRESHOLDS,
    )
    assert not any(s.name == "effect_of_interest" for s in rep2.subchecks)

    rep3 = run_ablation(
        "A1_structured_state",
        train=d["train"],
        test=d["test"],
        arms={a: RidgeGaussian() for a in ABLATIONS["A1_structured_state"].required_arms},
        thresholds=FIXTURE_THRESHOLDS,
    )
    assert not any(s.name == "effect_of_interest" for s in rep3.subchecks)


def test_A1_registry_carries_the_run2_preregistration():
    """The prereg is imported by the scoring path, not only filed in reports/."""
    from scwbd.bench.ablations import A1_EFFECT, A1_RUN2_PREREGISTRATION

    spec = ABLATIONS["A1_structured_state"]
    assert spec.required_effect is A1_EFFECT
    assert spec.note == A1_RUN2_PREREGISTRATION
    # Both capacity-matching definitions are named; picking one after the fact
    # is the defect the two-control design exists to prevent.
    assert "pooled_vector_per_region@param_matched" in spec.required_arms
    assert "pooled_vector_per_region@state_matched" in spec.required_arms
    # Attribution: heterogeneity without the anatomical assignment.
    assert "permuted_family_state" in spec.required_arms
    for token in (
        "BOTH_MUST_BE_BEATEN",
        "NLL_WIN_WITHOUT_MSE_WIN_GRANTS_NO_MECHANISTIC_CLAIM",
        "V_ABLATION_AND_V_CLAIM_ARE_SEPARATE",
        "RUN1_IS_A_CONTROL_CLASS_ARTIFACT_NOT_RUN2S_CONTROL_ARM",
    ):
        assert token in A1_RUN2_PREREGISTRATION


def test_A1_blocks_on_the_budgets_its_preregistration_declared_binding():
    """PREREG_A1_run2 §3.1 B2/B3/B4 bind, and arms that do not declare them block.

    Parameter parity does NOT say two arms with structurally different state are
    matched, which is the whole difficulty of this ablation.  Arms that declare
    only `n_parameters` must therefore be COULD_NOT_RUN on capacity, not PASS.
    """
    spec = ABLATIONS["A1_structured_state"]
    assert spec.require_budgets == ("state_width", "train_steps", "n_configs_trained")

    d = make_graph_dataset(seed=0, n_train=120, n_test=200)
    rep = run_ablation(
        "A1_structured_state",
        train=d["train"],
        test=d["test"],
        arms={a: RidgeGaussian() for a in spec.required_arms},
        thresholds=FIXTURE_THRESHOLDS,
    )
    sub = next(s for s in rep.subchecks if s.name == "matched_capacity")
    assert sub.status == "COULD_NOT_RUN"
    assert "required budget fields not declared" in sub.reason
    for f in spec.require_budgets:
        assert f in sub.reason
    assert rep.status != "PASS"


def test_an_ablation_without_required_budgets_still_records_what_went_unchecked():
    """The green-row regression: silence must never read as coverage."""
    d = make_graph_dataset(seed=12, n_train=200, n_test=300)
    arms = {"typed_operators": RidgeGaussian(name="typed", mask=d["anatomy"]),
            "generic_equal_parameter": RidgeGaussian(name="generic", mask=d["anatomy"])}
    rep = run_ablation("A5_typed_operators", train=d["train"], test=d["test"], arms=arms,
                       thresholds=FIXTURE_THRESHOLDS, seed=0)
    sub = next(s for s in rep.subchecks if s.name == "matched_capacity")
    assert sub.status == "PASS"  # unchanged: undeclared fields do not fail an arm
    assert "NOT CHECKED" in sub.reason
    cap = rep.artifacts["capacity"]
    assert "flops" in cap["unchecked_fields"]
    assert cap["required_budgets"] == []


def test_A1_blocks_when_no_arm_path_is_supplied():
    """Budgets can match exactly while an arm is handicapped at the boundary."""
    spec = ABLATIONS["A1_structured_state"]
    assert spec.require_path_parity
    d = make_graph_dataset(seed=0, n_train=120, n_test=200)
    rep = run_ablation(
        "A1_structured_state", train=d["train"], test=d["test"],
        arms={a: RidgeGaussian() for a in spec.required_arms},
        thresholds=FIXTURE_THRESHOLDS,
    )
    sub = next(s for s in rep.subchecks if s.name == "path_parity")
    assert sub.status == "COULD_NOT_RUN"
    assert "observation boundary" in sub.reason


def test_A1_path_parity_fires_on_a_narrowed_observation_interface():
    """🌊 Hodgkin's defect, end to end through run_ablation."""
    from scwbd.bench.matching import ArmPath

    spec = ABLATIONS["A1_structured_state"]
    full = dict(
        observation_ports=(("eeg", (("rate_e", 1), ("rate_i", 1), ("spectral", 16))),),
        variance_model="state_dependent_logvar",
        calibration_protocol="as_emitted",
        score_metric="gaussian_nll_raw_units",
        split_fingerprint="5cfa14eb",
        context_length=64,
        input_normalisation="per_window_std",
        anatomy_provenance="schaefer400_real",
    )
    narrowed = dict(full)
    narrowed["observation_ports"] = (("eeg", (("rate_e", 1), ("rate_i", 1))),)

    d = make_graph_dataset(seed=0, n_train=120, n_test=200)
    paths = {a: ArmPath(**full) for a in spec.required_arms}
    paths["structured_state"] = ArmPath(**narrowed)  # the treatment arm, handicapped
    rep = run_ablation(
        "A1_structured_state", train=d["train"], test=d["test"],
        arms={a: RidgeGaussian() for a in spec.required_arms},
        arm_paths=paths, thresholds=FIXTURE_THRESHOLDS,
    )
    sub = next(s for s in rep.subchecks if s.name == "path_parity")
    assert sub.status == "COULD_NOT_RUN"
    assert any("observation_ports" in m for m in rep.artifacts["path_parity"]["mismatches"])
    assert rep.status != "PASS"

    # identical paths clear this subcheck (so it discriminates, not just blocks)
    ok = run_ablation(
        "A1_structured_state", train=d["train"], test=d["test"],
        arms={a: RidgeGaussian() for a in spec.required_arms},
        arm_paths={a: ArmPath(**full) for a in spec.required_arms},
        thresholds=FIXTURE_THRESHOLDS,
    )
    assert next(s for s in ok.subchecks if s.name == "path_parity").status == "PASS"


def test_A1s_preregistered_arms_are_REQUIRED_not_optional():
    """"Mandatory" in a report is prose; `required_arms` is the mechanism.

    Leaving the run-2 arms in `optional_arms` would have let A1 be scored with
    the two capacity-matched controls, the stage-2 conditioning control and the
    attribution control all absent -- while PREREG_A1_run2 §1 called them
    mandatory. That is decorative_guards row 11's failure inside my own registry.
    """
    spec = ABLATIONS["A1_structured_state"]
    for arm in (
        "pooled_vector_per_region@param_matched",
        "pooled_vector_per_region@state_matched",
        "theta_conditioned_pooled",
        "permuted_family_state",
    ):
        assert arm in spec.required_arms, f"{arm} is preregistered mandatory"
        assert arm not in spec.optional_arms

    # and a run missing one of them names it rather than proceeding
    d = make_graph_dataset(seed=3, n_train=80, n_test=80)
    partial = {a: RidgeGaussian() for a in spec.required_arms
               if a != "theta_conditioned_pooled"}
    rep = run_ablation("A1_structured_state", train=d["train"], test=d["test"],
                       arms=partial, thresholds=FIXTURE_THRESHOLDS)
    assert rep.status == "COULD_NOT_RUN"
    assert any("theta_conditioned_pooled" in r for r in rep.blocking_reasons)
