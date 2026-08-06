"""``A_safe`` as a refusal mechanism, R11, and the deferral path.

SIMULATION ONLY.  Nothing in this module or its subject authorises stimulating
anybody.  These tests check that the feasible set *blocks* an optimizer, that
the limits cannot be learned, and that the controller can answer ``Defer`` or
``NoRecommendation`` instead of a number.
"""

from __future__ import annotations


import pytest
import torch

from scwbd.intervene.safety import (
    DEFAULT_LIMITS_PATH,
    CompilerRefusal,
    Defer,
    FeasibleSet,
    NoRecommendation,
    ProposedIntervention,
    RiskSensitiveController,
    SafetyLimits,
    SimulatedRanking,
)

_DT = torch.float64


def _limits() -> SafetyLimits:
    return SafetyLimits.load()


def _ok(label: str = "pose_a", **exposure) -> ProposedIntervention:
    base = {"tms.peak_efield_v_per_m": 95.0, "tms.pulses_per_session": 600.0,
            "tms.coil_scalp_distance_mm": 4.0}
    base.update(exposure)
    return ProposedIntervention(
        label=label, modality="tms", exposure=base, pose_certified=True, reversible=True
    )


# ---------------------------------------------------------------------------
# the limits file
# ---------------------------------------------------------------------------


def test_limits_load_from_a_declarative_citable_file():
    lim = _limits()
    assert DEFAULT_LIMITS_PATH.exists()
    assert len(lim.keys()) >= 10
    assert all(lim.get(k).citation for k in lim.keys())
    assert all(lim.get(k).basis for k in lim.keys())
    assert lim.meta["scope"] == "simulation_only"
    assert lim.meta["human_use_authorized"] is False


def test_limits_cannot_be_mutated():
    lim = _limits()
    with pytest.raises(CompilerRefusal) as e:
        lim.some_new_attribute = 1.0
    assert e.value.code == "R11"


def test_limits_expose_no_learnable_parameters():
    with pytest.raises(CompilerRefusal, match="outside the learned objective"):
        _limits().parameters()


def test_an_undeclared_exposure_axis_is_refused():
    fs = FeasibleSet()
    p = ProposedIntervention(
        label="x", modality="tms", exposure={"tms.made_up_axis": 1.0},
        pose_certified=True,
    )
    with pytest.raises(CompilerRefusal, match="no declared limit"):
        fs.contains(p)


def test_a_limits_file_claiming_human_authorisation_is_refused(tmp_path):
    src = DEFAULT_LIMITS_PATH.read_text()
    bad = tmp_path / "bad.toml"
    bad.write_text(src.replace("human_use_authorized = false", "human_use_authorized = true"))
    with pytest.raises(CompilerRefusal, match="no ethics approval"):
        SafetyLimits.load(bad)


def test_an_uncited_limit_is_refused(tmp_path):
    bad = tmp_path / "uncited.toml"
    bad.write_text(
        'schema_version = "x"\nhuman_use_authorized = false\n\n'
        "[tms.peak_efield_v_per_m]\nmax = 200.0\nunits = \"V/m\"\n"
    )
    with pytest.raises(CompilerRefusal, match="no citation"):
        SafetyLimits.load(bad)


def test_the_feasible_set_has_no_score_or_penalty_method():
    """Turning A_safe into a soft penalty is the failure mode R11 names."""
    fs = FeasibleSet()
    for forbidden in ("score", "penalty", "cost", "loss", "soften"):
        assert not hasattr(fs, forbidden)


# ---------------------------------------------------------------------------
# R11
# ---------------------------------------------------------------------------


def test_a_feasible_proposal_passes():
    fs = FeasibleSet()
    v = fs.contains(_ok())
    assert v.feasible
    assert not v.violations
    assert fs.guard(_ok()).label == "pose_a"


@pytest.mark.parametrize(
    "axis,value",
    [
        ("tms.peak_efield_v_per_m", 900.0),
        ("tms.pulses_per_session", 100000.0),
        ("tms.coil_scalp_distance_mm", 90.0),
    ],
)
def test_r11_fires_when_a_proposal_leaves_a_safe(axis, value):
    fs = FeasibleSet()
    with pytest.raises(CompilerRefusal) as e:
        fs.guard(_ok(**{axis: value}))
    assert e.value.code == "R11"
    assert axis in str(e.value)
    assert e.value.violations[0].limit.citation


def test_an_uncertified_pose_is_outside_a_safe():
    fs = FeasibleSet()
    p = ProposedIntervention(
        label="scalp_label_only", modality="tms",
        exposure={"tms.peak_efield_v_per_m": 95.0}, pose_certified=False,
    )
    v = fs.contains(p)
    assert not v.feasible
    assert any("pose_certified" in str(x) for x in v.violations)


def test_r11_fires_when_an_optimizer_proposes_outside_a_safe():
    """The headline refusal: a learned objective wants a bigger field."""
    fs = FeasibleSet()
    state = {"gain": 1.0}

    def greedy_optimizer() -> ProposedIntervention:
        state["gain"] *= 4.0  # a learned objective monotone in field strength
        return _ok(label="greedy", **{"tms.peak_efield_v_per_m": 95.0 * state["gain"]})

    with pytest.raises(CompilerRefusal) as e:
        fs.guard_optimizer(greedy_optimizer, max_attempts=3)
    assert e.value.code == "R11"
    # no silent repair: the optimizer is not projected back into the set
    assert "leaves A_safe" in str(e.value)


def test_unchecked_declared_axes_are_reported_not_assumed_satisfied():
    fs = FeasibleSet()
    p = ProposedIntervention(
        label="partial", modality="tms",
        exposure={"tms.peak_efield_v_per_m": 95.0}, pose_certified=True,
    )
    v = fs.contains(p)
    assert v.feasible  # nothing declared was violated ...
    assert v.unchecked_declared_axes  # ... but the silence is recorded
    assert "tms.pulses_per_session" in v.unchecked_declared_axes


def test_a_simulated_ranking_can_never_be_marked_authorised():
    with pytest.raises(CompilerRefusal):
        SimulatedRanking(
            ordered_labels=("a",), objective_values=torch.zeros(1, dtype=_DT),
            benefit_gap=1.0, epistemic_uncertainty=0.0, limits_citations=(),
            human_use_authorized=True,
        )


# ---------------------------------------------------------------------------
# the controller
# ---------------------------------------------------------------------------


def _controller() -> RiskSensitiveController:
    return RiskSensitiveController(gamma=1.0)


def _candidates(n: int = 3):
    return [_ok(label=f"pose_{i}") for i in range(n)]


def test_controller_does_not_defer_when_the_winner_is_the_agreed_one():
    """Disagreement about a candidate the objective already rejects is not a
    reason to defer -- the deferral rule keys on the *top* candidate."""
    c = _controller()
    benefit = torch.tensor([[0.0, 0.5, 1.0], [1.0, 0.5, 0.02]], dtype=_DT)
    out = c.decide(
        _candidates(3),
        benefit=benefit,
        burden=torch.zeros(3, dtype=_DT),
        harm=torch.zeros(2, 3, dtype=_DT),
        model_log_weights=torch.zeros(2, dtype=_DT),
    )
    assert isinstance(out, SimulatedRanking)
    assert out.ordered_labels[0] == "pose_1"  # the pose both models agree on
    assert out.epistemic_uncertainty == pytest.approx(0.0, abs=1e-12)


def test_controller_ranks_admissible_candidates_when_models_agree():
    c = _controller()
    cands = _candidates(3)
    benefit = torch.tensor([[0.2, 0.6, 1.0], [0.21, 0.61, 1.02]], dtype=_DT)
    out = c.decide(
        cands,
        benefit=benefit,
        burden=torch.zeros(3, dtype=_DT),
        harm=torch.zeros(2, 3, dtype=_DT),
        model_log_weights=torch.zeros(2, dtype=_DT),
    )
    assert isinstance(out, SimulatedRanking)
    assert out.ordered_labels[0] == "pose_2"
    assert out.human_use_authorized is False
    assert out.limits_citations
    assert out.benefit_gap > 0.0


def test_controller_defers_when_model_disagreement_dominates_the_benefit_gap():
    """thesis_contract Sec. 0.5 step 6, made executable."""
    c = _controller()
    cands = _candidates(3)
    # two models that fit passive data equally well but disagree about which
    # pose helps; the benefit difference is small next to the disagreement
    benefit = torch.tensor([[1.0, 0.05, 0.0], [0.0, 0.95, 0.0]], dtype=_DT)
    out = c.decide(
        cands,
        benefit=benefit,
        burden=torch.zeros(3, dtype=_DT),
        harm=torch.zeros(2, 3, dtype=_DT),
        model_log_weights=torch.zeros(2, dtype=_DT),
    )
    assert isinstance(out, Defer)
    assert out.suggested_action == "reversible_probe"
    assert "dominates the benefit difference" in out.reason
    assert out.detail["epistemic"] >= out.detail["benefit_gap"]


def test_controller_defers_to_calibration_when_no_reversible_probe_exists():
    c = _controller()
    out = c.decide(
        _candidates(3),
        benefit=torch.tensor([[1.0, 0.05, 0.0], [0.0, 0.95, 0.0]], dtype=_DT),
        burden=torch.zeros(3, dtype=_DT),
        harm=torch.zeros(2, 3, dtype=_DT),
        model_log_weights=torch.zeros(2, dtype=_DT),
        reversible_probe_available=False,
    )
    assert isinstance(out, Defer)
    assert out.suggested_action == "additional_calibration_measurement"


def test_controller_defers_when_transform_uncertainty_dominates():
    """Pose-chain uncertainty alone is enough to withhold a recommendation."""
    c = _controller()
    benefit = torch.tensor([[0.2, 0.6, 1.0], [0.21, 0.61, 1.02]], dtype=_DT)
    kwargs = dict(
        benefit=benefit,
        burden=torch.zeros(3, dtype=_DT),
        harm=torch.zeros(2, 3, dtype=_DT),
        model_log_weights=torch.zeros(2, dtype=_DT),
    )
    assert isinstance(c.decide(_candidates(3), **kwargs), SimulatedRanking)
    out = c.decide(
        _candidates(3),
        transform_uncertainty=torch.tensor([2.0, 2.0, 2.0], dtype=_DT),
        **kwargs,
    )
    assert isinstance(out, Defer)


def test_controller_refuses_to_compare_under_a_single_response_model():
    c = _controller()
    out = c.decide(
        _candidates(2),
        benefit=torch.tensor([[0.2, 1.0]], dtype=_DT),
        burden=torch.zeros(2, dtype=_DT),
        harm=torch.zeros(1, 2, dtype=_DT),
        model_log_weights=torch.zeros(1, dtype=_DT),
    )
    assert isinstance(out, Defer)
    assert "unresolved" in out.reason


def test_controller_returns_no_recommendation_when_everything_is_infeasible():
    c = _controller()
    bad = [
        ProposedIntervention(
            label=f"bad_{i}", modality="tms",
            exposure={"tms.peak_efield_v_per_m": 5000.0}, pose_certified=True,
        )
        for i in range(2)
    ]
    out = c.decide(
        bad,
        benefit=torch.tensor([[1.0, 2.0], [1.0, 2.0]], dtype=_DT),
        burden=torch.zeros(2, dtype=_DT),
        harm=torch.zeros(2, 2, dtype=_DT),
        model_log_weights=torch.zeros(2, dtype=_DT),
    )
    assert isinstance(out, NoRecommendation)
    assert "outside A_safe" in out.reason


def test_infeasible_candidates_never_get_a_score_to_trade_against():
    """A_safe is applied BEFORE the objective, so a huge benefit cannot buy in."""
    c = _controller()
    cands = [
        _ok(label="safe_but_modest"),
        ProposedIntervention(
            label="unsafe_but_wonderful", modality="tms",
            exposure={"tms.peak_efield_v_per_m": 5000.0}, pose_certified=True,
        ),
        _ok(label="safe_alternative", **{"tms.peak_efield_v_per_m": 90.0}),
    ]
    benefit = torch.tensor([[0.2, 1000.0, 0.9], [0.21, 1001.0, 0.92]], dtype=_DT)
    out = c.decide(
        cands,
        benefit=benefit,
        burden=torch.zeros(3, dtype=_DT),
        harm=torch.zeros(2, 3, dtype=_DT),
        model_log_weights=torch.zeros(2, dtype=_DT),
    )
    assert isinstance(out, SimulatedRanking)
    assert "unsafe_but_wonderful" not in out.ordered_labels


def test_harm_is_averaged_once_and_epistemic_uncertainty_is_not_double_counted():
    """`L_harm` under the posterior; `U_epi` only for reducible disagreement."""
    c = RiskSensitiveController(beta=1.0, gamma=0.0)
    cands = _candidates(2)
    benefit = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=_DT)
    harm = torch.tensor([[0.0, 0.4], [0.0, 0.4]], dtype=_DT)
    out = c.decide(
        cands, benefit=benefit, burden=torch.zeros(2, dtype=_DT), harm=harm,
        model_log_weights=torch.zeros(2, dtype=_DT),
    )
    assert isinstance(out, SimulatedRanking)
    assert out.ordered_labels[0] == "pose_0"
    # harm subtracted exactly once: 1.0 - 0.4 vs 1.0 - 0.0
    assert out.benefit_gap == pytest.approx(0.4, rel=1e-9)
    # models agree here, so epistemic uncertainty is exactly zero
    assert out.epistemic_uncertainty == pytest.approx(0.0, abs=1e-12)


def test_no_candidates_yields_no_recommendation():
    out = _controller().decide(
        [],
        benefit=torch.zeros(2, 0, dtype=_DT),
        burden=torch.zeros(0, dtype=_DT),
        harm=torch.zeros(2, 0, dtype=_DT),
        model_log_weights=torch.zeros(2, dtype=_DT),
    )
    assert isinstance(out, NoRecommendation)


def test_every_controller_outcome_carries_the_simulation_only_notice():
    outs = [
        Defer(reason="r", suggested_action="no_action"),
        NoRecommendation(reason="r"),
        SimulatedRanking(("a",), torch.zeros(1, dtype=_DT), 1.0, 0.0, ()),
    ]
    for o in outs:
        assert "SIMULATION ONLY" in o.notice
