"""Matched-capacity accounting: an unmatched win is not a win."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pytest

from scwbd.bench.matching import Budget, budget_of, check_matched, matched_subcheck
from scwbd.bench.synthetic import RidgeGaussian, make_graph_dataset


@dataclass
class _Sized:
    n: int
    name: str = "sized"

    def n_parameters(self) -> int:
        return self.n


class _Opaque:
    name = "opaque"


def test_matched_when_budgets_agree():
    v = check_matched(_Sized(100), {"a": _Sized(100), "b": _Sized(105)}, tol=0.10)
    assert v.matched
    assert matched_subcheck(v).status == "PASS"


def test_over_budget_candidate_cannot_pass():
    v = check_matched(_Sized(300), {"a": _Sized(100)}, tol=0.10)
    assert not v.matched
    sub = matched_subcheck(v)
    assert sub.status == "COULD_NOT_RUN"
    assert "unmatched win is not a win" in sub.reason
    assert "3.00x" in sub.reason


def test_smaller_candidate_is_matched_and_flagged_favourable_to_the_null():
    v = check_matched(_Sized(50), {"a": _Sized(100)}, tol=0.10)
    assert v.matched and v.favourable_to_null
    assert matched_subcheck(v).status == "PASS"


def test_unknown_capacity_is_could_not_run_not_a_pass():
    v = check_matched(_Opaque(), {"a": _Sized(100)})
    sub = matched_subcheck(v)
    assert sub.status == "COULD_NOT_RUN"
    assert "capacity accounting unavailable" in sub.reason


def test_budget_of_reads_a_torch_module_if_present():
    torch = __import__("torch")
    m = torch.nn.Linear(4, 3)
    b = budget_of(m)
    assert b.n_parameters == 4 * 3 + 3
    assert b.source in ("n_parameters()", "torch.parameters()")


def test_reference_arms_report_the_parameters_they_actually_fit():
    d = make_graph_dataset(seed=0, n_regions=6, n_train=80, n_test=80, density=0.5)
    dense = RidgeGaussian(mask=np.ones((6, 6))).fit(d["train"])
    sparse = RidgeGaussian(mask=d["anatomy"]).fit(d["train"])
    assert dense.n_parameters() > sparse.n_parameters()
    v = check_matched(sparse, {"dense": dense}, tol=0.10)
    # the sparse candidate is smaller: the comparison is valid and favourable
    # to the null, which is exactly the direction §11.4 permits
    assert v.matched and v.favourable_to_null


# ---------------------------------------------------------------------------
# Every field Budget declares must BIND, not just n_parameters.
#
# Until 2026-08-06 check_matched compared n_parameters and nothing else, while
# ablations.py promised arms "at matched capacity AND compute".  A comparison
# could be compute-unmatched by any factor and still carry a green
# matched_capacity check.  One test per binding field, each demonstrated to
# FIRE -- a guard asserted but never shown to fire is the thing this file is
# about.
# ---------------------------------------------------------------------------


@dataclass
class _Budgeted:
    """An arm that declares a full budget."""

    n: int = 100
    flops_: float | None = None
    steps: int | None = None
    width: int | None = None
    configs: int | None = None
    secs: float | None = None
    name: str = "budgeted"

    def n_parameters(self) -> int:
        return self.n

    @property
    def compute_flops(self):  # noqa: D401 - duck-typed by budget_of
        return self.flops_

    @property
    def train_steps(self):
        return self.steps

    @property
    def state_width(self):
        return self.width

    @property
    def n_configs_trained(self):
        return self.configs

    @property
    def wall_seconds(self):
        return self.secs


@pytest.mark.parametrize(
    "kw, field_name",
    [
        ({"flops_": 1.0}, "flops"),
        ({"steps": 100}, "train_steps"),
        ({"width": 64}, "state_width"),
        ({"configs": 1}, "n_configs_trained"),
    ],
)
def test_each_binding_field_fires_when_the_candidate_is_over_budget(kw, field_name):
    """3x over on this field alone, parameters identical -> NOT matched."""
    (attr, val), = kw.items()
    cand = _Budgeted(n=100, **{attr: type(val)(val * 3)})
    base = _Budgeted(n=100, **kw)
    v = check_matched(cand, {"a": base}, tol=0.10)
    assert not v.matched, f"{field_name} did not bind"
    assert f"a.{field_name}" in v.over_budget_fields
    sub = matched_subcheck(v)
    assert sub.status == "COULD_NOT_RUN"
    assert field_name in sub.reason and "unmatched win is not a win" in sub.reason

    # ...and the same field WITHIN tolerance is matched, so the check is
    # discriminating rather than merely restrictive.
    ok = check_matched(_Budgeted(n=100, **kw), {"a": base}, tol=0.10)
    assert ok.matched


@pytest.mark.parametrize(
    "kw, field_name",
    [
        ({"flops_": 1.0}, "flops"),
        ({"steps": 100}, "train_steps"),
        ({"width": 64}, "state_width"),
        ({"configs": 1}, "n_configs_trained"),
    ],
)
def test_declared_on_one_side_only_is_a_defect_not_a_skip(kw, field_name):
    v = check_matched(_Budgeted(n=100, **kw), {"a": _Budgeted(n=100)}, tol=0.10)
    assert not v.matched
    assert f"a.{field_name}" in v.asymmetric
    assert "asymmetric" in matched_subcheck(v).reason


def test_wall_seconds_is_advisory_and_never_binds():
    """Reported, never enforced: wall-clock on a shared pool measures contention."""
    v = check_matched(
        _Budgeted(n=100, secs=900.0), {"a": _Budgeted(n=100, secs=1.0)}, tol=0.10
    )
    assert v.matched, "wall_seconds must not bind"
    assert v.advisory_ratios["wall_seconds"]["a"] == pytest.approx(900.0)
    assert any(
        m.name == "capacity.wall_seconds_ratio_vs_a" and "ADVISORY" in (m.note or "")
        for m in v.metrics()
    )
    # asymmetric wall_seconds is also tolerated, unlike every binding field
    assert check_matched(_Budgeted(n=100, secs=5.0), {"a": _Budgeted(n=100)}).matched


def test_a_passing_check_names_what_it_did_not_check():
    """The regression that made this fix necessary: silence read as coverage."""
    v = check_matched(_Sized(100), {"a": _Sized(100)}, tol=0.10)
    assert v.matched
    for f in ("flops", "train_steps", "state_width", "n_configs_trained"):
        assert f in v.unchecked_fields
    sub = matched_subcheck(v)
    assert sub.status == "PASS"
    assert "NOT CHECKED" in sub.reason and "flops" in sub.reason
    checked = next(m for m in v.metrics() if m.name == "capacity.binding_fields_checked")
    assert checked.value == 1.0


def test_require_turns_an_unchecked_field_into_a_blocker():
    """PREREG_A1_run2 §3.1: B2/B3/B4 are binding, so run 2 requires them."""
    v = check_matched(_Sized(100), {"a": _Sized(100)}, require=("state_width",))
    assert not v.matched
    assert v.undeclared_required == ["a.state_width", "sized.state_width"]
    assert "required budget fields not declared" in matched_subcheck(v).reason

    full = {"width": 64, "steps": 10, "configs": 1}
    ok = check_matched(
        _Budgeted(n=100, **full),
        {"a": _Budgeted(n=100, **full)},
        require=("state_width", "train_steps", "n_configs_trained"),
    )
    assert ok.matched and not ok.undeclared_required


def test_require_rejects_an_advisory_field_rather_than_pretending_to_enforce_it():
    with pytest.raises(ValueError, match="advisory by design"):
        check_matched(_Sized(100), {"a": _Sized(100)}, require=("wall_seconds",))


def test_budget_declared_lists_only_fields_carrying_a_number():
    assert budget_of(_Sized(100)).declared == ("n_parameters",)
    b = budget_of(_Budgeted(n=1, steps=2, secs=3.0))
    assert set(b.declared) == {"n_parameters", "train_steps", "wall_seconds"}


# ---------------------------------------------------------------------------
# PATH PARITY -- the second matching axis.
#
# 🌊 Hodgkin's finding, 2026-08-06, caught on his branch before it shipped: the
# A1 treatment arm's EEGHead saw ("rate_e","rate_i") = 2 dims against the
# control's ("rate_e","rate_i","spectral") = 18.  Every field Budget declares
# could match exactly.  A1 would have concluded heterogeneity does not help,
# with a green harness, because the treatment arm was handicapped at the
# OBSERVATION BOUNDARY rather than at the hypothesis.
# ---------------------------------------------------------------------------

from scwbd.bench.matching import (  # noqa: E402
    ArmPath,
    check_path_parity,
    parity_subcheck,
)

_FULL = dict(
    observation_ports=(("eeg", (("rate_e", 1), ("rate_i", 1), ("spectral", 16))),),
    variance_model="state_dependent_logvar",
    calibration_protocol="as_emitted",
    score_metric="gaussian_nll_raw_units",
    split_fingerprint="5cfa14eb",
    context_length=64,
    input_normalisation="per_window_std",
    anatomy_provenance="schaefer400_real",
)


def test_path_parity_passes_when_every_arm_presents_the_same_path():
    v = check_path_parity(
        {"structured_state": ArmPath(**_FULL), "pooled": ArmPath(**_FULL)},
        candidate="structured_state",
    )
    assert v.matched
    assert parity_subcheck(v).status == "PASS"


def test_hodgkins_narrowed_observation_interface_fires():
    """The exact defect: 2 dims vs 18, with every Budget field identical."""
    handicapped = dict(_FULL)
    handicapped["observation_ports"] = (("eeg", (("rate_e", 1), ("rate_i", 1))),)
    v = check_path_parity(
        {"structured_state": ArmPath(**handicapped), "pooled": ArmPath(**_FULL)},
        candidate="structured_state",
    )
    assert not v.matched
    assert any("observation_ports" in m for m in v.mismatches)
    sub = parity_subcheck(v)
    assert sub.status == "COULD_NOT_RUN"
    assert "OFF the manipulated variable" in sub.reason

    # ...and check_matched is blind to it, which is why this axis exists.
    b = check_matched(_Sized(100), {"pooled": _Sized(100)}, tol=0.10)
    assert b.matched


def test_same_total_width_but_different_typed_quantities_still_fires():
    """Width parity is not enough: 18 dims of the WRONG quantities is not parity."""
    other = dict(_FULL)
    other["observation_ports"] = (
        ("eeg", (("rate_e", 1), ("rate_i", 1), ("adaptation", 16))),
    )
    v = check_path_parity(
        {"structured_state": ArmPath(**_FULL), "pooled": ArmPath(**other)},
        candidate="structured_state",
    )
    assert not v.matched


@pytest.mark.parametrize(
    "field_name, other_value",
    [
        ("variance_model", "per_channel_scalar_broadcast"),
        ("calibration_protocol", "held_out_training_windows"),
        ("score_metric", "gaussian_nll_normalised_units"),
        ("split_fingerprint", "deadbeef"),
        ("context_length", 32),
        ("input_normalisation", "none"),
        ("anatomy_provenance", "synthetic_fallback"),
    ],
)
def test_every_path_field_fires(field_name, other_value):
    """One test per field, each demonstrated to fire.

    `calibration_protocol` is run 1's own §11.2 defect: five baselines received a
    held-out residual-variance calibration and SC-WBD received none.
    """
    other = dict(_FULL)
    other[field_name] = other_value
    v = check_path_parity(
        {"cand": ArmPath(**_FULL), "ctrl": ArmPath(**other)}, candidate="cand"
    )
    assert not v.matched, f"{field_name} did not bind"
    assert any(m.startswith(f"{field_name}:") for m in v.mismatches)


def test_undeclared_parity_blocks_rather_than_passing():
    """Unlike an unchecked BUDGET field, unverified parity is not parity.

    Budgets pass-and-name because nothing has ever declared them; path parity is
    new and carries no legacy, so it starts strict.
    """
    v = check_path_parity(
        {"cand": ArmPath(**_FULL), "ctrl": ArmPath(variance_model="x")},
        candidate="cand",
    )
    assert not v.matched
    assert "ctrl.observation_ports" in v.undeclared
    assert "not verifiable" in parity_subcheck(v).reason


def test_adding_a_path_field_extends_the_check_automatically():
    """A check that must be manually extended falls behind the thing it guards."""
    from dataclasses import fields as _f

    names = {f.name for f in _f(ArmPath)}
    # every declared field participates in the comparison loop
    for n in names:
        other = dict(_FULL)
        other[n] = "___different___" if isinstance(_FULL[n], str) else 999999
        v = check_path_parity(
            {"cand": ArmPath(**_FULL), "ctrl": ArmPath(**other)}, candidate="cand"
        )
        assert not v.matched, f"field {n} is declared but not compared"


def test_every_budget_field_is_classified_binding_or_advisory():
    """The regression guard for row 11 itself.

    The original defect was a field declared on `Budget` that no comparison
    consumed.  Adding a new field without classifying it would recreate it
    silently, so the classification is asserted to be total rather than assumed.
    """
    from dataclasses import fields as _f

    from scwbd.bench.matching import ADVISORY_FIELDS, BINDING_FIELDS

    declared = {f.name for f in _f(Budget)} - {"source"}
    classified = set(BINDING_FIELDS) | set(ADVISORY_FIELDS)
    assert declared == classified, (
        f"unclassified Budget field(s) {sorted(declared - classified)}: a field that is "
        "neither binding nor advisory is never compared, which is exactly the defect "
        "this module was fixed for"
    )
    assert not (set(BINDING_FIELDS) & set(ADVISORY_FIELDS))


def test_RL4_amendment_a_broadcast_control_against_a_state_dependent_candidate_fires():
    """ARCHITECTURE.md §5c RL-4, as amended on 🌊 Hodgkin's disagreement.

    The original ruling disabled the observation interface on the CONTROL arm to
    leave the §11.4 control untouched, which would have made the state-dependent
    variance path a property of *which arm you are in* -- an unmatched stage 5,
    in the ruling that exists to prevent unmatched stages. Last time the
    interface silently narrowed one arm; that would have silently widened the
    other.

    RL-4 now says: "An A1 run with the control on the broadcast constant may not
    be reported as a test of structured state." This is that sentence, executable.
    `ModelConfig.state_dependent_variance` is a declared config choice, so the
    two arms CAN legitimately both be False -- what may not happen is a split.
    """
    cand = dict(_FULL, variance_model="state_dependent_logvar")
    ctrl = dict(_FULL, variance_model="per_channel_scalar_broadcast")

    split = check_path_parity(
        {"structured_state": ArmPath(**cand), "pooled": ArmPath(**ctrl)},
        candidate="structured_state",
    )
    assert not split.matched
    assert any(m.startswith("variance_model:") for m in split.mismatches)
    assert parity_subcheck(split).status == "COULD_NOT_RUN"

    # Both arms on the SAME setting is a declared config choice, not a defect --
    # in either direction. The ruling forbids the split, not the setting.
    for shared in ("state_dependent_logvar", "per_channel_scalar_broadcast"):
        both = dict(_FULL, variance_model=shared)
        v = check_path_parity(
            {"structured_state": ArmPath(**both), "pooled": ArmPath(**both)},
            candidate="structured_state",
        )
        assert v.matched, f"both arms at {shared} must be matched"


# ---------------------------------------------------------------------------
# 🔥 Turing's P0 result: the run-1 variance defect is a FITTING failure, not
# architecture -- which creates a stage-5 parity failure ArmPath cannot see.
# ---------------------------------------------------------------------------


def test_run1s_actual_variance_scale_is_flagged_unconverged():
    """Regenerated from the checkpoint and evaluation.json, not quoted.

    `eeg.log_noise` mean +0.2732 (64 channels, verified directly from
    stage_V_individual.pt) against held-out MSE 3.9697, whose closed-form
    optimum is log(3.9697) = 1.3787.  The fitted scalar reached 19.8% of it.
    """
    from scwbd.bench.matching import check_variance_convergence

    v = check_variance_convergence({"scwbd_001_beta": (0.2732, 3.9697)}, tol=0.1)
    assert not v.matched
    assert "scwbd_001_beta" in v.unconverged
    # overconfident: fitted log-variance BELOW the optimum
    assert v.gaps["scwbd_001_beta"] == pytest.approx(0.2732 - math.log(3.9697), abs=1e-9)
    assert v.gaps["scwbd_001_beta"] < 0
    assert math.exp(-v.gaps["scwbd_001_beta"]) == pytest.approx(3.02, abs=0.02)
    assert "not converged" in v.reason


def test_two_arms_both_declaring_state_dependence_can_still_be_unmatched():
    """The gap ArmPath cannot close: same DECLARED config, different CONVERGENCE."""
    from scwbd.bench.matching import check_variance_convergence

    same = dict(_FULL, variance_model="state_dependent_logvar")
    parity = check_path_parity(
        {"cand": ArmPath(**same), "ctrl": ArmPath(**same)}, candidate="cand"
    )
    assert parity.matched, "path parity sees only the declared configuration"

    # one arm 20% converged, the other essentially converged -- same declaration
    v = check_variance_convergence(
        {"cand": (0.2732, 3.9697), "ctrl": (math.log(4.0), 4.0)}, tol=0.1
    )
    assert not v.matched
    assert v.spread > 1.0
    assert "cand" in v.unconverged and "ctrl" not in v.unconverged


def test_variance_convergence_passes_when_both_arms_reached_their_optimum():
    from scwbd.bench.matching import check_variance_convergence

    v = check_variance_convergence(
        {"cand": (math.log(3.9697), 3.9697), "ctrl": (math.log(4.07), 4.07)}, tol=0.1
    )
    assert v.matched and not v.unconverged
    assert v.spread == pytest.approx(0.0, abs=1e-9)


def test_equally_unconverged_arms_are_still_refused():
    """Matched handicaps are not a licence: both arms mis-scaled by 3x is still
    a fitting failure, and neither arm's NLL measures its predictive quality."""
    from scwbd.bench.matching import check_variance_convergence

    v = check_variance_convergence(
        {"cand": (0.2732, 3.9697), "ctrl": (0.2732, 3.9697)}, tol=0.1
    )
    assert v.spread == pytest.approx(0.0, abs=1e-12)  # perfectly matched...
    assert not v.matched                               # ...and still refused
    assert set(v.unconverged) == {"cand", "ctrl"}


def test_dead_parameters_are_a_capacity_fact_and_bind():
    """`bold.log_noise`: 454 values ALL exactly -4.0, its initialiser -- verified
    from the checkpoint.  A parameter that never received a gradient is not
    capacity, and CLAIM_BOUNDARY §3.2d already showed dead parameters making a
    capacity confound 5x worse than reported."""
    cand = _Budgeted(n=100_000)
    cand.__dict__["_eff"] = None
    v = check_matched(
        _Budgeted(n=100_000, **{"width": 1}),
        {"a": _Budgeted(n=100_000, **{"width": 1})},
        tol=0.10,
    )
    assert v.matched  # nominal parity
    assert "n_parameters_effective" in v.unchecked_fields  # ...but unverified, and named

    # declared: nominal parity, 3x EFFECTIVE disparity -> not matched
    class _Eff(_Budgeted):
        @property
        def n_parameters_effective(self):
            return self._eff

    a = _Eff(n=100_000); a._eff = 90_000
    b = _Eff(n=100_000); b._eff = 30_000
    v2 = check_matched(a, {"ctrl": b}, tol=0.10)
    assert not v2.matched
    assert "ctrl.n_parameters_effective" in v2.over_budget_fields
