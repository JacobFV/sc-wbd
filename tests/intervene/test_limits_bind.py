"""Every declared bound in ``a_safe.toml`` is made to FIRE. SIMULATION ONLY.

``reports/decorative_guards.md`` catalogues ~26 guards in this codebase that
looked green and were incapable of failing.  A safety limit in that category is
the worst kind: it is cited, it is reviewed, it is in the report, and it cannot
refuse anything.

Before this file, the audit was: of 15 numeric bounds declared in
``a_safe.toml``, **6** had a test that made them refuse.  Nine did not --
including every tFUS bound (mechanical index, ISPPA, ISPTA, thermal dose,
temperature rise, duty cycle), the session-duration ceiling, and all three
``min`` sides, so ``LimitSpec.check``'s entire lower-bound comparator had never
executed in anger.  ``protocol.reversibility`` was worse: declared with
``required = true`` and no ``min``/``max``, it was silently skipped by
``SafetyLimits.load`` and never became a ``LimitSpec`` at all.

The tests here are written against the *loaded limits*, not against a hardcoded
list, so a bound added to the file tomorrow is covered tomorrow -- and a bound
that cannot be made to fire fails this suite rather than passing it quietly.
"""

from __future__ import annotations

import math

import pytest

from scwbd.intervene.safety import (
    CompilerRefusal,
    FeasibleSet,
    LimitSpec,
    ProposedIntervention,
    SafetyLimits,
)

LIMITS = SafetyLimits.load()
#: Every numeric bound, as (axis key, spec).
ALL_SPECS: list[tuple[str, LimitSpec]] = sorted(
    (s.key, s) for s in LIMITS.all_specs()
)

#: (axis, spec, side) for every bound side actually declared.
BOUND_SIDES: list[tuple[str, LimitSpec, str]] = [
    (key, spec, side)
    for key, spec in ALL_SPECS
    for side in ("min", "max")
    if getattr(spec, {"min": "minimum", "max": "maximum"}[side]) is not None
]


def _violating(spec: LimitSpec, side: str) -> float:
    if side == "max":
        assert spec.maximum is not None
        return spec.maximum + max(abs(spec.maximum), 1.0)
    assert spec.minimum is not None
    return spec.minimum - max(abs(spec.minimum), 1.0)


def _satisfying(spec: LimitSpec) -> float:
    lo = spec.minimum if spec.minimum is not None else 0.0
    hi = spec.maximum if spec.maximum is not None else lo + 1.0
    return (lo + hi) / 2.0


def _proposal(axis: str, value: float, modality: str = "tms") -> ProposedIntervention:
    return ProposedIntervention(
        label=f"probe::{axis}",
        modality=modality,
        exposure={axis: value},
        pose_certified=True,
        reversible=True,
    )


def _ids(items):
    return [f"{k}:{side}" for k, _, side in items]


# ---------------------------------------------------------------------------
# 1. every declared bound side refuses when crossed
# ---------------------------------------------------------------------------


class TestEveryDeclaredBoundFires:
    @pytest.mark.parametrize(
        "key,spec,side", BOUND_SIDES, ids=_ids(BOUND_SIDES)
    )
    def test_crossing_the_bound_raises_r11(self, key, spec, side):
        fs = FeasibleSet()
        with pytest.raises(CompilerRefusal) as e:
            fs.guard(_proposal(key, _violating(spec, side)))
        assert e.value.code == "R11"
        assert key in str(e.value)

    @pytest.mark.parametrize(
        "key,spec,side", BOUND_SIDES, ids=_ids(BOUND_SIDES)
    )
    def test_the_violation_names_the_axis_the_side_and_the_citation(
        self, key, spec, side
    ):
        """A refusal that does not say what it enforced is not reviewable."""
        verdict = FeasibleSet().contains(_proposal(key, _violating(spec, side)))
        assert not verdict.feasible
        v = next(x for x in verdict.violations if x.limit.key == key)
        assert v.kind == {"min": "below_minimum", "max": "above_maximum"}[side]
        assert spec.citation
        assert spec.citation in str(v)

    @pytest.mark.parametrize("key,spec", ALL_SPECS, ids=[k for k, _ in ALL_SPECS])
    def test_a_value_inside_the_bound_does_not_refuse(self, key, spec):
        """Proves the refusals above are caused by the value, not by the axis."""
        assert FeasibleSet().contains(_proposal(key, _satisfying(spec))).feasible

    @pytest.mark.parametrize(
        "key,spec,side", BOUND_SIDES, ids=_ids(BOUND_SIDES)
    )
    def test_the_bound_itself_is_admitted(self, key, spec, side):
        """The comparison is inclusive at the bound; crossing it is what refuses."""
        at = spec.maximum if side == "max" else spec.minimum
        assert FeasibleSet().contains(_proposal(key, float(at))).feasible


# ---------------------------------------------------------------------------
# 2. the lower-bound comparator specifically
# ---------------------------------------------------------------------------


class TestTheLowerBoundComparatorExecutes:
    """``LimitSpec.check``'s ``below_minimum`` branch had no firing test at all.

    Three axes declare a ``min``. Each is fired individually here so the branch
    is exercised by name rather than by a parametrized sweep alone.
    """

    def test_a_train_repeated_faster_than_the_safety_interval_refuses(self):
        with pytest.raises(CompilerRefusal) as e:
            FeasibleSet().guard(_proposal("tms.intertrain_interval_s", 0.2))
        assert "below_minimum" in str(e.value)
        assert "Rossi" in str(e.value)

    def test_a_negative_coil_distance_refuses(self):
        """A coil inside the scalp is not a small number, it is a wrong one."""
        with pytest.raises(CompilerRefusal) as e:
            FeasibleSet().guard(_proposal("tms.coil_scalp_distance_mm", -1.0))
        assert "below_minimum" in str(e.value)

    def test_a_negative_flash_rate_refuses(self):
        with pytest.raises(CompilerRefusal) as e:
            FeasibleSet().guard(
                _proposal("sensory.luminance_flash_hz", -0.5, modality="sensory")
            )
        assert "below_minimum" in str(e.value)


# ---------------------------------------------------------------------------
# 3. the tFUS envelope, none of which fired before
# ---------------------------------------------------------------------------


class TestTheTfusEnvelopeBinds:
    @pytest.mark.parametrize(
        "axis,value,citation_fragment",
        [
            ("tfus.mechanical_index", 3.0, "FDA"),
            ("tfus.isppa_w_per_cm2", 400.0, "FDA"),
            ("tfus.ispta_mw_per_cm2", 1500.0, "FDA"),
            ("tfus.duty_cycle", 0.9, "Aubry"),
            ("tfus.cem43_minutes", 5.0, "Sapareto"),
            ("tfus.temperature_rise_c", 4.0, "Aubry"),
        ],
    )
    def test_an_over_envelope_exposure_refuses(self, axis, value, citation_fragment):
        with pytest.raises(CompilerRefusal) as e:
            FeasibleSet().guard(_proposal(axis, value, modality="tfus"))
        assert axis in str(e.value)
        assert citation_fragment in str(e.value)

    def test_the_thermal_axes_have_no_producer_and_that_is_recorded(self):
        """An honest statement of a real gap, asserted so it cannot drift.

        ``ExposureMetrics.as_safety_axes`` emits MI, ISPPA, ISPTA and duty
        cycle. It does **not** emit thermal dose or temperature rise: nothing
        in ``scwbd`` computes them. The two limits above therefore bind only
        when a caller supplies the values by hand -- which is why a *live*
        tFUS plan is additionally required to cover every declared axis (see
        ``TestALivePlanCannotOmitAnAxis``). If a producer is added later, this
        test fails and should be deleted.
        """
        from scwbd.intervene.tfus.exposure import ExposureMetrics

        emitted = set(
            ExposureMetrics(
                peak_positive_pressure_pa=1e5,
                peak_negative_pressure_pa=-1e5,
                isppa_w_per_cm2=1.0,
                ispta_mw_per_cm2=10.0,
                mechanical_index=0.5,
                frequency_hz=5e5,
                duty_cycle=0.1,
                focal_volume_mm3=20.0,
            ).as_safety_axes()
        )
        declared = {s.key for s in LIMITS.for_modality("tfus")}
        assert declared - emitted == {
            "tfus.cem43_minutes",
            "tfus.temperature_rise_c",
        }


# ---------------------------------------------------------------------------
# 4. an omitted axis is silence -- and silence is refused for a live plan
# ---------------------------------------------------------------------------


class TestALivePlanCannotOmitAnAxis:
    def test_a_computational_plan_may_omit_declared_axes(self):
        """Unchanged behaviour, stated so the change below is visibly scoped."""
        v = FeasibleSet().contains(_proposal("tfus.mechanical_index", 0.5, "tfus"))
        assert v.feasible
        assert "tfus.cem43_minutes" in v.unchecked_declared_axes

    def test_a_live_plan_that_omits_the_thermal_axes_refuses(self):
        """The exact evasion the coverage rule exists to close.

        Supply the three comfortable acoustic axes, omit thermal dose and
        temperature rise, and before this rule the plan was 'feasible' with the
        omission recorded in a field nobody reads.
        """
        live = ProposedIntervention(
            label="live_tfus",
            modality="tfus",
            exposure={
                "tfus.mechanical_index": 0.5,
                "tfus.isppa_w_per_cm2": 10.0,
                "tfus.ispta_mw_per_cm2": 100.0,
            },
            pose_certified=True,
            reversible=True,
            application="live",
        )
        with pytest.raises(CompilerRefusal) as e:
            FeasibleSet().guard(live)
        msg = str(e.value)
        assert "tfus.cem43_minutes" in msg
        assert "tfus.temperature_rise_c" in msg

    def test_the_coverage_rule_cannot_be_switched_off_for_a_live_plan(self):
        """``require_complete_coverage=False`` does not reach the live path."""
        fs = FeasibleSet(require_complete_coverage=False)
        live = ProposedIntervention(
            label="live_tms",
            modality="tms",
            exposure={"tms.peak_efield_v_per_m": 95.0},
            pose_certified=True,
            reversible=True,
            application="live",
        )
        assert not fs.contains(live).feasible

    def test_a_live_plan_covering_every_declared_axis_is_feasible(self):
        """The rule can be satisfied; it is a gate, not a wall."""
        declared = {s.key for s in LIMITS.for_modality("tms")} | {
            s.key for s in LIMITS.for_modality("protocol")
        }
        exposure = {k: _satisfying(LIMITS.get(k)) for k in declared}
        live = ProposedIntervention(
            label="live_tms_full",
            modality="tms",
            exposure=exposure,
            pose_certified=True,
            reversible=True,
            application="live",
        )
        assert FeasibleSet().contains(live).feasible


# ---------------------------------------------------------------------------
# 5. a limit that cannot fire is refused at load
# ---------------------------------------------------------------------------


class TestADecorativeLimitIsRefusedAtLoad:
    def test_an_entry_with_neither_min_nor_max_refuses(self, tmp_path):
        """How ``protocol.reversibility`` sat here unenforced for the whole project."""
        p = tmp_path / "a_safe.toml"
        p.write_text(
            'schema_version = "x"\n\n'
            "[protocol.reversibility]\n"
            "required = true\n"
            'basis = "sounds binding"\n'
            'citation = "body.tex Sec. 7.4"\n'
        )
        with pytest.raises(CompilerRefusal) as e:
            SafetyLimits.load(p)
        assert "cannot fire is not a bound" in str(e.value)

    def test_the_shipped_file_has_no_such_entry(self):
        import tomllib

        from scwbd.intervene.safety import DEFAULT_LIMITS_PATH

        with open(DEFAULT_LIMITS_PATH, "rb") as fh:
            raw = tomllib.load(fh)
        for modality, block in raw.items():
            if not isinstance(block, dict) or modality == "decision":
                continue
            for quantity, entry in block.items():
                if isinstance(entry, dict):
                    assert "min" in entry or "max" in entry, (
                        f"{modality}.{quantity} declares no numeric bound"
                    )

    def test_every_loaded_limit_carries_a_citation(self):
        for _, spec in ALL_SPECS:
            assert spec.citation.strip()
            assert spec.basis.strip()

    def test_the_reversibility_rule_moved_rather_than_vanished(self):
        """Deleting a decorative guard silently is its own defect."""
        assert LIMITS.require_reversible_for_live() is True

    def test_and_it_now_fires(self):
        """Moving it would be pointless if it still could not refuse anything."""
        declared = {s.key for s in LIMITS.for_modality("tms")} | {
            s.key for s in LIMITS.for_modality("protocol")
        }
        irreversible = ProposedIntervention(
            label="live_irreversible",
            modality="tms",
            exposure={k: _satisfying(LIMITS.get(k)) for k in declared},
            pose_certified=True,
            reversible=False,
            application="live",
        )
        with pytest.raises(CompilerRefusal) as e:
            FeasibleSet().guard(irreversible)
        assert "reversibility" in str(e.value)

    def test_it_does_not_fire_on_the_computational_path(self):
        """Scoped to live application, as the rule says."""
        assert FeasibleSet().contains(
            ProposedIntervention(
                label="sim",
                modality="tms",
                exposure={"tms.peak_efield_v_per_m": 95.0},
                pose_certified=True,
                reversible=False,
            )
        ).feasible


# ---------------------------------------------------------------------------
# 6. the limits are still unlearnable and unwritable
# ---------------------------------------------------------------------------


class TestTheLimitsRemainOutsideTheObjective:
    def test_no_bound_is_a_tensor_or_a_parameter(self):
        for _, spec in ALL_SPECS:
            for bound in (spec.minimum, spec.maximum):
                if bound is not None:
                    # int is fine (`pulses_per_session = 6000`); what must never
                    # appear is anything carrying a gradient.
                    assert isinstance(bound, (int, float))
                    assert not hasattr(bound, "requires_grad")
                    assert math.isfinite(float(bound))

    def test_the_feasible_set_has_no_scorer(self):
        """A soft penalty is exactly the failure mode R11 names."""
        for banned in ("score", "penalty", "loss", "soft"):
            assert not hasattr(FeasibleSet(), banned)


# ---------------------------------------------------------------------------
# 7. the last outcome in safety.py that no test reached
# ---------------------------------------------------------------------------


class TestASingleAdmissibleCandidateNeverRanks:
    """``NoRecommendation("admissible candidates are not distinguishable")``.

    Reading `RiskSensitiveController.decide`: with one feasible candidate the
    code sets ``gap = float("inf")``, the disagreement branch is skipped
    (``numel() > 1`` is False), and ``not math.isfinite(gap)`` then returns
    `NoRecommendation`. So **a single admissible candidate can never produce a
    `SimulatedRanking`** — a real behaviour of this controller that was not
    written down anywhere and that no test reached.

    It is arguably the right behaviour: "the best of one" is not a comparison,
    and this controller exists to compare. But it was undocumented and
    unexercised, which is how a behaviour becomes a surprise.
    """

    def test_one_feasible_candidate_yields_no_recommendation_not_a_ranking(self):
        import torch

        from scwbd.intervene.safety import (
            NoRecommendation,
            RiskSensitiveController,
            SimulatedRanking,
        )

        dt = torch.float64
        only = ProposedIntervention(
            label="the_only_one",
            modality="tms",
            exposure={"tms.peak_efield_v_per_m": 95.0},
            pose_certified=True,
            reversible=True,
        )
        out = RiskSensitiveController(gamma=1.0).decide(
            [only],
            benefit=torch.tensor([[1.0], [1.0]], dtype=dt),
            burden=torch.zeros(1, dtype=dt),
            harm=torch.zeros((2, 1), dtype=dt),
            model_log_weights=torch.zeros(2, dtype=dt),
        )
        assert isinstance(out, NoRecommendation)
        assert not isinstance(out, SimulatedRanking)
        assert "not distinguishable" in out.reason

    def test_two_candidates_do_rank(self):
        """The discriminating control: the refusal above is about arity."""
        import torch

        from scwbd.intervene.safety import RiskSensitiveController, SimulatedRanking

        dt = torch.float64
        cands = [
            ProposedIntervention(
                label=f"pose_{i}",
                modality="tms",
                exposure={"tms.peak_efield_v_per_m": 95.0},
                pose_certified=True,
                reversible=True,
            )
            for i in range(2)
        ]
        out = RiskSensitiveController(beta=1.0, gamma=0.0).decide(
            cands,
            benefit=torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=dt),
            burden=torch.zeros(2, dtype=dt),
            harm=torch.tensor([[0.0, 0.4], [0.0, 0.4]], dtype=dt),
            model_log_weights=torch.zeros(2, dtype=dt),
        )
        assert isinstance(out, SimulatedRanking)
