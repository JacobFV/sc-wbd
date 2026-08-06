"""The adjudication must be able to return every verdict, including neutral."""

from __future__ import annotations

import numpy as np
import pytest

from scwbd.bench.adjudication import (
    LR_RESCALE_STAGE_I,
    AdjudicationInputs,
    LoggingResolution,
    adjudicate,
)


def _inputs(treat_shift: float, *, noise: float = 0.0005, n: int = 40,
            secondary: float | None = None, metric: str = "sim_forecast_nll"):
    rng = np.random.default_rng(0)
    base = 20.0 + rng.normal(0, noise, size=n)
    treat = base * (1.0 + treat_shift) + rng.normal(0, noise, size=n)
    series = {metric: {"baseline": base, "treatment": treat}}
    if secondary is not None:
        sb = 5.0 + rng.normal(0, noise, size=n)
        series["composite_loss"] = {"baseline": sb, "treatment": sb * (1.0 + secondary)}
    return AdjudicationInputs(
        preregistered_metric=metric, series=series, steps=list(range(0, 20 * n, 20)),
        resolution=LoggingResolution(log_every=20), equivalence_margin=0.01,
        lower_is_better=True,
    )


def _verdict(rep):
    return rep.artifacts["primary"]["verdict"]


def test_pending_before_stage_one_ends():
    rep = adjudicate()
    assert rep.status == "COULD_NOT_RUN"
    assert "has not reached end of Stage I" in " ".join(rep.blocking_reasons)
    # preregistration is visible in the artifact itself
    assert rep.manifest.acceptance_thresholds["secondary_metrics_may_change_the_verdict"] is False
    assert rep.manifest.acceptance_thresholds["sub_grid_timing_claims_permitted"] is False


def test_verdict_improved_when_the_metric_really_improves():
    rep = adjudicate(inputs=_inputs(-0.05))     # 5% lower NLL
    assert _verdict(rep) == "IMPROVED"
    assert rep.status == "PASS"


def test_verdict_harmed_when_the_metric_really_worsens():
    rep = adjudicate(inputs=_inputs(+0.05))
    assert _verdict(rep) == "HARMED"
    assert rep.status == "FAIL"


def test_neutral_is_a_real_verdict_and_means_equivalence_was_demonstrated():
    """~0.2% difference with tight intervals: inside the margin, not merely 'ns'."""
    rep = adjudicate(inputs=_inputs(+0.002))
    assert _verdict(rep) == "NEUTRAL"
    m = next(mm for s in rep.subchecks for mm in s.metrics
             if mm.name == "adjudication.equivalence_demonstrated")
    assert m.value == 1.0
    assert "equivalence shown, not merely" in m.note
    assert "LR-insensitive" in rep.artifacts["architecture_hypothesis"]


def test_inconclusive_is_distinct_from_neutral():
    """A wide interval is underpowered, and must not be sold as equivalence."""
    rep = adjudicate(inputs=_inputs(+0.002, noise=0.5, n=12))
    assert _verdict(rep) == "INCONCLUSIVE"
    m = next(mm for s in rep.subchecks for mm in s.metrics
             if mm.name == "adjudication.equivalence_demonstrated")
    assert m.value == 0.0
    assert "UNDERPOWERED" in m.note
    assert "architecture_hypothesis" not in rep.artifacts


def test_secondary_metric_cannot_override_the_preregistered_one():
    """Composite loss says 'improved'; forecast NLL says 'harmed'. NLL governs."""
    rep = adjudicate(inputs=_inputs(+0.05, secondary=-0.20))
    assert _verdict(rep) == "HARMED"
    assert rep.status == "FAIL"
    sub = next(s for s in rep.subchecks if s.name == "secondary_metrics_do_not_govern")
    assert sub.mandatory is False
    assert "DISAGREEMENT" in sub.reason
    assert "resolved in favour of whichever looks better" in sub.reason


def test_a_substitute_metric_is_refused_outright():
    inp = _inputs(-0.05, metric="composite_loss")
    inp = AdjudicationInputs(
        preregistered_metric="sim_forecast_nll", series=inp.series, steps=inp.steps,
        resolution=inp.resolution, equivalence_margin=inp.equivalence_margin)
    rep = adjudicate(inputs=inp)
    assert rep.status == "COULD_NOT_RUN"
    assert "cannot be taken on a substitute metric" in " ".join(rep.blocking_reasons)


def test_reported_steps_are_widened_to_the_logging_window():
    res = LoggingResolution(log_every=20)
    assert res.window(80) == (61, 80)
    assert "could sit anywhere inside it" in res.describe(80)
    rep = adjudicate(inputs=_inputs(+0.002))
    sub = next(s for s in rep.subchecks
               if s.name == "timing_claims_within_instrument_resolution")
    assert "(61, 80]" in sub.metrics[0].note
    assert "never visible" in sub.metrics[0].note


def test_a_non_improving_verdict_is_attributed_to_the_decision_not_the_model():
    rep = adjudicate(inputs=_inputs(+0.05))
    notes = " ".join(rep.notes)
    assert "ATTRIBUTION" in notes
    assert "not against SC-WBD-001-beta" in notes
    assert "SHARED reasoning error" in rep.manifest.consequence_if_failed
    assert "common cause" in rep.manifest.consequence_if_failed
    assert "artifact did nothing wrong" in rep.manifest.consequence_if_failed


def test_withdrawn_evidence_is_recorded_with_credit():
    rep = adjudicate()
    wd = rep.artifacts["decision"]["withdrawn_evidence"]
    assert len(wd) == 2
    assert any("truncated at step 140" in w for w in wd)
    assert any("Withdrawn by agent Turing, unprompted" in w for w in wd)
    assert "to their credit" in " ".join(rep.notes)


def test_missing_variability_estimate_is_declared_prior_specified():
    rep = adjudicate(inputs=_inputs(+0.002))
    assert any("prior-specified sensitivity" in n for n in rep.notes)
    assert any("fixed before the data existed" in n for n in rep.notes)


def test_a_prefix_is_not_admissible_for_a_horizon_registration():
    """The registered test named step 900; the data stops at 260."""
    inp = _inputs(+0.017)
    inp = AdjudicationInputs(
        preregistered_metric=inp.preregistered_metric, series=inp.series,
        steps=list(range(0, 280, 20)), resolution=inp.resolution,
        equivalence_margin=inp.equivalence_margin, preregistered_horizon=900)
    rep = adjudicate(inputs=inp)
    assert rep.status == "COULD_NOT_RUN"
    assert rep.artifacts["admissibility"]["verdict"] == "UNADJUDICATED"
    reason = " ".join(rep.blocking_reasons)
    assert "short of the registered horizon 900" in reason
    assert "which is not NEUTRAL" in reason
    # the prefix is reported, clearly marked as not the test
    assert rep.artifacts["exploratory_prefix"]["is_the_registered_test"] is False
    assert rep.artifacts["exploratory_prefix"]["resolvable"] is False


def test_unadjudicated_is_not_collapsed_into_neutral():
    inp = _inputs(+0.002)
    withheld = AdjudicationInputs(
        preregistered_metric=inp.preregistered_metric, series=inp.series,
        steps=list(range(0, 800, 20)), resolution=inp.resolution,
        equivalence_margin=inp.equivalence_margin, preregistered_horizon=900)
    rep = adjudicate(inputs=withheld)
    assert rep.artifacts["admissibility"]["verdict"] == "UNADJUDICATED"
    assert "primary" not in rep.artifacts, "no verdict may be computed past the horizon gate"


def test_condition_3_override_ruling_is_recorded_with_its_verification():
    from scwbd.bench.adjudication import CONDITION_3_OVERRIDE, CONDITION_3_RULING

    assert "JUSTIFIED_ON_THE_MERITS" in CONDITION_3_RULING
    assert "RECORDING_REQUIREMENT_UPHELD" in CONDITION_3_RULING
    assert "rate-invariant" in CONDITION_3_OVERRIDE.description
    # the override was disclosed for audit, not asserted
    assert "disclosed unprompted for audit" in CONDITION_3_OVERRIDE.owner


def test_condition_2_bar_question_is_separated_from_whether_it_was_met():
    from scwbd.bench.adjudication import CONDITION_2_BAR

    assert "NOT whether the model met the bar" in CONDITION_2_BAR.description
    assert "whether the BAR WAS" in CONDITION_2_BAR.description
    # the two findings must not merge
    assert "different findings" in CONDITION_2_BAR.consequence_if_not_improved
    assert "not as a verdict on the architecture" in CONDITION_2_BAR.consequence_if_not_improved


def test_uninterpretable_is_a_distinct_verdict_from_unadjudicated():
    from scwbd.bench.adjudication import CONDITION_2_RULING, Verdict
    import typing

    assert "UNINTERPRETABLE" in typing.get_args(Verdict)
    assert "UNADJUDICATED" in typing.get_args(Verdict)
    assert CONDITION_2_RULING.startswith("UNINTERPRETABLE_ABOUT_THE_MODEL")
    assert "THREE_LAYER_RULING_UPHELD" in CONDITION_2_RULING


def test_bench_records_its_own_disqualification_from_setting_the_new_bar():
    """Independence consumed by disclosure is a checkable reason, not modesty."""
    import scwbd.bench.adjudication as adj

    src = open(adj.__file__, encoding="utf-8").read()
    assert "specifically disqualified" in src
    assert "independence for this particular bar has already been consumed" in src
    assert "could not honestly call it preregistered" in src


def test_adj1_double_confound_is_recorded():
    from scwbd.bench.adjudication import ADJ1_CONFOUNDS

    assert len(ADJ1_CONFOUNDS) == 2
    assert any("never reached" in c for c in ADJ1_CONFOUNDS)
    assert any("predate the normaliser fix" in c for c in ADJ1_CONFOUNDS)


def test_layer3_ruling_rejects_the_asymmetry_argument_on_stated_grounds():
    import scwbd.bench.adjudication as adj

    src = open(adj.__file__, encoding="utf-8").read()
    assert "ASYMMETRY_ARGUMENT_REJECTED" in adj.CONDITION_2_LAYER3_RULING
    assert "CHANGED THE TARGETS" in src          # the ground it fails on
    assert '"easier" is undefined' in src
    # and the axis where an asymmetry IS available runs the other way
    assert "SAMPLING_BIAS_FAVOURS_THE_MODEL" in adj.CONDITION_2_LAYER3_RULING
    assert "UPPER BOUND on the true running minimum" in src


def test_layer3_ruling_separates_a_bad_bar_from_a_bad_model():
    import scwbd.bench.adjudication as adj

    src = open(adj.__file__, encoding="utf-8").read()
    assert "BAR_INAPPROPRIATE_NO_REFERENCE_CLASS" in adj.CONDITION_2_LAYER3_RULING
    assert "A GUESS WITH A TIMESTAMP" in src
    assert "matched controls" in src.lower()
    # layer 1 is not softened
    assert "does not soften" in src
