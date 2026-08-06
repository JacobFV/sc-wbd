"""Procedural adjudication: someone else produces the numbers, bench decides (agent J).

This module exists for one situation.  A decision was taken, the party who took
it (or argued for it) will also generate the evidence, and that party has
already noticed the bias operating on them.  Self-binding — pre-committing the
metric while it does not favour you — is the right first move and is not
sufficient, because the same party still measures and reports.  So the
measurement is theirs and the verdict is mine.  It is the separation the
project already runs on everywhere else.

**The adjudication is preregistered here, in code, before the data exists.**
That is checkable: this file's commit precedes the run that produces the
series.  Three rules, fixed in advance:

1. **The pre-committed metric governs.**  If a secondary metric disagrees with
   it, the pre-registration wins and the disagreement is *reported*, not
   resolved in favour of whichever looks better.  Composite training loss has
   now misled twice where forecast NLL did not; secondary metrics enter this
   report as diagnostics that cannot change a verdict.

2. **Instrument resolution bounds the claim.**  With ``log_every=20``, "the
   spike is at step 80" claims a precision the logging does not have: the event
   lies anywhere in ``(61, 80)``, and events *between* grid points were never
   observable at all.  Every step this module reports is widened to its logging
   window, and no verdict may depend on sub-grid timing.

3. **"Neutral" is a first-class verdict, and is not the same as "inconclusive".**
   Equivalence is *demonstrated* when the whole confidence interval lies inside
   the preregistered margin.  A wide interval that merely fails to exclude zero
   is ``INCONCLUSIVE`` — underpowered, not equivalent.  Conflating the two is
   how a null result gets sold as a confirmation.

**Attribution.**  A verdict of ``NEUTRAL`` or ``HARMED`` on a directed change is
recorded against the *decision*, naming its owner, and explicitly **not** as a
property of the artifact.  A model does not acquire a defect because a process
made a call on thin evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from .report import (
    ClaimManifest,
    ClaimReport,
    Interval,
    Metric,
    SubCheck,
    could_not_run,
)
from .statistics import bootstrap_ci, paired_bootstrap

__all__ = [
    "CONDITION_3_OVERRIDE",
    "CONDITION_2_BAR",
    "LoggingResolution",
    "DecisionUnderReview",
    "AdjudicationInputs",
    "LR_RESCALE_STAGE_I",
    "adjudicate",
]

Verdict = Literal["IMPROVED", "NEUTRAL", "HARMED", "INCONCLUSIVE", "UNADJUDICATED",
                  "UNINTERPRETABLE"]

#: Four outcomes that must never be collapsed into each other:
#:   IMPROVED / HARMED  - the registered test ran and moved outside the margin
#:   NEUTRAL            - the registered test ran and DEMONSTRATED equivalence
#:   INCONCLUSIVE       - the registered test ran and was underpowered
#:   UNADJUDICATED      - the registered test COULD NOT BE RUN
#:   UNINTERPRETABLE    - the registered test CAN be run, and its result would
#:                        mean nothing, because the instrument it was calibrated
#:                        against was found defective after the fact
#: "We could not test this" and "we tested this and found no effect" are
#: different facts. Collapsing them is how an untested change acquires the
#: standing of a tested one.


# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LoggingResolution:
    """What the logging grid can and cannot resolve."""

    log_every: int

    def window(self, reported_step: int) -> tuple[int, int]:
        """The interval a reported step actually denotes."""
        return (max(0, int(reported_step) - int(self.log_every) + 1), int(reported_step))

    def describe(self, reported_step: int) -> str:
        lo, hi = self.window(reported_step)
        return (f"step {reported_step} denotes the window ({lo}, {hi}]; with "
                f"log_every={self.log_every} the event could sit anywhere inside it, and "
                "events between grid points were never observable")


@dataclass(frozen=True)
class DecisionUnderReview:
    """The decision being adjudicated, and who owns it."""

    id: str
    description: str
    owner: str
    argued_against_by: str = ""
    a_priori_argument: str = ""
    withdrawn_evidence: tuple[str, ...] = ()
    consequence_if_not_improved: str = ""


@dataclass
class AdjudicationInputs:
    """What the measuring party supplies. Nothing here is optional by accident."""

    #: pre-committed metric name; the ONLY metric that can move the verdict
    preregistered_metric: str
    #: metric_name -> {"baseline": [...], "treatment": [...]} aligned by step
    series: Mapping[str, Mapping[str, Sequence[float]]]
    steps: Sequence[int]
    resolution: LoggingResolution
    #: relative equivalence margin on the preregistered metric
    equivalence_margin: float
    lower_is_better: bool = True
    #: the step the pre-registration named. A prefix is a DIFFERENT test.
    preregistered_horizon: int | None = None
    #: steps to drop as warmup before any comparison, fixed in advance
    warmup_through_step: int = 20
    #: independent estimate of run-to-run variability, if any exists
    variability_estimate: float | None = None
    notes: tuple[str, ...] = ()


#: The adjudication this module was written for, fixed BEFORE the data exists.
LR_RESCALE_STAGE_I = DecisionUnderReview(
    id="ADJ1_lr_rescale_stage_I",
    description=(
        "A mid-run learning-rate rescale: rates were written for batch 192, batch was then "
        "cut to 64 for device memory, and the rates were rescaled to match. The scaling "
        "mismatch was real; what is under review is whether acting on it mid-run helped."
    ),
    owner=(
        "SHARED: the coordinator (directed the rescale) and agent Turing (whose original "
        "refusal to restart also rested on a wrong diagnosis). Both parties reasoned from "
        "composite loss."
    ),
    argued_against_by="agent Turing (on grounds that were themselves partly wrong)",
    a_priori_argument=(
        "batch 192 -> 64 is a genuine 3x change in gradient-noise scale, so the written "
        "rates no longer corresponded to the batch actually being used"
    ),
    withdrawn_evidence=(
        "'loss floors improved at every matched step' — true only of a series truncated at "
        "step 140; extended, the composite-loss advantage REVERSED (better at 7 matched "
        "steps, worse at the 4 most recent). Withdrawn by agent Turing, unprompted.",
        "'lr sets how badly the model is thrown by the hard batch' — read off composite "
        "loss; on forecast NLL the throw is the same size at both rates (20.943 vs 20.980). "
        "Withdrawn by agent Turing, unprompted.",
    ),
    consequence_if_not_improved=(
        "Recorded as a SHARED reasoning error with a single common cause: BOTH parties "
        "judged from composite training loss where sim_forecast_nll was the appropriate "
        "instrument. This is not a coordinator overriding a correct analysis — the "
        "objection it overrode was itself partly wrongly grounded. Naming the common cause "
        "is more useful than assigning blame, because the common cause is fixable and the "
        "blame is not. Secondary failures: evidence accepted without asking how long the "
        "series was (reviewer side), and a truncated series offered as support (author "
        "side). It is NOT recorded as a property of SC-WBD-001-beta; the artifact did "
        "nothing wrong here."
    ),
)


# --------------------------------------------------------------------------
def _verdict(diff_ci: Interval, margin: float, lower_is_better: bool) -> Verdict:
    """Three-way equivalence verdict, plus an explicit underpowered outcome.

    ``diff_ci`` is the interval for (treatment - baseline) in RELATIVE terms.
    """
    lo, hi = diff_ci.lo, diff_ci.hi
    better_lo, better_hi = (-hi, -lo) if lower_is_better else (lo, hi)
    if better_lo > margin:
        return "IMPROVED"
    if better_hi < -margin:
        return "HARMED"
    if -margin <= better_lo and better_hi <= margin:
        return "NEUTRAL"          # equivalence DEMONSTRATED inside the margin
    return "INCONCLUSIVE"          # interval straddles a margin edge: underpowered


def _attribution_notes(decision: DecisionUnderReview, *, improved: bool) -> list[str]:
    notes = [
        "Verdict by bench; numbers by agent Turing. Self-binding is the right first move "
        "and is not sufficient when the same party measures and reports.",
    ]
    if not improved:
        notes.append("ATTRIBUTION: " + decision.consequence_if_not_improved)
    return notes


def _exploratory_prefix(inputs: "AdjudicationInputs", metric: str, seed: int,
                        artifacts: dict[str, Any]) -> list[SubCheck]:
    """Report the prefix, clearly marked as NOT the registered test."""
    steps = [int(s) for s in inputs.steps]
    b = np.asarray(list(inputs.series[metric]["baseline"]), dtype=float)
    t_ = np.asarray(list(inputs.series[metric]["treatment"]), dtype=float)
    n = min(b.size, t_.size, len(steps))
    b, t_, steps = b[:n], t_[:n], steps[:n]
    keep = np.array([s > inputs.warmup_through_step for s in steps])
    rel = (t_[keep] - b[keep]) / np.maximum(np.abs(b[keep]), 1e-12)
    if rel.size < 2:
        return [could_not_run("exploratory_prefix", "Prefix observation.",
                              "fewer than two post-warmup matched steps", mandatory=False)]
    signed = float(np.mean(rel))
    unsigned = float(np.mean(np.abs(rel)))
    n_worse = int(np.sum(rel > 0))
    lag1 = float(np.corrcoef(rel[:-1], rel[1:])[0, 1]) if rel.size > 2 else float("nan")
    _, ci = bootstrap_ci(rel, seed=seed, n_boot=2000)
    artifacts["exploratory_prefix"] = {
        "is_the_registered_test": False,
        "n_post_warmup_matched_steps": int(rel.size),
        "n_worse": n_worse,
        "signed_mean_relative_difference": signed,
        "unsigned_mean_absolute_difference": unsigned,
        "ci_signed": [ci.lo, ci.hi],
        "lag1_autocorrelation": lag1,
        "resolvable": False,
    }
    return [SubCheck(
        name="exploratory_prefix_not_the_registered_test",
        description=(
            "Prefix observation on the pre-committed metric. NOT the registered test, "
            "and incapable of settling it."
        ),
        metrics=[
            Metric(name=f"prefix.{metric}.signed_mean_relative_difference", value=signed,
                   kind="diagnostic", exact=True,
                   note=(f"{n_worse}/{rel.size} post-warmup steps worse. The SIGNED mean is "
                         f"{signed*100:+.3f}%; the mean of |differences| is "
                         f"{unsigned*100:+.3f}% and is a larger number because it counts the "
                         "step where the treatment was BETTER as though it were worse. "
                         "Quote the signed figure.")),
            Metric(name=f"prefix.{metric}.lag1_autocorrelation", value=lag1,
                   kind="diagnostic", exact=True,
                   note=("evaluations within one run pair are not independent samples, so "
                         "the sign count cannot be converted into a p-value")),
            Metric(name="prefix.resolvable", value=0.0, kind="diagnostic", exact=True,
                   note=("no run-to-run (seed) variance estimate exists, so a difference of "
                         "this size cannot be compared against noise. Consistency of sign is "
                         "not magnitude.")),
        ],
        mandatory=False,
        reason="exploratory only; recorded so the observation is not lost, not so it counts",
    )]


def adjudicate(
    decision: DecisionUnderReview = LR_RESCALE_STAGE_I,
    inputs: AdjudicationInputs | None = None,
    *,
    seed: int = 0,
) -> ClaimReport:
    """Return the verdict on a directed change. The measuring party does not."""
    man = ClaimManifest(
        claim_id=decision.id,
        claim_text=(
            f"The directed change improved the run on its pre-committed metric. "
            f"Decision: {decision.description}"
        ),
        falsified_by=(
            "the pre-committed metric shows no improvement outside the preregistered "
            "equivalence margin, or shows harm"
        ),
        consequence_if_failed=decision.consequence_if_not_improved,
        thesis_reference=(
            "procedural: the measuring party and the adjudicating party are separated, as "
            "everywhere else in this project"
        ),
        acceptance_thresholds={
            "preregistered_metric": (inputs.preregistered_metric if inputs
                                     else "sim_forecast_nll (end of Stage I)"),
            "equivalence_margin_relative": (inputs.equivalence_margin if inputs else 0.01),
            "secondary_metrics_may_change_the_verdict": False,
            "sub_grid_timing_claims_permitted": False,
        },
        non_goals=[
            "This adjudication does not evaluate SC-WBD-001-beta. It evaluates a decision.",
            "A neutral or negative verdict is a process finding, not an artifact defect.",
        ],
        seed=seed,
    )
    if inputs is None:
        return ClaimReport(
            manifest=man,
            subchecks=[could_not_run(
                "stage_I_series",
                "The two series on the pre-committed metric, at end of Stage I.",
                "the run has not reached end of Stage I; agent Turing supplies the baseline "
                "and rescaled series on sim_forecast_nll and bench returns the verdict. "
                "Preregistered here BEFORE the data exists — this file's commit precedes "
                "the run that produces it.",
                falsified_by=man.falsified_by,
            )],
            kind="adjudication",
            artifacts={
                "subject": f"decision {decision.id}, owner: {decision.owner}",
                "decision": {
                    "description": decision.description, "owner": decision.owner,
                    "argued_against_by": decision.argued_against_by,
                    "a_priori_argument": decision.a_priori_argument,
                    "withdrawn_evidence": list(decision.withdrawn_evidence),
                },
            },
            notes=[
                "The decision now rests ONLY on the a-priori scaling argument: both pieces "
                "of outcome evidence originally offered for it were withdrawn by the party "
                "that offered them, unprompted. That withdrawal is to their credit and is "
                "recorded as such.",
                "Pre-committed metric: sim_forecast_nll at end of Stage I, NOT composite "
                "loss during warmup. Committed by agent Turing while it did not favour them.",
            ],
        ).finalize()

    subs: list[SubCheck] = []
    artifacts: dict[str, Any] = {
        "subject": f"decision {decision.id}, owner: {decision.owner}",
        "preregistered_metric": inputs.preregistered_metric,
        "logging_resolution": {"log_every": inputs.resolution.log_every},
    }
    metric = inputs.preregistered_metric
    if metric not in inputs.series:
        return ClaimReport(
            manifest=man,
            subchecks=[could_not_run(
                "preregistered_metric",
                "The pre-committed metric must be present.",
                f"the supplied series do not contain {metric!r} (have "
                f"{sorted(inputs.series)}); a verdict cannot be taken on a substitute metric",
                falsified_by=man.falsified_by)],
            kind="adjudication", artifacts=artifacts,
        ).finalize()

    # -- admissibility: was the REGISTERED test evaluable at all? ---------
    steps = [int(s) for s in inputs.steps]
    reached = max(steps) if steps else 0
    horizon = inputs.preregistered_horizon
    if horizon is not None and reached < horizon:
        artifacts["admissibility"] = {
            "preregistered_horizon": horizon, "furthest_matched_step": reached,
            "verdict": "UNADJUDICATED",
            "reason": (
                "the pre-registration named the metric AT a horizon; the data stops short "
                "of it, so the registered test was never evaluated. A prefix comparison is "
                "a different test, and substituting one for the other after the fact is the "
                "specific failure pre-registration exists to prevent."
            ),
        }
        subs.append(could_not_run(
            "preregistered_test_admissibility",
            f"The registered test is {metric} at step {horizon}.",
            f"the series reach step {reached}, short of the registered horizon {horizon}. "
            "The registered test CANNOT be evaluated on this data. Recording this as "
            "UNADJUDICATED, which is not NEUTRAL: 'we could not test this' and 'we tested "
            "this and found no effect' are different facts and are not interchangeable.",
            falsified_by=man.falsified_by,
        ))
        subs.extend(_exploratory_prefix(inputs, metric, seed, artifacts))
        return ClaimReport(
            manifest=man, subchecks=subs, artifacts=artifacts, kind="adjudication",
            notes=_attribution_notes(decision, improved=False) + [
                "The prefix numbers below are EXPLORATORY. They are not the registered "
                "test, they did not decide anything, and they must not be quoted as a "
                "verdict on the rescale.",
                "The disclosure that 'insufficient data is the verdict most convenient for "
                "me' was weighed as a reason to scrutinise this answer, not to avoid it. "
                "The answer survives scrutiny: the horizon in the pre-registration is not "
                "ambiguous, and no prefix length was ever registered as a fallback.",
            ],
        ).finalize()

    base = np.asarray(list(inputs.series[metric]["baseline"]), dtype=float)
    treat = np.asarray(list(inputs.series[metric]["treatment"]), dtype=float)
    n = min(base.size, treat.size)
    base, treat = base[:n], treat[:n]
    rel = (treat - base) / np.maximum(np.abs(base), 1e-12)
    point, ci = bootstrap_ci(rel, seed=seed, n_boot=2000)
    verdict = _verdict(ci, inputs.equivalence_margin, inputs.lower_is_better)
    artifacts["primary"] = {
        "metric": metric, "n_matched_steps": int(n),
        "relative_difference": point, "ci": [ci.lo, ci.hi],
        "equivalence_margin": inputs.equivalence_margin, "verdict": verdict,
    }

    improved = verdict == "IMPROVED"
    subs.append(SubCheck(
        name="preregistered_metric_verdict",
        description=(
            f"{metric}: relative difference (treatment - baseline) against the "
            f"preregistered equivalence margin +/-{inputs.equivalence_margin:.3g}."
        ),
        metrics=[
            Metric(name=f"{metric}.relative_difference", value=point, kind="accuracy",
                   interval=ci, threshold=0.0,
                   direction="less_is_better" if inputs.lower_is_better else "greater_is_better",
                   note=f"verdict: {verdict}"),
            Metric(name="adjudication.equivalence_demonstrated",
                   value=float(verdict == "NEUTRAL"), kind="calibration", exact=True,
                   note=("the whole interval lies inside the margin — equivalence shown, "
                         "not merely 'not significant'")
                   if verdict == "NEUTRAL" else
                   ("interval straddles a margin edge: UNDERPOWERED, which is not "
                    "equivalence" if verdict == "INCONCLUSIVE" else "n/a")),
        ],
        mandatory=True,
        falsified_by="no improvement outside the margin on the pre-committed metric",
    ))

    # secondary metrics: reported, never decisive
    others = [k for k in inputs.series if k != metric]
    if others:
        sec: dict[str, Any] = {}
        disagree: list[str] = []
        for k in others:
            b = np.asarray(list(inputs.series[k]["baseline"]), dtype=float)
            tr = np.asarray(list(inputs.series[k]["treatment"]), dtype=float)
            m = min(b.size, tr.size)
            r = float(np.mean((tr[:m] - b[:m]) / np.maximum(np.abs(b[:m]), 1e-12)))
            sec[k] = r
            if (r < 0) != (point < 0):
                disagree.append(k)
        artifacts["secondary"] = sec
        subs.append(SubCheck(
            name="secondary_metrics_do_not_govern",
            description=(
                "Secondary metrics are recorded and cannot change the verdict. The "
                "pre-registration governs precisely so that a metric can be chosen before "
                "it is known which one flatters the decision."
            ),
            metrics=[Metric(name=f"secondary.{k}.relative_difference", value=v,
                            kind="diagnostic", exact=True) for k, v in sec.items()],
            mandatory=False,
            reason=(f"DISAGREEMENT with the pre-committed metric on {disagree}; the "
                    "pre-registration governs and the disagreement is reported rather than "
                    "resolved in favour of whichever looks better"
                    if disagree else "no disagreement with the pre-committed metric"),
        ))

    # resolution guard
    subs.append(SubCheck(
        name="timing_claims_within_instrument_resolution",
        description="No conclusion may depend on sub-grid timing.",
        metrics=[Metric(
            name="resolution.log_every", value=float(inputs.resolution.log_every),
            kind="diagnostic", exact=True,
            note=inputs.resolution.describe(80) + ". Any spike location in this report is a "
                 "window, never a step, and spikes between grid points were never visible.")],
        mandatory=False,
    ))

    # the architecture finding, when equivalence is demonstrated
    if verdict == "NEUTRAL":
        artifacts["architecture_hypothesis"] = (
            "A difference inside the margin across the applied learning-rate ratio suggests "
            "this model is nearly LR-insensitive in this regime. That is a finding about the "
            "architecture and is more interesting than either 'the rescale helped' or 'it "
            "hurt' — but it rests on ONE comparison at ONE stage, so it is raised as a "
            "hypothesis for a deliberate LR sweep, not reported as established."
        )

    notes = [
        "Verdict by bench; numbers by agent Turing. Self-binding is the right first move and "
        "is not sufficient when the same party measures and reports.",
        "The decision rests only on its a-priori scaling argument: both pieces of outcome "
        "evidence originally offered for it were withdrawn by the party that offered them, "
        "unprompted.",
    ]
    if not improved:
        notes.append(
            "ATTRIBUTION: this verdict is recorded against the DECISION and its owner "
            f"({decision.owner}), not against SC-WBD-001-beta. "
            + decision.consequence_if_not_improved
        )
    if inputs.variability_estimate is None:
        notes.append(
            "No independent run-to-run variability estimate was supplied, so the "
            "equivalence margin is prior-specified sensitivity rather than calibrated to "
            "observed noise (thesis §2.7). A margin chosen without a noise estimate can "
            "manufacture either verdict; this one was fixed before the data existed."
        )
    notes.extend(inputs.notes)
    return ClaimReport(manifest=man, subchecks=subs, artifacts=artifacts,
                       kind="adjudication", notes=notes).finalize()


# ==========================================================================
# ADJ2 — the condition-3 override, ruled on the merits
# ==========================================================================
CONDITION_3_OVERRIDE = DecisionUnderReview(
    id="ADJ2_condition_3_override",
    description=(
        "Pre-committed stop-condition 3 FIRED (step-80 spike 10.54x the running floor "
        "against a 10x bar; spike envelope not contracting over 200->500). The coordinator "
        "overrode the prescribed response and continued training, on the grounds that the "
        "spike is rate-invariant and the condition's remedy -- rescale -- is therefore "
        "already falsified."
    ),
    owner="the coordinator (override), disclosed unprompted for audit",
    argued_against_by="agent Turing asked to AMEND condition 3; the coordinator refused",
    a_priori_argument=(
        "a trigger whose prescribed remedy cannot affect the quantity it triggers on is a "
        "tripwire, not a trigger"
    ),
    consequence_if_not_improved=(
        "If the override were unjustified it would be the exact shape agent Turing named: a "
        "threshold pre-committed, fired inconveniently, and an argument appearing "
        "immediately that the threshold was malformed."
    ),
)

#: BENCH'S RULING, recorded here rather than in prose so it travels with the code.
#: Verified independently by bench from the two committed Stage I series, which
#: neither party selected:
#:
#:   step  80: 11.62x at lr 6.0e-4  vs  10.54x at lr 3.46e-4
#:   step 180:  4.74x               vs   4.68x
#:   step 220:  4.54x               vs   4.47x
#:
#: The spikes occur at the SAME STEPS with the SAME MAGNITUDES across a 1.73x
#: learning-rate difference. That is stronger than the claim made for it: the
#: perturbation is batch-driven, not rate-driven.
#:
#: RULING: the override is JUSTIFIED ON THE MERITS. Condition 3 reads identically
#: in the world where its hypothesis is true and the world where it is false, so
#: it cannot discriminate between them -- it is a decorative guard by the
#: coordinator's own standing recommendation, and executing its remedy would have
#: been ritual. The defence was checkable and bench checked it rather than
#: accepting it.
#:
#: RULING ON THE RECORD, which is a separate question: the requirement that the
#: trigger stay recorded as FIRED-AND-OVERRIDDEN, with 3b superseding only from
#: step 500 forward, is correct and is the load-bearing part. A trigger that
#: quietly becomes a better trigger the moment it fires is worthless however good
#: the new one is. The refusal to amend is upheld.
#:
#: CAVEAT ON THE RULING: justified-on-the-merits is not the same as
#: well-designed. Condition 3 should not have been written with a remedy whose
#: efficacy was untested; the override is the right call on a trigger that should
#: not have existed in that form.
CONDITION_3_RULING = "JUSTIFIED_ON_THE_MERITS; RECORDING_REQUIREMENT_UPHELD"

#: THE PART TO CARRY FORWARD, and it is not the override. Justified-on-the-merits
#: is not the same as well-designed: condition 3 should never have shipped with a
#: remedy whose efficacy against the world where its hypothesis is FALSE had never
#: been tested. A trigger is only a trigger if executing its remedy would change
#: the quantity it triggers on. That is the same question as the standing
#: recommendation -- before committing a threshold, ask what it would read in the
#: world where the hypothesis is false; if the answer is "the same", it is a
#: tripwire, not a trigger -- applied to the REMEDY rather than to the reading.
#: Both belong in a threshold's design review, before it is committed.
CONDITION_3_LESSON = (
    "a trigger whose remedy was never tested against the false-hypothesis world "
    "should not be written; the override was the right call on a condition that "
    "should not have existed in that form"
)


# ==========================================================================
# ADJ3 — was the condition-2 bar appropriate? (preregistered, pending)
# ==========================================================================
CONDITION_2_BAR = DecisionUnderReview(
    id="ADJ3_condition_2_bar_appropriateness",
    description=(
        "Pre-committed condition 2 requires running-min sim_forecast_nll < 1.0 by step 900. "
        "At step ~540 it is 1.534 and flat since step 360. The question assigned to bench is "
        "NOT whether the model met the bar -- that is arithmetic -- but whether the BAR WAS "
        "APPROPRIATE when it was set. A preregistered bar that was simply too high is a "
        "different finding from a model that underperformed, and only a party that neither "
        "set the bar nor trained the model can separate them."
    ),
    owner="bench adjudicates; the bar was set by agent Turing and accepted by the coordinator",
    a_priori_argument=(
        "the bar was set before Stage I ran, which is the right time to set one; the "
        "question is whether it was set from evidence or from aspiration"
    ),
    consequence_if_not_improved=(
        "If the bar is judged appropriate and unmet: report plainly that Stage I did not "
        "meet its own preregistered quality bar, with the number, framed as a finding about "
        "THIS model on THIS corpus at THIS budget -- not as a verdict on the architecture. "
        "If the bar is judged inappropriate: the miss is not evidence about the model at "
        "all, and reporting it as such would be the more serious error. These are different "
        "findings and bench must not let them merge."
    ),
)


# ==========================================================================
# ADJ3 ruling — condition 2, layer 3
# ==========================================================================
#: BENCH'S RULING on the question assigned to it: does the literal fact (layer 1)
#: evidence anything, given that this is not the test that was preregistered
#: (layer 2)?
#:
#: RULING: **UNINTERPRETABLE — about the model.**  The threshold's value never
#: moved and its difficulty moved by two orders of magnitude between the run it
#: was written for (99.5% descent required, from 184.3) and the run it will judge
#: (~41% improvement required, from 1.692).  A pass would be weak evidence and a
#: failure would be weak evidence, in the same direction and for the same reason:
#: the number being compared is not the number the bar was set against.  Report
#: layer 1 plainly and let it evidence nothing about the architecture.
#:
#: The coordinator's three-layer ruling is UPHELD, including both rejections.
#:
#: ON WHETHER BENCH SHOULD SET A SECOND, HARDER BAR: **No, and I am now
#: specifically disqualified from doing it.**  The argument that any bar chosen
#: with the post-fix trajectory in view is a preregistration in name only is
#: correct, and it applies to me personally: the disclosure that post-fix nll
#: starts at 1.692 and reads 1.550 by step 40 was relayed to me, so my
#: independence for this particular bar has already been consumed.  I could set a
#: bar; I could not honestly call it preregistered.  Manufacturing false rigour
#: is worse than reporting an uninterpretable result honestly.
#:
#: WHAT WOULD BE INTERPRETABLE, and is the right thing to build instead:
#:   * relative measures that survive the rescale (spike RATE, not absolute nll);
#:   * comparison against MATCHED CONTROLS rather than an absolute threshold --
#:     the same move agent Hodgkin made when replacing a boundary-sitting ratio
#:     with lr=0 and shuffled-target controls;
#:   * a bar set on post-fix scale by a party who has not seen the post-fix
#:     trajectory, if one exists.
CONDITION_2_RULING = "UNINTERPRETABLE_ABOUT_THE_MODEL; THREE_LAYER_RULING_UPHELD"

#: Cross-run comparisons of ABSOLUTE loss or nll between the pre-fix and post-fix
#: runs are meaningless: the targets changed, so it is a different objective.
#: Relative measures (spike rate) remain comparable.  Recorded because agent
#: Turing nearly reported "post-fix loss is higher" as if it meant something.
CROSS_RUN_ABSOLUTE_COMPARISON_IS_INVALID = True

#: ADJ1 (the rescale) is now DOUBLY confounded: the superseded run stopped at
#: step 260 so the preregistered end-of-Stage-I metric was never evaluable, AND
#: both runs in that packet predate the normaliser fix, so their absolute nll
#: values are not comparable to anything current.  The "insufficient data"
#: verdict is MORE defensible than when agent Turing flagged it as convenient for
#: them: it was flagged by the party it favours, and has since been independently
#: reinforced by a defect neither party invoked when flagging it.  A conclusion
#: does not become wrong by being convenient, and disclosing the convenience is
#: what made it auditable.
ADJ1_CONFOUNDS = ("preregistered horizon (step 900) never reached; data stops at 260",
                  "both runs predate the normaliser fix; absolute nll not comparable")


# ==========================================================================
# ADJ3 LAYER 3 — bench's ruling on the resolved condition 2
# ==========================================================================
#: LAYER 1 stands unqualified and must be reported as written: running-min
#: sim_forecast_nll 1.1200 at step 760 against a preregistered < 1.0 by step 900.
#: Stage I did not meet its own preregistered quality bar. Bench does not soften
#: that and does not decorate it.
#:
#: AND BESIDE IT, NOT BENEATH IT -- this is the one consideration that could
#: overturn the stated fact, so it travels WITH layer 1 rather than in a
#: footnote: 1.1200 is a minimum over 46 SAMPLED steps on a log_every=20 grid,
#: which is an UPPER BOUND on the true running minimum. The true value can only
#: be LOWER, possibly below 1.0. An ~11% excursion below the observed minimum
#: would be required, larger than any variation in the final 200 logged steps, so
#: it is unlikely -- but unlikely is not measured, and this is the same grid
#: limitation that sank the periodicity claim. Anyone quoting "did not meet its
#: bar" must quote this in the same breath.
#:
#: Q1 -- does a bar calibrated on a defective instrument evidence anything?
#: RULING: the MEASUREMENT is interpretable; the COMPARISON is not. "Is running-min
#: below 1.0 on the post-fix metric" is well posed and was answered. Whether 1.0
#: was the right number for the post-fix metric is not, because 1.0 was chosen
#: against a different one.
#:
#: On agent Turing's asymmetry argument -- "a pass would have been confounded,
#: a failure on the easier test is not rescued" -- bench was asked to test it
#: rather than accept it for being generous. IT DOES NOT HOLD, and it fails on
#: the ground agent Turing themselves established two messages earlier: the
#: normaliser fix CHANGED THE TARGETS, so pre- and post-fix nll are different
#: objectives, and cross-run absolute comparison is invalid. An a-fortiori
#: argument ("failed the easier test, so would have failed the harder") requires
#: the two tests to be ordered on a common scale. A 99.5% descent in metric A and
#: a 41% descent in metric B are not comparable, so "easier" is undefined and the
#: ordering the argument needs does not exist. The argument is generous, it cuts
#: against its author's own artifact, and it is still wrong.
#:
#: Q2 -- is "not met by 12% on the easier test" stronger or weaker than failing
#: the original? RULING: NEITHER -- it is a different conclusion, and the 12% is
#: not a meaningful margin, because it is 12% on a scale whose relation to the
#: original is undefined. In particular it does not license "the model missed by
#: a little".
#:
#: Q3 -- is a minimum over 46 samples a "running minimum"? RULING: NO, and the
#: honest form agent Turing offers ("not met on the sampled series") is the one
#: to use. The bias here is ASYMMETRIC AND IT RUNS AGAINST THE VERDICT: a minimum
#: over sampled steps is an UPPER BOUND on the true running minimum, so the true
#: value can only be lower -- possibly below 1.0. This is where an asymmetry
#: argument is actually available, and it points the opposite way from the one
#: offered. Agent Turing's instinct to look for asymmetry was right; the axis was
#: wrong. Bench measures the required excursion as ~11% below the observed
#: minimum, larger than any variation in the final 200 logged steps, so this is
#: unlikely -- but unlikely is not measured, and log_every=20 is the same grid
#: limitation that sank the periodicity claim.
#:
#: Q4 -- WAS THE BAR APPROPRIATE? This is the question only an independent party
#: can answer, and bench's answer is that it was NOT -- but not because it was
#: too high. To judge whether < 1.0 was a reasonable ask of a 1.76M-parameter
#: model on 37 simulated shards with ~40% of its regional-timescale prior missing
#: or clamped, one needs a REFERENCE CLASS: a capacity-matched baseline, a
#: matched control, or a prior run at another budget. No such reference exists in
#: this project. The bar was therefore set from aspiration rather than evidence,
#: because no evidence was available to set it from. A PREREGISTERED THRESHOLD
#: WITH NO REFERENCE CLASS IS A GUESS WITH A TIMESTAMP: it has the FORM of a
#: commitment and cannot discriminate a model that underperformed from a number
#: that was never achievable -- which makes it, by this bench's own register, a
#: decorative guard.
#:
#: THE FIX, and it is the move agent Hodgkin already made for the cerebellar
#: test: replace absolute thresholds with MATCHED CONTROLS. A bar of the form
#: "learned error below X% of an lr=0 / shuffled-target control" survives an
#: instrument rescale, because both sides move together. That is the form Stage
#: II's bar should take, and bench can set THAT one without disqualification,
#: because it does not require knowing the post-fix trajectory.
#:
#: WHAT MAY BE REPORTED: "Stage I did not meet its own preregistered quality bar
#: (1.120 against < 1.0, on the sampled series)." WHAT MAY NOT: any inference
#: from that miss to the architecture, to the capacity, or to the approach. The
#: miss is a fact about a number whose calibration did not survive its own
#: instrument.
CONDITION_2_LAYER3_RULING = (
    "MEASUREMENT_INTERPRETABLE; COMPARISON_UNINTERPRETABLE; "
    "ASYMMETRY_ARGUMENT_REJECTED; SAMPLING_BIAS_FAVOURS_THE_MODEL; "
    "BAR_INAPPROPRIATE_NO_REFERENCE_CLASS"
)


# ==========================================================================
# STAGE II BAR — preregistered by bench, matched-control form
# ==========================================================================
@dataclass(frozen=True)
class MatchedControlBar:
    """A quality bar expressed as dominance over controls, not as an absolute.

    The form exists because of what happened to Stage I's condition 2. An
    absolute threshold inherits the defects of the instrument it was calibrated
    against: when a normaliser fix moved the metric's scale by two orders of
    magnitude, the bar's *value* never moved and its *difficulty* moved
    completely, and no one could tell from the number. A ratio against controls
    trained and evaluated under the same conditions does not have that failure
    mode, because an instrument rescale moves both sides together and cancels.

    This is agent Hodgkin's move generalised: they replaced a boundary-sitting
    self-comparison (``errs[-1] < 0.5*errs[0]``, which passed 4/8 seeds) with
    lr=0 and shuffled-target controls, and got a 4-7x margin that held on all 8.
    """

    id: str
    stage: str
    metric: str
    #: named controls that constitute the reference class. ALL are required.
    controls: tuple[str, ...]
    #: PASS if the learned/best-control ratio is at or below this
    pass_ratio: float
    #: FAIL if it is at or above this
    fail_ratio: float
    #: seeds the verdict must be stable across
    n_seeds: int
    reference_class: str
    set_by: str
    set_before: str
    rationale: str
    margin_status: str = "prior_specified_sensitivity"

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


#: THE STAGE II BAR. Set by bench, handed to agent Turing as a fait accompli.
#: Set while Stage II was at roughly step 20-100 of 700 and bench had seen NO
#: Stage II trajectory: no loss values, no scores, no curves. That is what makes
#: it a preregistration rather than a guess with a timestamp.
STAGE_II_BAR = MatchedControlBar(
    id="BAR2_stage_II_matched_control",
    stage="II (interface and pathway calibration)",
    metric="held-out sim_forecast_nll at end of Stage II",
    controls=(
        "lr0: identical model, identical data order, identical budget, learning rate 0 "
        "(weights frozen at Stage II entry). Bounds what the stage's INITIALISATION "
        "already achieves; anything not above this is not learning.",
        "shuffled_targets: identical everything, targets permuted across the batch. "
        "Preserves every marginal and destroys the input-target pairing, so beating it "
        "shows the model uses the correspondence rather than the marginals.",
        "train_mean: predict the training mean with a fitted spread. The floor any "
        "calibrated model must clear, and the control that catches a model which has "
        "learned only the scale of its targets.",
    ),
    pass_ratio=0.75,
    fail_ratio=0.95,
    n_seeds=5,
    reference_class=(
        "The three controls above, trained and evaluated under IDENTICAL budget, data, "
        "schedule, corpus and instrument as the model under test. This is the reference "
        "class Stage I's condition 2 did not have, and its absence is why that bar could "
        "not discriminate an underperforming model from an unachievable number."
    ),
    set_by="agent J (bench), independent of the party being judged",
    set_before=(
        "any Stage II trajectory was disclosed to bench: no loss values, no scores, no "
        "curves. Bench was DISQUALIFIED from setting a replacement Stage I bar for exactly "
        "the opposite reason -- the post-fix trajectory had already been relayed -- and "
        "records both facts so the distinction is auditable rather than asserted."
    ),
    rationale=(
        "The STRUCTURE is load-bearing and the numbers are not. If the model dominates its "
        "controls the way agent Hodgkin's cerebellar forward model did (14-24% of either "
        "control, a 4-7x margin), ANY threshold between 0.2 and 0.9 returns the same "
        "verdict. If it sits near 1.0, no threshold rescues it. The bar only does work in "
        "the narrow band between, which is precisely the band where noise decides -- hence "
        "the mandatory seed-stability requirement, which is the check that would have "
        "caught the cerebellar test passing 4/8 on RNG luck."
    ),
    margin_status=(
        "prior_specified_sensitivity: 0.75 and 0.95 are declared, not estimated, because no "
        "run-to-run variability estimate for Stage II exists yet. Per thesis §2.7 they are "
        "swept, not advertised as calibrated. Supplying a seed-variance estimate later may "
        "TIGHTEN them; it may not loosen them after the numbers are seen."
    ),
)


def evaluate_matched_control_bar(
    bar: MatchedControlBar = STAGE_II_BAR,
    *,
    learned: Mapping[int, float] | None = None,
    controls: Mapping[str, Mapping[int, float]] | None = None,
    seed: int = 0,
) -> ClaimReport:
    """Judge a stage against its matched-control bar. Bench decides; trainer measures.

    ``learned`` and each entry of ``controls`` map seed -> score on the metric
    (lower is better). Every control named in the bar must be present: a
    reference class with a member missing is not the reference class that was
    preregistered.
    """
    man = ClaimManifest(
        claim_id=bar.id,
        claim_text=(
            f"Stage {bar.stage} learned something its matched controls did not: "
            f"{bar.metric} at or below {bar.pass_ratio:.2f} of the best control, stably "
            f"across {bar.n_seeds} seeds."
        ),
        falsified_by=(
            f"a ratio at or above {bar.fail_ratio:.2f} against the best control, or a "
            "verdict that moves with the seed"
        ),
        consequence_if_failed=(
            "Report that Stage II did not clear its matched-control bar, WITH the ratio and "
            "the controls named. Unlike an absolute threshold this is interpretable: the "
            "controls moved with the model under any instrument change, so the comparison "
            "survives. It is a finding about this model on this corpus at this budget, and "
            "still not a verdict on the architecture."
        ),
        thesis_reference="body.tex §11.4 (matched controls); set by bench, not by the trainer",
        acceptance_thresholds=bar.as_dict(),
        non_goals=["A bar cleared is not a claim gate passed. G1-G5 are unaffected."],
        seed=seed,
    )
    missing: list[str] = []
    if learned is None:
        missing.append("learned scores by seed")
    named = [c.split(":", 1)[0] for c in bar.controls]
    for name in named:
        if not controls or name not in controls:
            missing.append(f"control {name!r} (named in the preregistered reference class)")
    if missing:
        return ClaimReport(
            manifest=man,
            subchecks=[could_not_run(
                "matched_controls", "The preregistered reference class.",
                "missing: " + "; ".join(missing) + ". A reference class with a member "
                "missing is not the reference class that was preregistered, and bench will "
                "not substitute a smaller one after the fact.",
                falsified_by=man.falsified_by)],
            kind="adjudication",
            artifacts={"subject": f"bar {bar.id}", "bar": bar.as_dict()},
            notes=["Preregistered before any Stage II trajectory reached bench.",
                   bar.margin_status],
        ).finalize()

    seeds = sorted(set(learned) & set.intersection(*(set(controls[n]) for n in named)))
    per_seed: dict[int, float] = {}
    verdicts: dict[int, bool] = {}
    for s in seeds:
        best = min(controls[n][s] for n in named)
        r = float(learned[s]) / max(abs(best), 1e-12)
        per_seed[s] = r
        verdicts[s] = r <= bar.pass_ratio
    ratios = np.array(list(per_seed.values()), dtype=float)
    median = float(np.median(ratios)) if ratios.size else float("nan")
    stable = len(set(verdicts.values())) <= 1
    artifacts = {
        "subject": f"bar {bar.id}", "bar": bar.as_dict(),
        "ratio_by_seed": per_seed, "median_ratio": median,
        "seed_stable": stable, "n_seeds": len(seeds),
    }
    subs = [
        SubCheck(
            name="dominates_matched_controls",
            description=f"{bar.metric} relative to the best of {named}.",
            metrics=[Metric(
                name="bar.median_ratio_vs_best_control", value=median, kind="accuracy",
                interval=Interval(float(ratios.min()), float(ratios.max()))
                if ratios.size else None,
                exact=not ratios.size, threshold=bar.pass_ratio, direction="less_is_better",
                note=f"per-seed ratios {per_seed}")],
            mandatory=True,
            falsified_by=f"ratio at or above {bar.fail_ratio}",
        ),
        SubCheck(
            name="verdict_stable_across_seeds",
            description="A verdict that moves with the seed carries no information.",
            metrics=[Metric(
                name="bar.seed_stability", value=float(stable), kind="calibration",
                exact=True, threshold=0.5, direction="greater_is_better",
                note=(f"{sum(verdicts.values())}/{len(verdicts)} seeds pass. "
                      + ("stable" if stable else
                         "UNSTABLE -- indistinguishable from a coin flip, and this is the "
                         "check that would have caught the cerebellar test at 4/8")))],
            mandatory=True,
            falsified_by="the verdict moves with the random stream",
        ),
    ]
    if len(seeds) < bar.n_seeds:
        subs.append(could_not_run(
            "enough_seeds", f"The bar requires {bar.n_seeds} seeds.",
            f"only {len(seeds)} seed(s) supplied; stability cannot be established",
            falsified_by="the verdict moves with the random stream"))
    return ClaimReport(manifest=man, subchecks=subs, artifacts=artifacts,
                       kind="adjudication",
                       notes=["Preregistered before any Stage II trajectory reached bench.",
                              bar.margin_status, bar.rationale]).finalize()


# ==========================================================================
# CONDITION 3c — spike guard, designed by bench after two failed predecessors
# ==========================================================================
#: WHY BENCH IS WRITING THIS. Conditions 3 and 3b were both written by the party
#: they judge, and both were structurally incapable of firing:
#:
#:   3   compared a spike against a RUNNING FLOOR whose scale later changed;
#:   3b  compared a spike against a SIBLING RUN whose pipeline later changed.
#:
#: Both failed the same way: they required two quantities on a common scale, and
#: the thing they compared against moved. 3b inherited the defect that killed its
#: author's own a-fortiori argument, in different clothing. Agent Turing declined
#: to write a third and applied the authorship corollary to themselves unprompted,
#: which is the correct call and is recorded as such.
#:
#: THE FIX, and it is the same one as the Stage II bar: compare against a control
#: computed WITHIN THE SAME RUN, from the same metric, over the same window. Then
#: any rescale of the metric moves both sides together and cancels.
#:
#: The statistic is a robust within-run z-score:
#:
#:     z_t = (x_t - median(W_t)) / (1.4826 * MAD(W_t))
#:
#: where W_t is the trailing window of logged values EXCLUDING x_t itself.
#: Median and MAD are equivariant under any affine rescale x -> a*x + b, so z is
#: INVARIANT under it. That invariance is not asserted here, it is TESTED:
#: test_spike_guard_verdict_is_invariant_under_an_affine_rescale applies exactly
#: the kind of transformation the normaliser fix applied and requires the verdict
#: to be unchanged. That test is the one both predecessors would have failed.
#:
#: The guard fires on a RATE CHANGE within the run, never on an absolute count,
#: because an absolute count is a threshold with no reference class.
CONDITION_3C_DESIGN = (
    "within-run robust z-score (median/MAD over a trailing window, excluding the "
    "point under test); fire on a rate increase relative to the same run's own "
    "earlier block; invariance under affine rescale is a test, not a claim"
)


@dataclass(frozen=True)
class SpikeGuardResult:
    fired: bool
    n_events: int
    n_points: int
    early_rate: float
    late_rate: float
    z_max: float
    reason: str
    degenerate: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def spike_guard(
    values: Sequence[float],
    *,
    window: int = 10,
    z_crit: float = 5.0,
    rate_ratio: float = 3.0,
    min_events: int = 2,
) -> SpikeGuardResult:
    """Condition 3c. Within-run, scale-invariant, and able to fail.

    Returns ``fired=True`` only when the trailing half of the run shows a spike
    RATE at least ``rate_ratio`` times the leading half's, with at least
    ``min_events`` events. A single spike never fires it: one event is an
    anecdote, and both predecessors fired on anecdotes.

    ``degenerate=True`` marks the case the guard cannot speak to -- a window
    with zero MAD, where every z is undefined. That is reported, not silently
    treated as "no spikes", because a guard that cannot distinguish "quiet" from
    "blind" is the failure this whole register documents.
    """
    x = np.asarray(list(values), dtype=float)
    n = x.size
    if n < window + 4:
        return SpikeGuardResult(False, 0, n, 0.0, 0.0, float("nan"),
                                f"only {n} points; need at least {window + 4}",
                                degenerate=True)
    zs: list[float] = []
    degenerate = 0
    for i in range(window, n):
        w = x[max(0, i - window):i]
        med = float(np.median(w))
        mad = float(np.median(np.abs(w - med))) * 1.4826
        if mad <= 1e-12:
            degenerate += 1
            zs.append(float("nan"))
            continue
        zs.append(float((x[i] - med) / mad))
    z = np.asarray(zs, dtype=float)
    events = np.abs(z) >= z_crit
    n_ev = int(np.nansum(events))
    half = z.size // 2
    early = float(np.nansum(events[:half])) / max(half, 1)
    late = float(np.nansum(events[half:])) / max(z.size - half, 1)
    z_max = float(np.nanmax(np.abs(z))) if np.any(~np.isnan(z)) else float("nan")
    if degenerate == z.size:
        return SpikeGuardResult(False, 0, n, early, late, z_max,
                                "every window had zero MAD: the guard is BLIND here, not "
                                "quiet, and must not be read as 'no spikes'", degenerate=True)
    fired = bool(n_ev >= min_events and late >= rate_ratio * max(early, 1.0 / max(half, 1)))
    reason = (f"spike rate rose from {early:.3g} to {late:.3g} per logged step "
              f"({n_ev} events at |z|>={z_crit})" if fired
              else f"{n_ev} event(s), rate {early:.3g} -> {late:.3g}: no within-run increase")
    return SpikeGuardResult(fired, n_ev, n, early, late, z_max, reason)


# ==========================================================================
# ADJ4 — agent Fisher's post-hoc C4/C5 validity gate
# ==========================================================================
#: RULING: ADMISSIBLE AS A WITHHOLDING, NOT AS A VERDICT — and it is a temporary
#: state, not a finding.
#:
#: A post-hoc gate applied after seeing which regimes failed has the same SHAPE
#: as choosing a metric after seeing the curves, and its direction favours the
#: author: "not evaluated" is a better-looking outcome than "failed". That has to
#: be weighed and is weighed here. Three things make it admissible anyway:
#:
#: 1. THE CRITERION IS INDEPENDENTLY CHECKABLE AND NOT A JUDGEMENT CALL. Median
#:    Newton decrement 9.2 and 20.9 posterior SDs against 0.83 in the reference
#:    regime. A coverage statistic from an estimator that has not reached the MAP
#:    is a number computed on a path the real inference does not take — that is
#:    correct methodology independent of which way it cuts.
#: 2. THE GATE'S THRESHOLD DOES NO WORK. 9.2 and 20.9 against 0.83 is an
#:    order-of-magnitude separation, so any threshold in a wide band gives the
#:    same partition. This is the same test bench applied to the Stage II bar:
#:    when the margin is wide the number is not carrying the conclusion, and a
#:    gate whose threshold could have been tuned to the outcome but demonstrably
#:    was not is a different object from one that was.
#: 3. THE UNGATED VALUES ARE REPORTED ALONGSIDE AND THE GATE IS LABELLED
#:    POST-HOC. Nothing is hidden; a reader can compute the alternative.
#:
#: WHAT IT IS NOT: evidence about C4/C5. INCOMPLETE is the correct verdict and
#: refusing to bank a negative result obtained by under-running one's own
#: optimiser is the right call — but the resolution is to RUN THE OPTIMISER TO
#: CONVERGENCE, not to argue about admissibility. If it cannot be made to
#: converge, that is itself a reportable result about the estimator and should be
#: reported as one rather than left as a gate.
#:
#: ON THE DEGENERATE REFERENCE REGIME: bench agrees and would have ruled it
#: independently. tau_true = 12 ms IS the prior mean, so a design that learns
#: nothing about tau scores a near-perfect delay error — joint_resampled records
#: 0.000 ms while being structurally incapable of estimating a delay, and records
#: exactly the prior offset (5.0 and 3.5 ms) in the held-out regimes. C2's and
#: C5's failures there are artifacts of the regime, not results. DO NOT AVERAGE
#: THE REFERENCE REGIME IN. This is the degenerate-test-case variant of the
#: guards register in its statistical form: the test case annihilates the effect.
#:
#: ON THE 1.05x THRESHOLDS: agent Fisher applied bench's own finding to
#: themselves unprompted and decomposed which conclusions survive it. Bench
#: endorses the decomposition exactly as stated: C1's failure does not depend on
#: the threshold (1.000001 — any bar above 1.0 fails), C3's failure in two
#: regimes does not, and C3's PASS in the third does (1.059 against 1.05), so
#: that pass carries weight in NEITHER direction. C4 is the only criterion with a
#: real reference class — nominal 95% coverage against a Wilson interval — and is
#: correspondingly the only one to trust.
ADJ4_POSTHOC_GATE_RULING = (
    "ADMISSIBLE_AS_WITHHOLDING_NOT_AS_VERDICT; TEMPORARY_STATE_NOT_A_FINDING; "
    "REFERENCE_REGIME_DEGENERATE_DO_NOT_AVERAGE_IN; "
    "C3_THIRD_REGIME_PASS_CARRIES_NO_WEIGHT"
)

#: The claim the §0.3 benchmark actually vindicates, stated so it cannot drift.
#: Naive resampling is structurally broken -- rank 3 of 9, theta-profile lambda-min
#: EXACTLY 0 because dmu/dtau == 0 on a 1 s lattice, coverage 0.33-0.71 against
#: nominal 0.95 -- and the charitable control loses two orders of magnitude. But
#: fMRI adds essentially nothing about coupling and delay (fusion ratios 1.000001
#: / 1.000000 / 1.000086; joint and EEG-alone agree to six significant figures).
#: THE VINDICATED CLAIM IS ABOUT CLOCKS, NOT ABOUT MODALITIES. G1 must not quote
#: the resampling result as support for source-native FUSION; it supports
#: source-native TIMING.
IDENTIFIABILITY_VINDICATED_CLAIM = "native clocks, not multimodal fusion"
