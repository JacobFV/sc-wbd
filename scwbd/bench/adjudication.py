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
