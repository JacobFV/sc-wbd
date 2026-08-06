"""Do my own instruments discriminate? (agent J)

``tests/bench`` already answers the gate-level question: *a gate that cannot
fail is worthless*, so every gate ships with a world in which its claim is
false and it must say so.  This module generalises that discipline one level
down, to the **guards and provenance fields the gates themselves rely on**.

The generalisation was forced by evidence.  Twenty times in this project an
instrument reported a discrimination it was structurally incapable of making,
and every one of them looked green:

1. ``torch.compile`` renamed parameters, so exact-name gradient permissions
   matched *nothing* on CUDA while passing on CPU — the permission check was
   reading a name that no longer existed;
2. ``systemd-run MemoryMax`` did not charge CUDA allocations on unified
   memory, so ``memory.current`` read a reassuring 8 GB against a 40 GB cap
   that was not binding anything;
3. at OOM, "allocated by PyTorch" always equalled the ceiling, so the number
   could not distinguish batch-linear growth from batch-independent growth;
4. ``git_sha()`` appended ``-dirty`` whenever the whole tree was dirty — which
   it always is during a run, because the run writes tracked logs — so every
   checkpoint ever produced carried the same flag, and it could not
   distinguish modified source from a run writing its own output;
5. **this module's own owner did it too**: ``run_g4`` gated its parameter-partition
   probe on how the Fisher map arrived, so a caller passing a bound map got a
   ``COULD_NOT_RUN`` whose stated reason was not the actual reason.

The fourth appeared *inside the mechanism built to catch stale artifacts*, and
the fifth inside the gate machinery itself — which is the sharpest possible
argument that this belongs in the falsification machinery rather than in a
comment.  Two variants are worth naming, because neither is "a flag that cannot move":
an instrument can fail by reporting a **reason that is not the actual reason**
(a discrimination failure about causes), and a *correct* instrument can be
**selected after the fact** from among several — the measurement-choice
variant, where the bias is in the choosing rather than in the instrument.

The rule this module enforces:

    **For every guard or provenance field a claim relies on, there must exist
    an input under which it reads differently. If there is not, it is
    decoration, and it must be labelled as such rather than reported.**

:func:`audit_instruments` runs each registered instrument over at least two
labelled inputs and FAILs any whose reads are all identical.  The audit has its
own negative control in ``tests/bench``: a deliberately constant instrument
must make it fail.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .report import (
    ClaimManifest,
    ClaimReport,
    Metric,
    SubCheck,
    could_not_run,
    source_dirty_entries,
)

__all__ = [
    "SeedStability",
    "seed_stability",
    "Instrument",
    "UninformativeField",
    "KNOWN_UNINFORMATIVE",
    "default_instruments",
    "audit_instruments",
]


# ==========================================================================
# the register of fields known not to discriminate
# ==========================================================================
@dataclass(frozen=True)
class UninformativeField:
    """A field that is recorded but cannot vary, with the fix."""

    name: str
    reads: str
    why_it_cannot_discriminate: str
    remedy: str
    found_by: str
    owner: str
    still_reported: bool = False
    #: measured recurrence in this project, where one exists
    recurrence: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "reads": self.reads,
            "why_it_cannot_discriminate": self.why_it_cannot_discriminate,
            "remedy": self.remedy,
            "found_by": self.found_by,
            "owner": self.owner,
            "still_reported": self.still_reported,
        }


KNOWN_UNINFORMATIVE: tuple[UninformativeField, ...] = (
    UninformativeField(
        name="git_sha() -dirty suffix (whole-tree scope)",
        reads="always '-dirty' during any run",
        why_it_cannot_discriminate=(
            "the run writes tracked output (reports/training/train_main.log, "
            "reports/training/scwbd-001-beta_train.jsonl), so git status --porcelain is "
            "never empty while a run is in flight; the flag therefore cannot separate "
            "'source was modified' from 'the run wrote its own log', and every checkpoint "
            "the project has produced carries it"
        ),
        remedy=(
            "scope the check to source paths and record the offending PATHS rather than a "
            "boolean: scwbd.bench.report.source_dirty_entries(SOURCE_PATHS)"
        ),
        found_by="agent Turing",
        owner="training (agent I)",
        still_reported=False,
    ),
    UninformativeField(
        name="run_g4 parameter_partition COULD_NOT_RUN reason (agent J's own bug)",
        reads="'theta_index / nuisance_index not supplied' whenever fisher was passed bound",
        why_it_cannot_discriminate=(
            "the partition probe was gated on auto_probed, which is only set when fisher is "
            "None; a caller passing a BOUND fisher map -- the only usable form, since bare "
            "expected_fisher needs u/cfg/proto -- skipped the probe entirely, so the gate "
            "reported a reason that was not the actual reason and never reached "
            "fisher_information at all"
        ),
        remedy=(
            "drop the auto_probed guard: the probe returns a Dependency that reports its own "
            "unavailability, so nothing is lost. Regression-tested by "
            "test_g4_resolves_the_partition_from_agent_h_even_for_a_bound_fisher"
        ),
        found_by="agent Fisher (running the gate end to end)",
        owner="bench (agent J)",
        still_reported=False,
    ),
    UninformativeField(
        name="exact-name gradient permission matching under torch.compile",
        reads="'permission matched' on CPU, silently matches nothing on CUDA",
        why_it_cannot_discriminate=(
            "torch.compile renames parameters, so a permission keyed on an exact parameter "
            "name matches an empty set and the source appears authorised while updating "
            "nothing"
        ),
        remedy=(
            "N1's gradient.unmatched_permission_patterns metric fails when a permission "
            "pattern matches nothing; run it against the compiled module, not only the "
            "eager one"
        ),
        found_by="agent Turing",
        owner="foundation (agent I) / compiler (agent A)",
        still_reported=True,
    ),
    UninformativeField(
        name="composite training loss used to judge a learning-rate change",
        reads="whichever direction the most recent action needs, at short series lengths",
        why_it_cannot_discriminate=(
            "this is the MEASUREMENT-CHOICE variant: not an instrument incapable of varying, "
            "but the wrong instrument selected from among available ones. Composite loss "
            "during warmup mixes objectives whose weights are still moving, so it moved "
            "twice in the direction that made a just-taken decision look correct where "
            "sim_forecast_nll did not move at all. Both readings were withdrawn. Selecting "
            "the metric AFTER seeing the curves is the failure; the metric itself is fine "
            "for what it is for"
        ),
        remedy=(
            "pre-commit the judging metric while it does not favour you, and have a "
            "different party return the verdict: scwbd.bench.adjudication, where secondary "
            "metrics are recorded and structurally cannot change the outcome"
        ),
        found_by="agent Turing (self-reported, unprompted)",
        owner="training (agent I) / bench (agent J) adjudicates",
        still_reported=True,
    ),
    UninformativeField(
        name="leakage_checked=True, hard-coded on every source card",
        reads="'the leakage audit passed' -- identically to 'no audit ever ran'",
        why_it_cannot_discriminate=(
            "THE ABSENCE VARIANT IN THE PROVENANCE LAYER, and it is worse than the cached- "
            "artifact cases above. compiler_bridge.py:1351 sets leakage_checked=True "
            "unconditionally on every observation SourceCard, with no reference to whether "
            "any audit ran -- and the trainer never ran one: train.py:335 calls "
            "participant_split directly and never calls leakage_check, while the correct "
            "routine in realdata.py splits, audits and RAISES on failure. So the field "
            "asserts a property nothing established, and the schema resolves the ambiguity "
            "in the FAVOURABLE direction by construction. "
            "The lesson is not carelessness: leakage_checked=True was TRUE WHEN WRITTEN, for "
            "a pipeline that later stopped guaranteeing it, and nothing forced "
            "re-examination. That is staleness in an ASSERTION rather than in an artifact, "
            "and it is the worse case -- a stale artifact can be re-derived, a stale "
            "assertion has no source to re-derive from"
        ),
        remedy=(
            "POLICY, and it is what kept bench clean here: AN AUDIT MUST RECOMPUTE THE "
            "PROPERTY, NEVER READ AN ASSERTION OF IT. scwbd.bench.leakage consumes agent "
            "Ada's GroupedSplitter and leakage_audit and re-derives grouping from immutable "
            "lineage records on every run; it reads no provenance flag anywhere. The same "
            "idea one layer up is finalize()'s refusal to let a numerical check PASS without "
            "recording what it measured"
        ),
        found_by="agent Turing (gating Stage III before real EEG entered any loss)",
        owner="foundation (agent I) + compiler bridge; bench audited itself and is unaffected",
        still_reported=False,
        recurrence=(
            "BENCH IMPACT: NONE. leakage_checked appears nowhere in scwbd/bench/** or "
            "tests/bench/**; no gate, ablation or Appendix-D audit consumes it, and no PASS "
            "verdict depends on it. All twelve Appendix D rows are additionally "
            "COULD_NOT_RUN for want of records, so there is no leakage verdict on the "
            "scoreboard to requalify in either direction"
        ),
    ),
    UninformativeField(
        name="a degraded path that records its degradation in a field nobody reads",
        reads="normal operation, with the degradation noted where it will not be seen",
        why_it_cannot_discriminate=(
            "participant_split falls back to a deterministic HASH split when Ada's splitter "
            "fails to import or adapt, setting participant_split_backend accordingly but not "
            "warning loudly. A hash split is not lineage-aware grouping -- R10 requires "
            "grouping by immutable lineage BEFORE splitting -- so the fallback silently "
            "violates the refusal the split exists to honour, and records that fact only in "
            "a field nobody reads. A degraded path whose degradation is invisible is "
            "functionally a silent one"
        ),
        remedy="refuse rather than warn; a fallback that cannot honour R10 must not run",
        found_by="agent Turing",
        owner="foundation (agent I)",
        still_reported=False,
    ),
    UninformativeField(
        name="a gate whose threshold is read from the artifact it judges",
        reads="PASS, at whatever bound the judged party currently declares",
        why_it_cannot_discriminate=(
            "THE MUTABLE-THRESHOLD VARIANT, found by bench re-running N9 rather than "
            "accepting the relayed verdict. N9 reads discrepancy_fraction from "
            "scwbd.runtime AT RUN TIME rather than copying it -- adopted as an anti-staleness "
            "measure, and it does prevent staleness. It also makes the threshold editable by "
            "the party being judged, AFTER seeing the result, with no record in the gate of "
            "what the bound was. Agent Faraday measured max_relative_overestimate = 1.0628904 "
            "against a DECLARED bound of 0.8 and correctly reported FAIL. Bench re-ran it and "
            "measured 1.0628904 -- the physics reproduces EXACTLY -- against a bound of 2.29, "
            "and the gate returned PASS. The runtime now declares "
            "solution_discrepancy_fraction = (0.0, 1.35) composed with a geometry term. "
            "The gate cannot distinguish 'the approximation is inside its bound' from 'the "
            "bound was moved until the approximation fit', which is precisely agent "
            "Hodgkin's criterion between REPAIRING an instrument and ACCOMMODATING a failure"
        ),
        remedy=(
            "read the threshold at run time AND record it in the report, then FAIL when it "
            "moves without a recorded, dated justification. Anti-staleness and "
            "anti-mutability are different requirements and this design bought the first by "
            "surrendering the second. Note the widening may be entirely justified -- the "
            "derivation is documented at scwbd/runtime/backends.py:271 -- but a gate that "
            "cannot tell a justified widening from an accommodating one is not adjudicating"
        ),
        found_by="agent J (bench), by re-running rather than relaying",
        owner="bench (gate design); intervene + runtime (the bound)",
        still_reported=True,
        recurrence=(
            "SUBSTANTIALLY ANSWERED, and by the judged party asking for a STRICTER bound. "
            "Agent Asimov did not widen the interval -- widening would have preserved the "
            "actual defect with more headroom -- they SPLIT it: solution_discrepancy_fraction "
            "(0.0, 1.35) MEASURED against the exact solution of the same geometry, "
            "geometry_discrepancy_fraction (-0.4, 0.4) a DECLARED prior, and "
            "discrepancy_fraction derived and never settable. They then asked that N9 be "
            "judged against the measured term alone, because testing an approximation's own "
            "error against an interval that also carries a geometry prior is the very "
            "conflation the gate exists to catch, one level up. BENCH ENDORSES: against "
            "(-0.4, +2.29) the gate passes trivially; against (0, 1.35) it is a real test of "
            "a real claim, and 1.06289 <= 1.35 still passes. What would CLOSE this row "
            "entirely: record the bound used IN the report and fail when it moves without a "
            "dated justification. Splitting removed the conflation; it did not yet make the "
            "threshold immutable"
        ),
    ),
    UninformativeField(
        name="a check that is sound, and too expensive to be invoked",
        reads="green, on the rare occasions anybody runs it",
        why_it_cannot_discriminate=(
            "THE COST VARIANT, reached by a different route from every row above. The check "
            "CAN fire, measures the right quantity and has a reference class; it simply "
            "costs enough that people stop running it, and a check that is not invoked "
            "reaches the same end state as one that cannot fire. Agent J introduced this "
            "one: auto-wiring the field gates into run_numerics_suite bought a real "
            "correctness property -- a default run can no longer silently revert a verified "
            "verdict to COULD_NOT_RUN -- and paid for it by running FDTD marches and BEM "
            "solves on every test invocation, taking one subset from seconds to 469.85 s "
            "MEASURED. I traded a correctness property for a runtime cost without checking "
            "the size of the cost"
        ),
        remedy=(
            "the remedy differs from every other row here, which is why it needs its own: a "
            "guard that CANNOT fire is fixed by redesigning the check, one that WILL NOT BE "
            "INVOKED by making it cheap enough to run. Keep the guarantee and drop the cost "
            "-- never weaken the guarantee to fit the clock. scwbd.bench.solver_cache keys "
            "on a sha256 of the solver MODULE'S SOURCE TEXT (not its path, import name, "
            "mtime or id -- each of those can hold constant across a change to the solver, "
            "which is the failure the auto-wiring existed to prevent, one layer down), and "
            "records hits in the report because an invisible cache hit is indistinguishable "
            "from a fresh solve"
        ),
        found_by="agent J (bench), self-reported after measuring its own suite",
        owner="bench (agent J)",
        still_reported=False,
    ),
    UninformativeField(
        name="an intervention compared at unmatched INPUT ENERGY",
        reads="decisive -- 9.3x, 28x and 6.9x on theta-profile lambda-min",
        why_it_cannot_discriminate=(
            "the matched-capacity discipline recurring in a new dimension: not parameters, "
            "not compute, but INPUT ENERGY. Agent Fisher measured the impulse's apparent "
            "gain and then measured it again at matched input energy: 0.839, 0.839, 1.059 -- "
            "WORSE in two regimes and marginal in the third. Spending the same energy on the "
            "background drive buys MORE theta information than concentrating it in one "
            "impulse. The unmatched ratio is measuring the energy, not the perturbation, and "
            "it is the number that would have been quoted"
        ),
        remedy=(
            "G4 now carries a MANDATORY input_energy_matched sub-check and refuses to report "
            "the unmatched ratio alone, exactly as it refuses an unmatched parameter count. "
            "Negative control: test_g4_fails_when_the_impulse_gain_is_entirely_input_energy"
        ),
        found_by="agent Fisher (measured and reported against their own design)",
        owner="infer (agent Fisher); bench enforces",
        still_reported=True,
    ),
    UninformativeField(
        name="a bound justified by the intended use rather than by what the code admits",
        reads="a tighter, more flattering interval, defensible on every biological ground",
        why_it_cannot_discriminate=(
            "agent Asimov chose 1.35 over a suggested 1.10 because the measured error at a "
            "60 mm head with 40 mm standoff is 1.3204, and DECLINED to justify the smaller "
            "number by arguing that 60 mm is sub-adult. It is sub-adult -- adult radii are "
            "~80-100 mm where the error is 0.74-0.89 -- but NOTHING IN HeadModel ENFORCES AN "
            "ADULT RADIUS. A bound justified by what the caller is expected to do cannot "
            "discriminate the case where the caller does something else, and the code admits "
            "that case. This is the difference between a bound on the physics and a bound on "
            "the intended use"
        ),
        remedy=(
            "the bound must cover what the CODE ADMITS, not what biology suggests -- or the "
            "code must enforce the assumption the bound relies on. Corollary applied in the "
            "same change: a declared-but-UNCHECKED envelope is the same defect one level up, "
            "so outside radius >= 60 mm / standoff <= 40 mm the fallback now raises "
            "DiscrepancyBoundNotEstablished and the runtime answers Defer with all four "
            "quantities Unresolved -- never a number under a bound nobody established"
        ),
        found_by="agent Asimov (declined the flattering justification unprompted)",
        owner="robotics bridge (agent Asimov)",
        still_reported=False,
    ),
    UninformativeField(
        name="a validation case on which the mechanism under test is degenerate",
        reads="3.8e-12 -- round-off. A perfect result, from a correct test, on a real coil",
        why_it_cannot_discriminate=(
            "THE DEGENERATE-TEST-CASE VARIANT, and it is the most dangerous entry in this "
            "register because nothing about it looks wrong. The fallback field model's "
            "tangential-projection approximation is EXACT for a circular coil: for loops "
            "coaxial with the head radius the primary vector potential is purely azimuthal, "
            "phi-hat . r-hat = 0 everywhere, the Neumann data vanishes identically, there is "
            "NO secondary field, and the projection is the exact answer. A validation suite "
            "that used a circular coil -- the easier, more symmetric, more obvious phantom -- "
            "would have certified the approximation as perfect. "
            "CRUCIALLY THIS IS NOT THE RESOLUTION VARIANT: the error is a function of source "
            "SYMMETRY, not of a discretisation parameter. Nothing converges to reveal it and "
            "no amount of refinement finds it, because on that case there is no error to "
            "refine away. Only changing the coil does"
        ),
        remedy=(
            "agent Faraday's corollary to the guard rule, and it generalises past physics: "
            "CHOOSE THE TEST CASE BY WHAT BREAKS THE MECHANISM, NOT BY WHAT IS CONVENIENT TO "
            "COMPUTE. Ask which symmetry of the test case the mechanism under test is blind "
            "to, then break it. N9 sweeps both the degenerate circular case and the "
            "figure-eight case so the exact result cannot be quoted alone"
        ),
        found_by="agent Faraday",
        owner="intervene (agent Faraday); the lesson is everyone's",
        still_reported=True,
        recurrence=(
            "distinct from every other row: the instrument varies, measures the right "
            "quantity, has adequate resolution and was chosen before the data -- and the "
            "TEST CASE annihilates the effect"
        ),
    ),
    UninformativeField(
        name="uniform-mesh error as a convergence indicator at contact geometry",
        reads="falling, then rising, then falling again under refinement",
        why_it_cannot_discriminate=(
            "on uniform meshes at contact, 80 -> 320 -> 1280 -> 5120 panels give errors "
            "1.061 / 1.506 / 0.171 / 0.042: refining from 80 to 320 makes the answer WORSE. "
            "A user watching the error fall between two of those points cannot distinguish "
            "convergence from a non-monotone excursion, so the metric cannot serve as the "
            "convergence indicator it is being read as. A NEW variant: not an instrument "
            "that cannot vary, and not the wrong instrument chosen -- an instrument whose "
            "variation is not monotone in the thing it is taken to track"
        ),
        remedy=(
            "refine where the error actually lives (graded_icosphere refines only panels "
            "under the source) and REFUSE outside the validated envelope at the solver "
            "(ChargeBEM.assert_resolves_sources) rather than leaving the judgement to a "
            "caller watching a number"
        ),
        found_by="agent Faraday",
        owner="intervene (agent Faraday)",
        still_reported=True,
    ),
    UninformativeField(
        name="a capability probe that branches on failure instead of asserting on success",
        reads="'connected' -- indistinguishably from 'silently fell back'",
        why_it_cannot_discriminate=(
            "THE SILENT-ADAPTER VARIANT, and it has THREE independent instances in this "
            "project, all in probe/adapter layers, all found only when somebody exercised "
            "the path end to end: (1) torch.compile renamed parameters so exact-name "
            "gradient permissions matched nothing; (2) agent Cajal's graph_controls probe; "
            "(3) the runtime probed for a solve_efield symbol THAT NEVER EXISTED UNDER THAT "
            "NAME, so it ran its own internal physics while presenting as though it consumed "
            "the gated solver -- every earlier runtime field number came from unvalidated "
            "physics. A working fallback and a working connection produce the same "
            "observable, so the probe's negative result carries no signal"
        ),
        remedy=(
            "a capability probe must ASSERT ON SUCCESS, not merely branch on failure, and "
            "something must exercise the wired path. Agent Asimov's CoilFrameBinding test "
            "is the pattern: drive the bridge with an identity binding and prove the "
            "upstream R06 guard FIRES, which makes the binding known load-bearing rather "
            "than decorative"
        ),
        found_by="agents Turing, Cajal and Asimov independently",
        owner="every module boundary",
        still_reported=True,
        recurrence="three independent instances, all in probe/adapter layers",
    ),
    UninformativeField(
        name="a metric that scores 1.00 because it measures its own definition",
        reads="a perfect score, at every percentile",
        why_it_cannot_discriminate=(
            "a normaliser candidate scored 1.00 at EVERY percentile because for that "
            "estimator std(z) = std(x)/rms(sd) == 1 identically, by construction. The metric "
            "was not measuring the candidate; it was restating the candidate's definition. "
            "It nearly selected the shipped normaliser"
        ),
        remedy=(
            "STANDING RECOMMENDATION: a perfect score is a reason to check whether the "
            "metric COULD have failed, not a reason to adopt the candidate. Ask what input "
            "would have produced a different number"
        ),
        found_by="agent Turing",
        owner="training (agent I)",
        still_reported=True,
    ),
    UninformativeField(
        name="a preregistration calibrated against a defective instrument",
        reads="as a commitment, while silently changing difficulty",
        why_it_cannot_discriminate=(
            "condition 2 ('running-min sim_forecast_nll < 1.0 by step 900') was chosen by "
            "looking at pre-fix numbers. A normaliser defect inflated ~5.9% of windows by "
            "10-767x; fixing it moved the metric's scale by two orders of magnitude. The "
            "threshold's VALUE never moved -- pre-fix it demanded a 99.5% descent from "
            "184.3, post-fix a ~41% improvement from 1.692. The bar is the same number and "
            "a different test. A PREREGISTRATION INHERITS THE DEFECTS OF THE INSTRUMENT IT "
            "WAS CALIBRATED AGAINST, and freezing it in advance makes that inheritance "
            "HARDER to see, not easier -- a genuine limitation of the technique, discovered "
            "by using it properly"
        ),
        remedy=(
            "when the instrument a preregistration was written against is found defective, "
            "the preregistration does not become WRONG, it becomes UNINTERPRETABLE. Report "
            "it as uninterpretable rather than re-setting it; re-setting substitutes the "
            "experimenter's later judgement for their earlier commitment, which is what "
            "preregistration exists to prevent -- and that holds REGARDLESS OF DIRECTION. A "
            "harder bar is not a cleaner one"
        ),
        found_by="agent Turing (caught that the fix left the bar contaminated)",
        owner="training (agent I); bench adjudicates",
        still_reported=True,
        recurrence=(
            "THE SHARPEST ILLUSTRATION THIS REGISTER WILL GET, because nothing was skipped: "
            "condition 2 was preregistered, honoured, escalated when it fired, and formally "
            "adjudicated -- and it was STILL structurally incapable of discriminating a "
            "model that underperformed from a number that was never achievable. Every "
            "procedural safeguard was applied correctly to a guard that could not "
            "discriminate. Process rigour does not manufacture discriminating power; only a "
            "reference class does. Policy adopted from the ruling: A PREREGISTERED "
            "THRESHOLD WITH NO REFERENCE CLASS IS A GUESS WITH A TIMESTAMP. Remedy in "
            "force: scwbd.bench.adjudication.STAGE_II_BAR, set in matched-control form"
        ),
    ),
    UninformativeField(
        name="a caveat that does not change the claim",
        reads="as rigour, from every angle",
        why_it_cannot_discriminate=(
            "THE INERT-QUALIFICATION VARIANT. Nothing is broken and nothing is unmeasurable: "
            "the qualification is present, correct, and does no work. A periodicity finding "
            "was relayed WITH the aliasing caveat attached ('log_every=20, so only periods "
            "that are multiples of 20 are detectable') and the period was reported anyway. "
            "The caveat was true, was stated, and changed nothing about what was claimed -- "
            "so it could not have prevented the error it appeared to guard against"
        ),
        remedy=(
            "OPERATIONAL TEST: if the caveat were true, would the claim change? If not, the "
            "caveat is ornament and the claim is unearned. Apply it before relaying, not "
            "after"
        ),
        found_by="agent Turing (self-reported, on its own withdrawn finding)",
        owner="everyone",
        still_reported=True,
    ),
    UninformativeField(
        name="a threshold test whose noise exceeds the effect it must detect",
        reads="pass or fail depending on which random stream it drew",
        why_it_cannot_discriminate=(
            "THE VARIANCE VARIANT, and it is distinct from every row above: this instrument "
            "DOES vary and DOES measure the right quantity. Its variance simply exceeds its "
            "effect size, so its output is dominated by the seed and a green reading is "
            "indistinguishable from a coin flip. "
            "test_cerebellum_learns_a_forward_model asserted errs[-1] < 0.5*errs[0] on two "
            "single 16-sample batches with errs[0] taken AFTER learning had begun -- a noisy "
            "self-comparison against a moving baseline. Across 8 seeds the single-sample "
            "ratio ran 0.27-0.79 and passed 4/8, while the robust window-mean ratio ran "
            "0.53-0.71 and failed the bar in ALL 8. CONSEQUENCE: any prior green run of that "
            "test on CUDA was ~50/50 noise and must NOT be counted as historical evidence "
            "that the cerebellar forward model met that bar"
        ),
        remedy=(
            "run the verdict across seeds and check it is STABLE (scwbd.bench.instruments."
            "seed_stability); then replace the self-comparison with matched controls rather "
            "than relaxing the bar -- lr=0 and shuffled targets gave learned error at 14-24% "
            "of either control across all 8 seeds, a 4-7x reduction with 2x margin instead "
            "of a boundary. Agent Hodgkin drew the line that matters: 'I did not move a "
            "tolerance to fit an observation... Had the only available fix been relax 0.5 to "
            "0.75 so the measured 0.71 passes, the right answer would have been to report "
            "the test as unsupported and leave it red.' That is the criterion between "
            "REPAIRING an instrument and ACCOMMODATING a failure"
        ),
        found_by="agent Hodgkin (self-reported, with the prior evidence retracted)",
        owner="dynamics (agent Hodgkin)",
        still_reported=True,
    ),
    UninformativeField(
        name="an author's own reading of their own result (both directions)",
        reads="whichever direction makes the most recent action look correct",
        why_it_cannot_discriminate=(
            "THE HUMAN VARIANT. Direction one: the author reaches for the flattering "
            "reading -- agent Turing overclaimed a justification twice within an hour, both "
            "times toward the just-taken action, and caught it itself. Direction two, which "
            "is the more dangerous: the REVIEWER under-audits evidence because it arrives "
            "well-argued. That converts one party's error into everyone's, and it "
            "STRENGTHENS as collaboration improves, because the better the colleague the "
            "less one checks"
        ),
        remedy=(
            "separate the party that measures from the party that returns the verdict "
            "(scwbd.bench.adjudication), pre-commit the metric while it does not favour "
            "you, and regenerate the numbers from raw series instead of reading anyone's "
            "table. Turing's principle: a conclusion nobody is trying to break is not a "
            "finding, it is a consensus"
        ),
        found_by="agent Turing (direction one, self-reported); the coordinator (direction two)",
        owner="everyone; bench adjudicates",
        recurrence=(
            "MEASURED, not merely described: direction two has instantiated three times in "
            "this project, all by the party who wrote it into this register. Three structural "
            "findings were relayed to bench untested -- a HEAD-relative provenance claim, a "
            "truncated loss series, and a periodicity finding -- and all three were later "
            "retracted BY THEIR AUTHOR rather than caught by the reviewer. The author caught "
            "two of its three overclaims itself; the reviewer caught none by testing. "
            "Adopted protocol: when handed a structural finding, ask what would falsify it "
            "and whether that test has been run, BEFORE relaying it. "
            "FOURTH INSTANCE, and the first where the defect was in the STATISTIC rather "
            "than the claim: a relayed '2.21% mean relative difference' was the mean of "
            "|differences|; the signed mean is 1.72%, the larger figure counting the one "
            "step where the treatment was BETTER as though it were worse. Relayed without "
            "being recomputed. "
            "SUB-CASE supplied by the coordinator against itself, and it is the sharpest "
            "form: accepting a GENEROUS argument from the party it DISADVANTAGES is the "
            "same failure as accepting a flattering one from the party it favours. Agent "
            "Turing offered an asymmetry argument that cut against its own artifact; it was "
            "endorsed as sharp and relayed, and it was wrong on ground Turing themselves had "
            "established two messages earlier. THE AUDIT WAS OF THE DIRECTION OF THE "
            "INCENTIVE, NOT OF THE VALIDITY OF THE ARGUMENT. An argument that costs its "
            "author something is not thereby correct"
        ),
        still_reported=True,
    ),
    UninformativeField(
        name="systemd-run MemoryMax against CUDA unified memory",
        reads="memory.current ~8 GB against a 40 GB cap",
        why_it_cannot_discriminate=(
            "CUDA allocations on unified memory are not charged to the cgroup, so the cap "
            "is not binding and the reassuring number is measuring the wrong pool"
        ),
        remedy="measure the allocator's own accounting, and prove the cap binds by exceeding it",
        found_by="agent Turing",
        owner="training (agent I)",
        still_reported=False,
    ),
    UninformativeField(
        name="'allocated by PyTorch' at OOM",
        reads="always equal to the ceiling",
        why_it_cannot_discriminate=(
            "at the moment of OOM the allocated figure is pinned to the limit by "
            "construction, so it cannot distinguish batch-linear from batch-independent "
            "growth — the question the number was consulted to answer"
        ),
        remedy="sweep batch size and fit the growth curve; a single reading at OOM cannot",
        found_by="agent Turing",
        owner="training (agent I)",
        still_reported=False,
    ),
)


# ==========================================================================
# the standing diagnostic: is the verdict stable across seeds?
# ==========================================================================
@dataclass(frozen=True)
class SeedStability:
    """Whether a threshold test returns the same verdict across random streams."""

    name: str
    verdicts: dict[int, bool]
    values: dict[int, float] = field(default_factory=dict)

    @property
    def n_pass(self) -> int:
        return sum(1 for v in self.verdicts.values() if v)

    @property
    def stable(self) -> bool:
        return len(set(self.verdicts.values())) <= 1

    @property
    def summary(self) -> str:
        if self.stable:
            return (f"stable across {len(self.verdicts)} seeds "
                    f"({'pass' if self.n_pass else 'fail'} every time)")
        return (f"UNSTABLE: {self.n_pass}/{len(self.verdicts)} seeds pass. A green reading "
                "from this test is indistinguishable from a coin flip and must not be "
                "counted as evidence.")

    def metric(self) -> Metric:
        return Metric(
            name=f"seed_stability.{self.name}.pass_fraction",
            value=self.n_pass / max(len(self.verdicts), 1),
            kind="audit", exact=True,
            note=self.summary + (f" values: {self.values}" if self.values else ""),
        )


def seed_stability(fn: Callable[[int], Any], seeds: Sequence[int],
                   *, name: str = "test") -> SeedStability:
    """Run a threshold test across seeds; an unstable verdict carries no information.

    ``fn(seed)`` returns either a bool verdict or ``(verdict, value)``.  This is
    the diagnostic agent Hodgkin used to catch a test that was passing on RNG
    luck, and it generalises to **every threshold test in this repository**: if
    the verdict moves with the seed, the effect is smaller than the noise and a
    pass means nothing.
    """
    verdicts: dict[int, bool] = {}
    values: dict[int, float] = {}
    for s in seeds:
        out = fn(int(s))
        if isinstance(out, tuple):
            verdicts[int(s)], values[int(s)] = bool(out[0]), float(out[1])
        else:
            verdicts[int(s)] = bool(out)
    return SeedStability(name=name, verdicts=verdicts, values=values)


# ==========================================================================
# instruments
# ==========================================================================
@dataclass
class Instrument:
    """A guard, plus inputs under which it must read differently."""

    name: str
    description: str
    read: Callable[[Any], Any]
    inputs: Mapping[str, Any]
    #: what a non-discriminating reading would mean
    consequence: str = "the field is decoration and must not be reported as evidence"
    mandatory: bool = True

    def evaluate(self) -> tuple[dict[str, str], int]:
        reads: dict[str, str] = {}
        for label, value in self.inputs.items():
            try:
                reads[label] = repr(self.read(value))
            except Exception as exc:
                reads[label] = f"<raised {type(exc).__name__}: {exc}>"
        return reads, len(set(reads.values()))


# -- the concrete instruments this bench relies on -------------------------
def _tmp_git_repo(*, modify_source: bool, write_output: bool) -> str:
    """A throwaway repo in one of three states, for the dirty-flag instrument."""
    d = tempfile.mkdtemp(prefix="scwbd-instr-")
    src = Path(d) / "scwbd"
    out = Path(d) / "reports" / "training"
    src.mkdir(parents=True)
    out.mkdir(parents=True)
    (src / "mod.py").write_text("x = 1\n", encoding="utf-8")
    (out / "train.jsonl").write_text('{"step": 0}\n', encoding="utf-8")
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    for cmd in (["git", "init", "-q"], ["git", "add", "-A"],
                ["git", "commit", "-qm", "base"]):
        subprocess.run(cmd, cwd=d, env=env, capture_output=True, timeout=30)
    if modify_source:
        (src / "mod.py").write_text("x = 2\n", encoding="utf-8")
    if write_output:                      # the run writing its own tracked log
        (out / "train.jsonl").write_text('{"step": 1}\n', encoding="utf-8")
    return d


def default_instruments() -> list[Instrument]:
    """Guards this bench relies on, each with a discriminating input pair."""
    from .matching import check_matched
    from .report import Interval, Metric as M
    from .statistics import smoothing_check

    class _Sized:
        def __init__(self, n: int) -> None:
            self.n = n

        def n_parameters(self) -> int:
            return self.n

    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, size=(200, 2))
    honest = y + rng.normal(0, 0.2, size=y.shape)
    flat = np.zeros_like(y) + y.mean(axis=0)
    effect = lambda a: float(np.std(a.reshape(a.shape[0], -1), axis=0).mean())

    return [
        Instrument(
            name="source_dirty_flag",
            description=(
                "Scoped source-dirty provenance: must read clean when the run writes its "
                "own tracked output, and dirty when source is modified. The whole-tree "
                "flag it replaces could not tell those apart."
            ),
            read=lambda repo: source_dirty_entries(("scwbd",), cwd=repo),
            inputs={
                "clean": _tmp_git_repo(modify_source=False, write_output=False),
                "run_wrote_its_own_log": _tmp_git_repo(modify_source=False,
                                                       write_output=True),
                "source_modified": _tmp_git_repo(modify_source=True, write_output=False),
            },
            consequence=(
                "provenance cannot distinguish a modified source tree from a run writing "
                "its own log; no reader may infer anything from it"
            ),
        ),
        Instrument(
            name="capacity_matching",
            description="Matched-capacity guard must reject an over-budget candidate.",
            read=lambda pair: check_matched(_Sized(pair[0]), {"b": _Sized(pair[1])}).matched,
            inputs={"equal": (100, 100), "candidate_3x_larger": (300, 100),
                    "candidate_smaller": (50, 100)},
            consequence="an unmatched comparison would be reported as a win",
        ),
        Instrument(
            name="interval_strict_threshold",
            description=(
                "Interval-strict metrics must reject a win whose interval straddles the "
                "threshold and accept one that clears it."
            ),
            read=lambda iv: M(name="d", value=0.02, kind="accuracy", interval=iv,
                              threshold=0.0,
                              require_interval_beats_threshold=True).passed,
            inputs={"noisy": Interval(-0.05, 0.09), "clean": Interval(0.01, 0.03)},
            consequence="a noisy difference would be reported as a reproducible gain",
        ),
        Instrument(
            name="smoothing_check",
            description=(
                "The §11.4 smoothing guard must fire on a flat predictor and not on an "
                "honest one."
            ),
            read=lambda pred: smoothing_check(
                arm_name="a", reference_name="b", y_true=y, pred_arm=pred,
                pred_reference=honest, effect=effect, seed=0, n_boot=80).smoothed_away,
            inputs={"flat_predictor": flat, "honest_predictor": honest},
            consequence="a model that won by destroying the effect would be preferred",
        ),
        Instrument(
            name="report_provenance_rule",
            description=(
                "finalize() must refuse a passing numerical report that does not record "
                "what it measured, and accept one that does."
            ),
            read=_provenance_rule_reads,
            inputs={"without_subject": False, "with_subject": True},
            consequence="a stale artifact could pass without recording how it was produced",
        ),
    ]


def _provenance_rule_reads(with_subject: bool) -> str:
    from .report import ReportDisciplineError

    rep = ClaimReport(
        manifest=ClaimManifest(
            claim_id="INSTRUMENT_PROBE", claim_text="probe", falsified_by="probe",
            consequence_if_failed="probe",
        ),
        subchecks=[SubCheck(name="ok", description="d", metrics=[
            Metric(name="x", value=1.0, kind="numerical", exact=True, threshold=0.5)])],
        kind="numerics",
    )
    if with_subject:
        rep.artifacts["subject"] = "probe"
    try:
        rep.finalize()
        return "accepted"
    except ReportDisciplineError:
        return "refused"


# ==========================================================================
def audit_instruments(instruments: Sequence[Instrument] | None = None, *,
                      seed: int = 0) -> ClaimReport:
    """FAIL any guard or provenance field that cannot read differently."""
    man = ClaimManifest(
        claim_id="N7_instrument_discrimination",
        claim_text=(
            "Every guard and provenance field this bench relies on has an input under "
            "which it reads differently, so a green reading is evidence rather than "
            "decoration."
        ),
        falsified_by=(
            "an instrument that returns the same reading on every input it was given — it "
            "is structurally incapable of reporting the discrimination it is consulted for"
        ),
        consequence_if_failed=(
            "Stop reporting the affected field, and stop gating on it. Replace it with a "
            "measurement that varies with the thing it claims to measure, or label it "
            "explicitly as decoration in the report where it appears."
        ),
        thesis_reference=(
            "generalises ARCHITECTURE.md §4 ('a gate that fails is a result') and this "
            "bench's negative-control discipline from gates to their instruments"
        ),
        acceptance_thresholds={"min_distinct_reads": 2},
        non_goals=[
            "This audit does not check that an instrument is CORRECT, only that it is "
            "capable of varying. A field that varies can still be wrong.",
        ],
        seed=seed,
    )
    try:
        instruments = list(instruments) if instruments is not None else default_instruments()
    except Exception as exc:  # pragma: no cover - environment dependent
        return ClaimReport(
            manifest=man,
            subchecks=[could_not_run("instruments", "Build the instrument set.",
                                     f"could not construct instruments: "
                                     f"{type(exc).__name__}: {exc}")],
            kind="instrument",
            artifacts={"subject": "scwbd.bench guards"},
        ).finalize()

    subs: list[SubCheck] = []
    table: dict[str, Any] = {}
    for inst in instruments:
        if len(inst.inputs) < 2:
            subs.append(could_not_run(
                inst.name, inst.description,
                f"only {len(inst.inputs)} input supplied; discrimination needs at least two",
                mandatory=inst.mandatory))
            continue
        reads, n_distinct = inst.evaluate()
        table[inst.name] = {"reads": reads, "distinct": n_distinct,
                            "consequence_if_constant": inst.consequence}
        subs.append(SubCheck(
            name=inst.name,
            description=inst.description,
            metrics=[Metric(
                name=f"{inst.name}.distinct_reads", value=float(n_distinct),
                kind="audit", exact=True, threshold=1.5,
                direction="greater_is_better",
                note="; ".join(f"{k} -> {v}" for k, v in reads.items())[:400],
            )],
            mandatory=inst.mandatory,
            falsified_by=f"identical reading on every input: {inst.consequence}",
        ))

    return ClaimReport(
        manifest=man, subchecks=subs, kind="instrument",
        artifacts={
            "subject": "the guards and provenance fields of scwbd.bench",
            "instrument_reads": table,
            "known_uninformative_fields": [u.as_dict() for u in KNOWN_UNINFORMATIVE],
        },
        notes=[
            "A green reading from an instrument that cannot vary is not evidence. Four such "
            "instruments have already been found in this project; the fourth was inside the "
            "mechanism built to catch stale artifacts.",
            "This audit checks capability to vary, not correctness. An instrument that "
            "varies can still be measuring the wrong thing.",
        ],
    ).finalize()
