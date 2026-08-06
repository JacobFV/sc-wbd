# Run notes — scwbd-001-beta

Live notes, written during the run so the loss curve does not have to be
reconstructed from memory afterwards.

**The artifact is the run launched 2026-08-06 02:22:10 at commit `94b6ddc`** —
sqrt-scaled learning rates **and** the fixed window normaliser.

Three runs were started and two superseded. Read the tables below with the
pipeline they belong to in mind, because **magnitudes are not comparable across
the normaliser fix**:

| # | launched | commit | LR | normaliser | ended |
|---|---|---|---|---|---|
| 1 | 01:08:59 | `7f18528` | 6.0e-4 | old | stopped, step 260 |
| 2 | 01:30 | `4be98fc` | 3.46e-4 | old | stopped, step 820 |
| **3** | **02:22:10** | **`94b6ddc`** | **3.46e-4** | **fixed** | **running** |

Runs 1 and 2 differ only in learning rate, so comparisons between *them* are
valid. Neither is comparable in magnitude to run 3, whose targets changed.

## Restart 2 — the learning-rate rescale

The stage learning rates (I 6e-4 → V 1e-4) were written when the config had
`batch: 192`. Batch was cut to **64** for device-memory reasons and the rates
were **not revisited** — 3× noisier gradient estimates at unchanged step size.
Caught at step ~200 while writing up the warmup spikes, i.e. three hours after
introducing it and while looking at something else.

Rates were rescaled by `sqrt(64/192) = 0.5774` and the run restarted from zero:

| stage | before | after |
|---|---|---|
| I_regional | 6.0e-4 | **3.46e-4** |
| II_interface | 4.0e-4 | 2.31e-4 |
| III_sliced | 3.0e-4 | 1.73e-4 |
| IV_assembly | 2.0e-4 | 1.15e-4 |
| V_individual | 1.0e-4 | 5.77e-5 |

Batch 64, 12 h cap, and everything else unchanged.

**My initial call was to continue, and it was wrong.** The reasoning I gave —
`grad_clip = 1.0` is active, so no hard batch can produce a divergent update —
is true and answers the wrong question. Clipping to unit norm prevents blow-up;
it does not make a too-high learning rate optimal. With 3× noisier gradients,
clipping means taking unit-norm steps in *noisier directions*, which degrades
converged quality quietly rather than failing loudly. I had ruled out the
dramatic failure and treated that as having ruled out the failure.

I also discounted my own strongest evidence: the loss **floor** rose as lr
crossed ~5e-4 (0.33 at 2–4e-4, ~0.60 at 6e-4). Floors should fall as training
proceeds. That is a within-run signal agreeing with the a-priori scaling
argument, and two independent lines agreeing is enough to act on when acting
costs ten minutes.

The decisive argument was cost asymmetry, not evidence strength: 10 minutes to
restart at step 260, against either ~45 minutes if the pre-committed step-900
trigger fired anyway, or a permanent Stage I caveat that **cannot be separated
from the architecture after the fact** — the exact attribution problem this note
was written to prevent.

The near-miss stays on the record even though it no longer affects the artifact.

### What the rescaled run showed, including a correction to my diagnosis

Early comparison at matched steps (same seed, so the same data order):

> ⚠ **Same-pipeline comparison** — both columns are the *pre-normaliser-fix*
> pipeline, differing only in learning rate, so these magnitudes ARE comparable.
> Do not compare them against post-fix numbers elsewhere in this file.

| step | old lr | old loss | new lr | new loss |
|---|---|---|---|---|
| 20 | 8.01e-5 | 0.509 | 4.62e-5 | **0.469** |
| 40 | 2.27e-4 | 0.331 | 1.31e-4 | **0.225** |
| 60 | 4.06e-4 | 0.339 | 2.34e-4 | **0.256** |
| 80 | 5.49e-4 | 3.717 | 3.17e-4 | **2.447** |

The floor improved at every matched step, which is the result the rescale was
predicted to produce.

**But the spike recurs at exactly step 80 in both runs.** `set_determinism(seed)`
fixes the shuffle, so step 80 draws the *same batch* in both. The spike is
therefore substantially **data-driven — a reproducible hard batch — not purely a
learning-rate instability.** Lowering the rate reduced its magnitude (3.717 →
2.447), as a smaller step into a bad direction should, but did not remove it.

### The intervention was correct; one of the two arguments for it was not

The rescale rested on two legs. **Only one survives.**

- **Survives — the a-priori scaling argument.** Rates written for batch 192 and
  run at batch 64 is a real mismatch regardless of what step 80 turns out to be.
- **Does not survive — "the loss floor rose because lr crossed ~5e-4".** That
  inference read a batch-ordering artifact as an optimiser effect. The same
  elevated readings appear at the same sampled steps in both runs because the
  same batches are drawn there.

**Correction, ~40 minutes later: the outcome evidence does not survive either.**

I wrote that floors "improved at every matched step" on a partial series that
stopped at step 140. Extending it to step 200, and comparing `sim_forecast_nll`
— the physically interpretable quantity — rather than the composite `loss`:

| step | old loss | new loss | old nll | new nll | Δnll |
|---|---|---|---|---|---|
| 40 | 0.3315 | 0.2248 | 2.877 | 2.793 | −0.084 |
| 60 | 0.3387 | 0.2555 | 1.803 | 1.990 | +0.187 |
| 80 | 3.7173 | 2.4470 | **20.943** | **20.980** | **+0.037** |
| 100 | 1.0154 | 0.8018 | 3.784 | 3.800 | +0.016 |
| 120 | 0.8417 | 0.7569 | 2.616 | 2.637 | +0.021 |
| 140 | 0.6321 | 0.6271 | 1.728 | 1.768 | +0.040 |
| 160 | 0.5939 | **0.6150** | 1.515 | 1.544 | +0.029 |
| 180 | 1.8306 | **1.9600** | 7.186 | 7.229 | +0.043 |
| 200 | 0.6002 | **0.6497** | 2.175 | 2.214 | +0.039 |

Two things follow, both against the rescale.

**1. The composite-loss advantage reversed.** Better at 7 matched steps, worse at
4 — and the 4 are the most recent 3 plus step 1. The early gap was warmup
transient, not learning.

**2. On `sim_forecast_nll` the two runs are nearly identical, with the rescaled
run consistently and very slightly worse** (+0.02 to +0.04 at every point from
step 100 on). Across a **1.73× difference in learning rate**, forecast quality
differs by under 2 %.

The step-80 row is the sharpest: `sim_forecast_nll` is **20.943 vs 20.980** —
indistinguishable. So my claim that lr "sets how badly the model is thrown by"
the hard batch was also wrong; it was read off the composite `loss` (3.717 vs
2.447), while the forecast NLL says the throw is *the same size at both rates*.
Whatever the composite-loss difference at step 80 is, it is not forecast quality.

### A finding in its own right: this model is nearly LR-insensitive in this regime

Buried in the retraction above is a result that is more interesting than the
decision that produced it.

**A 1.73× change in learning rate moved `sim_forecast_nll` by under 2 %** at every
matched step through 200 steps of Stage I — and by 0.2 % at step 80, the point
where the composite loss suggested a large difference.

Two runs differing only in learning rate, sharing a seed and therefore a data
order, tracked each other to within noise. In this regime the trajectory is
dominated by **the data order and the warmup schedule, not the step size.**

Consequences worth carrying forward:

- **It bounds what the rescale could ever have bought.** The cost-asymmetry
  argument for doing it was sound in form — ten minutes to remove a confound —
  but the expected benefit was smaller than either of us implied. The decision
  was cheap and defensible rather than clearly beneficial.
- **It suggests where the real levers are.** If LR is nearly inert here, tuning
  it further is not where Stage I quality comes from; batch size (data per step),
  corpus composition, and the warmup schedule are the candidates.
- **It should be re-tested at a wider ratio and later in training.** 1.73× over
  200 warmup steps is a narrow probe. Insensitivity may not hold into Stage III/IV
  where the loss surface differs, and a 10× sweep would say much more. Logged as a
  question for the next run, not a settled property.

### Where that leaves the decision

- The **a-priori scaling argument** is still the only surviving justification:
  rates written for batch 192, run at batch 64, is a real mismatch.
- The **diagnosis** (floor rose because lr crossed ~5e-4) was wrong.
- The **outcome evidence** does not support the rescale either. At 200 steps of
  8,700 it is roughly neutral, trending marginally negative on the metric that
  means something.

**Honest status: the rescale is not yet vindicated by anything measured.** It is
justified by an argument from first principles and is so far neither helping nor
meaningfully hurting. 200 steps is warmup and far too early to conclude it was
wrong — but it is also far too early to have called it right, which I did.

Recorded this way deliberately. **A decision written up as fully justified, when
its support later failed, is itself an uninformative instrument** — it reads the
same whether the reasoning was sound or lucky. I have now overclaimed the
justification for this decision twice, in the same direction, within an hour: the
tell is that both times I reached for the reading that made the last action look
correct. The metric to judge this on is `sim_forecast_nll` at the end of Stage I,
not the composite loss during warmup, and I am fixing that comparison now so it
cannot be chosen after the fact.

### A precision caveat on "exactly step 80"

`stage.log_every = 20`, so the loss series is **sampled every 20 steps**. "The
spike is at exactly step 80" overstates what the data can support: what is
observed is that *the step-80 sample point* is elevated in both runs. The
underlying hard batch could sit anywhere in roughly steps 61–80 and would be
invisible except where it lands on the sampling grid, and there may be other
spikes between grid points that were never seen at all.

The reproducibility claim is unaffected — same seed, same shuffle, same grid, so
the two runs are comparable at matched points — but the *localisation* is
±20 steps. My earlier phrasing claimed a resolution the instrument does not have.

### Step-80 investigation — method, to run at end of Stage I

Deferred to Stage I completion so it does not contend with the live run.

1. **Localise properly.** Re-run steps ~40–120 with `log_every: 1` on the CI-sized
   config, same seed, to find the true spike step rather than the grid point.
2. **Identify the batch.** Reconstruct the deterministic sampler order —
   `set_determinism(20260805)`, `SimCorpus(..., trajectory_subset="train",
   val_fraction=0.05, seed=20260805)`, `DataLoader(..., batch_size=64,
   shuffle=True, drop_last=True)` — and map that batch's dataset indices to
   shard and backend.
3. **Classify.** If the batch is a shard boundary or a `linear_gaussian` block,
   this is a **corpus** property, not an optimisation one, and belongs in
   `corpus_composition.md` beside mechanisms A and B — plausibly a third
   mechanism.
4. **Generalise or don't.** If it is a shard boundary, check **every** shard
   boundary. One spiking boundary is an anecdote; all 37 spiking is a structural
   property of how the corpus was sharded, and a much more serious finding.

Pre-committed so the answer cannot be shaped after the fact: if boundaries spike
systematically, it goes in `corpus_composition.md` as mechanism C and into the
limitations list, whatever it does to the loss curve's appearance.

## Restart 3 — the normaliser fix, and why condition 2's clock resets

Stopped at **step 820** of Stage I and restarted from scratch on a fixed data
pipeline. Third and final restart.

**The decisive argument was not mine.** I framed this as "is the defect worth 35
minutes", and on that framing my own inclination (don't thrash) was suspect and I
said so. `main` reframed it: **condition 2 was being evaluated on a corrupted
signal.** The preregistered bar — running-min `sim_forecast_nll` < 1.0 by step
900 — was being tested against a loss computed from windows where 5.9 % carried
`max|z|` up to 2427, concentrated in `wilson_cowan`, which supplies ~76 % of
post-normalisation batch variance. **A preregistered test run on a known-corrupt
dominant signal is uninterpretable whichever way it comes out**, and 🛡️ Popper
could not have adjudicated it either.

That is not leniency toward condition 2. It is the opposite: it is what giving it
a *fair* test requires. The clock resets, the threshold does not move, and
`main`'s four pre-commitments stand verbatim.

Supporting reasons, in order: the defect is in the **loader**, so it would have
affected every epoch of all five stages (~8 h), not just Stage I; the cost is
~35 min against ~8 h remaining, ~7 %; and unlike the LR question — an a-priori
argument about a contested effect that measured under 2 % — this is a
**mechanism measured directly**, where the median collapses 111× while the peak
does not. Different epistemic class.

The fix, its validation, and the rejected `rms` candidate are documented in
`corpus_composition.md` under mechanism C. Bounded: worst `max|z|`
**2426.86 → 21.12**, p99 **740.60 → 10.27**, non-pathological windows returned
**bit-identical** (×1.000000). Covered by `tests/foundation/test_normaliser.py`,
which constructs the pathology directly and includes a premise-guard asserting
the *old* rule would have blown up on the same input.

**Two things kept separate on purpose, and they must stay separate in the final
report:** the normaliser defect is *measured and robust*; its link to the ~31 %
spike rate in `sim_forecast_nll` is *unestablished*. A "batch contains ≥1 extreme
window" model predicts 98 %, not 31 %. The rate can be reproduced by tuning the
extremeness threshold to roughly the top 0.5 %, which is precisely the
retrospective fitting that produced the retracted period-60 claim. It has not
been done and should not be, however tempting it becomes now that the fix is in.

### ⚠ Condition 2's threshold was calibrated on the corrupted metric

Recorded **before the bar resolves**, so this cannot be raised afterwards by
whichever party the outcome disfavours.

The restart fixed the *signal* condition 2 is evaluated on. It did **not** fix
the *threshold*, which I chose by looking at pre-fix numbers. Both were
contaminated by the same defect; only one was repaired.

> ⚠ **These two columns are values of different objectives.** The normaliser
> change altered the targets, so the magnitudes are **not comparable** across the
> restart and their difference says nothing about model quality. They are shown
> only to size the change in the *instrument*. Scale-free quantities (the spike
> *rate*) remain comparable; matched-step comparisons *within* one pipeline
> remain valid.

| | pre-fix | post-fix |
|---|---|---|
| `sim_forecast_nll` at step 1 | **184.338** | **1.692** |
| at step 20 | 14.445 | 1.571 |
| at step 40 | 2.793 | 1.550 |
| best reached | 1.459 (step 660) | — |

The inflated windows contributed enormous squared error to exactly this
quantity, so removing them deflated it by two orders of magnitude at the start
of training. Consequently:

- **pre-fix**, "< 1.0" required a ~99.5 % descent from 184.3, and the run never
  got below 1.459;
- **post-fix**, nll *begins* at 1.692 — better than the entire pre-fix run's
  best — and "< 1.0" requires roughly a 41 % improvement.

**The threshold's value is unchanged and has not been moved by me. Its difficulty
has changed materially.** A pass on the fixed pipeline is therefore **not**
equivalent to passing the bar as originally conceived, and must not be reported
as though it were.

This is the same shape as the fired trigger: a preregistered number that no
longer measures what it was written to measure. The difference is that this one
was caught *before* it resolved, which is the only reason it is not a dispute.

### RULING — threshold stays at 1.0, report in three separate layers

`main`'s call. **No new bar is set.** Any threshold chosen now would be chosen
with the post-fix trajectory in view (nll 1.692 at step 1, 1.550 by step 40) —
a bar wearing a preregistration's clothes, which would read in the report as more
rigorous than it is. **Manufacturing false rigour is worse than reporting an
uninterpretable result honestly.** Voiding it was also rejected: the commitment
was real and made in good faith, and the reason it stopped meaning what it meant
is itself worth preserving.

When step 900 resolves, report **all three layers, never merged**:

1. **The literal fact.** Did running-min `sim_forecast_nll` go below 1.0 by step
   900? **Yes or no, plainly, no qualifier.** The commitment is honoured to the
   letter.
2. **The interpretation.** That this is **not the test that was preregistered**,
   because the metric's scale changed by two orders of magnitude between the run
   the bar was written for and the run it judges — pre-fix "< 1.0" demanded a
   99.5 % descent from 184.3; post-fix it demands roughly 41 % from 1.692. Give
   the numbers so a reader can size the change themselves.
3. **The adjudication.** Whether layer 1 evidences anything **at all**, given
   layer 2. **🛡️ Popper decides. Not me, not `main`.** This falls under the
   existing pre-commitment that Popper rules on whether the bar was appropriate,
   now extended to whether it survives a change in the metric's scale.

Merging 1 and 2 would let a caveat quietly do the work of a result. Merging 2 and
3 would be grading my own homework.

**And the crossing must not be allowed to read as strength.**

> A comfortable pass on an easier bar is weaker evidence than a narrow pass on
> the intended one.

Running-min reached **1.396 by step 100** on the fixed pipeline — already below
the pre-fix run's best of 1.459, which took it 660 steps. If the bar is cleared
early and by a wide margin, that margin is a property of the **deflated metric**,
not of the model. The wider the pass, the more prominent layer 2 needs to be, not
less. Comfort is the thing most likely to be mistaken for evidence here.

I have no clean preference on the outcome and note that keeping the bar favours
the artifact while voiding it would have favoured me, since it retires a bar I
may be about to fail.

### Cross-run comparisons of absolute loss are no longer meaningful

The normaliser change altered the targets, so pre- and post-fix `loss` and
`sim_forecast_nll` are values of **different objectives**. Comparing their
magnitudes across the restart says nothing about model quality — I nearly
reported "post-fix loss is higher" as if it did.

Relative measures survive: the spike **rate** (fraction of steps above 3× the
running floor) is scale-free and remains comparable. Matched-step comparisons
*within* a single pipeline remain valid.

### Pre-committed wording for whatever the spike rate does

Written **before** the post-fix rate is known, because the outcome that would be
most flattering is also the most likely to be misreported.

- **If the spike rate falls:** report *"the spike rate fell from ~31 % to X %
  after the normaliser fix."* Do **not** report that the fix explained the
  spikes. The two runs differ in more than one respect, no causal link was ever
  established, and a "≥1 extreme window" model predicted 98 % rather than 31 %.
  A fall would be **consistent with** the fix and would not establish it.
- **If the spike rate is unchanged:** report that too, plainly. It would mean the
  normaliser defect — real, measured, and worth fixing on its own terms — was
  not what produced the spikes, and the cause remains unknown.
- **Either way**, the defect's justification stands on its own measurements
  (worst `max|z|` 2426.86 → 21.12) and needs no help from the spike rate.

Establishing the link properly would require holding everything else fixed and
varying only the normaliser, which is not what happened here.

## ⚠ CURRICULUM DEVIATION — Stage II ran 1320 effective steps, not 700

Caused by my own decision to stop mid-stage for the leakage gate. Recorded
because the final report must not describe this artifact as the designed
curriculum without qualification.

**Resume granularity is stage-level, not step-level.** `completed_stages`
records whole stages only (`train.py:722`), so an incomplete Stage II **replays
from step 1 with a fresh `OneCycleLR`**, carrying the model weights but
restarting the schedule.

```
step 600  lr 1.517e-05      <- annealed, pre-stop
step 620  lr 9.742e-06
--- resume ---
step   1  lr 9.296e-06
step 100  lr 2.310e-04      <- back to the Stage II maximum
```

| | designed | actual |
|---|---|---|
| Stage II steps | 700 | **1320** (620 + 700 replay) |
| Stage II LR schedule | one cycle | **two** — annealed to 9.7e-6, then ramped to 2.31e-4 |
| total curriculum | 8700 | **~9320** (+7.1 %) |

Also: `global_step` went **backwards** (1520 → 1501) and re-counts, so steps
1501–1520 appear twice in the log and `global_step` is no longer a clean index.
### ⏱ The wall deadline is BINDING at 14:22, not the recomputed one

`deadline = time.time() + max_wall_seconds` is recomputed on every resume
(`train.py:719`), so the 12 h budget silently restarted at 04:03 and would expire
at 16:03.

**That is an implementation artifact, not a grant.** The binding deadline is the
original **02:22 + 12 h = 14:22**, ruled by `main` and settled now rather than at
14:00 with two stages left. Remaining work projects to 09:30–11:40, so this costs
nothing — which is exactly why it is the right moment to fix it, while it is free.

A deadline that resets each time you stop is not a deadline; it is a per-segment
budget wearing one. Same shape as the resume granularity below: **an
implementation detail quietly redefining a commitment.**

### What this is a deviation *from* — corrected, in my own disfavour

I first wrote that the artifact "is no longer exactly SC-WBD-001-beta as
specified." **That overstates it, and the correction was `main`'s.**

Checked rather than accepted: `ARCHITECTURE.md` contains **no training step
counts at all** — `grep -E "[0-9]{3,4} steps|steps: *[0-9]|8700|max_wall"` returns
nothing. It specifies the model, its state, coupling, backends, heads, training
mixture and amortised posterior. **The 8700 came from `configs/scwbd_001_beta.yaml`
— my file.**

So this is a deviation from **my own config**, not from the architecture
contract. That is still a real deviation and is reported without softening — but
*"the artifact departs from its own config"* is a materially smaller claim than
*"the artifact departs from spec,"* and conflating them would overstate a fault
against the work. **Accuracy runs in both directions**; the same discipline that
forbids flattering the artifact forbids indicting it beyond the evidence.

The properties the architecture *does* constrain are intact: all five stages run,
each properly annealed, and Stage V — where individualisation lives and without
which **G5 cannot be tested at all** — is unaffected.

### Why this was not restarted a fourth time

- Stage II still ends **properly annealed**: the replay runs a complete cycle, so
  the end state is a well-annealed interface stage, not a truncated one.
- The deviation is 600 extra interface steps plus one warm restart — a known
  technique, not a pathology.
- The alternative (reload `stage_I_regional.pt`, run Stage II exactly once) costs
  ~30 min and discards 620 completed steps **to buy conformity to a step count
  rather than a property of the model.**

Stated with the conflict declared: I have already caused three restarts, so "do
not restart" is the convenient conclusion for me and should be discounted
accordingly.

### The structural finding, which outlives this run

**Stage-level resume granularity means any mid-stage stop silently rewrites that
stage's LR schedule.** Not a bug — a documented-by-implementation behaviour
nobody had reason to examine — but it makes *"resume is cheap"* **false for
anything stopped mid-stage**, and I had been reasoning as though it were true
every time I weighed stopping.

The honest options for future runs are **step-level resume**, or a rule that
**stops only happen at stage boundaries**. Queued, not done here.

## STAGE I RESULT — condition 2, reported in three layers

Stage I completed at step 900 (03:12, wall 2941 s). Stage II is running.

### Layer 1 — the literal fact

> **Running-min `sim_forecast_nll` over Stage I: 1.1200 (step 760).**
> **The preregistered bar was < 1.0 by step 900.**
> **NOT MET.**

No qualifier. The commitment is honoured to the letter. **Stage I did not meet its
own preregistered quality bar**, by 12.0 %.

> #### ⚠ Stated beside layer 1, not in a footnote: the one consideration that could overturn it
>
> `log_every = 20`, so 1.1200 is a minimum over **46 samples**, not over 900
> steps. **A minimum over sampled steps is an *upper bound* on the true running
> minimum** — the true value can only be **lower**, never higher.
>
> **The sampling bias therefore runs in favour of the model and against this
> verdict.** A step below 1.0 between samples cannot be excluded, and if one
> occurred the bar was met and layer 1 is wrong.
>
> What can be said: over the final 200 steps the logged series spans
> **1.120–1.292**, so reaching below 1.0 would require an excursion larger than
> any variation observed. Unlikely — **but not measured**, and "unlikely" is not
> the same claim as "did not happen". Per-step logging over a bounded window is
> the instrument change that would settle it.
>
> Placed here at 🛡️ Popper's direction. I had filed it as a caveat against my own
> result; it is in fact the more consequential half, because it is the only thing
> in the record that could reverse the stated fact.

### Layer 2 — the interpretation

**This is not the test that was preregistered**, and the reason is documented
above: the bar was chosen by looking at pre-fix numbers, and the normaliser fix
changed the metric's scale by two orders of magnitude.

| | run the bar was written for | run it judged |
|---|---|---|
| `sim_forecast_nll` at step 1 | 184.338 | 1.692 |
| what "< 1.0" demanded | a **99.5 %** descent | roughly **41 %** |
| running-min achieved | 1.459 (step 660, run stopped at 820) | **1.120** (step 760) |

> #### ~~So the bar was not met on the *easier* version of the test~~ — RETRACTED
>
> I argued that failing the easier test was a *stronger* statement than failing
> the original: a pass would have been confounded by the scale change, a failure
> would not be rescued by it. `main` endorsed it. **🛡️ Popper tested it and it
> does not hold.**
>
> An a-fortiori argument requires both tests ordered **on a common scale.** The
> normaliser fix changed the *targets*, so pre- and post-fix NLL are values of
> **different objectives**. A 99.5 % descent in metric A and a 41 % descent in
> metric B are not comparable, and therefore **"easier" is undefined.** The
> argument has no ground to stand on.
>
> **The refutation is a fact I established myself, two messages earlier**, when I
> flagged that cross-run absolute comparisons are meaningless because the targets
> changed — and then built an argument that required exactly the comparability I
> had just ruled out. Finding a constraint and failing to apply it to my own next
> claim is worse than never having found it, because the record shows I knew.
>
> Popper's closing line, kept because it is the general lesson: *it is generous,
> it cuts against its author's own artifact, and it is still wrong — which is
> exactly why it shouldn't have been accepted for being generous.*

**The correct statement of layer 2** is therefore narrower: the bar was calibrated
on a defective instrument, the instrument was then replaced, and **the two tests
cannot be ranked.** Not harder, not easier — **different**, with no defined
ordering.

**Q2 — the 12 % margin does not license "missed by a little."** It is 12 % on a
scale whose relation to the original is undefined. The result is neither stronger
nor weaker than failing the original would have been; it is a **different
conclusion**, and any phrasing implying near-miss is unsupported.

### Layer 3 — the adjudication (🛡️ Popper, returned)

**Layer 1 stands unqualified.** Beyond that:

- **Q2** — the 12 % margin does not license "missed by a little"; see above.
- **Q3** — the sampling bound runs *against* the verdict and is now stated beside
  layer 1 rather than as a caveat.
- **Q1/Q4 — the bar was inappropriate, and not because it was too high.**

> **A preregistered threshold with no reference class is a guess with a
> timestamp.**

To judge whether `< 1.0` was a reasonable ask of a **1.76 M-parameter** model, on
**37 simulated shards**, with **~40 % of its regional-timescale prior missing or
clamped**, you need a reference class: a capacity-matched baseline, a matched
control, a prior run at another budget. **None exists in this project.**

So condition 2 was preregistered, honoured to the letter, escalated when it
fired, and adjudicated by a third party — and was **structurally incapable of
discriminating** a model that underperformed from a number that was never
achievable. Every procedural safeguard worked; the instrument still could not
answer the question. **That is the sharpest entry in the register**, and it is not
a failure of preregistration hygiene but of preregistration *without a reference
class*, which the hygiene cannot supply.

**Consequence, and I have no input into it:** Popper is setting the **Stage II bar
as a matched control** rather than an absolute threshold — the correct move, since
both sides of a comparison move together under an instrument rescale. It will
arrive as a fait accompli with its reference class stated. I have not asked for
input and will not.

### Consequences, per `main`'s pre-commitments (unchanged, made before this resolved)

- **No third rescale.** Rate-invariance is established; a third application of the
  same remedy was ruled out in advance.
- **Training continues through Stages II–V regardless** — already underway. **G5
  cannot be tested at all without Stage V.**
- This is a finding about **this model, this corpus and this budget**, and
  **must not be written as a verdict on the architecture.**

### Final spike rate — pre-committed wording

**0 / 45 sampled steps** above 3× the running floor, against **31 %** pre-fix.

Per the wording fixed before the data existed: **the spike rate fell from ~31 % to
0 %.** It is **not** reported that the normaliser fix explained the spikes. Run 3
differs from runs 1 and 2 in **both** normaliser and learning rate, so nothing
here attributes the difference to either, and the "≥1 extreme window" model
predicted 98 % rather than 31 %. Attribution would require a fourth run varying
one factor, which `main` has declined for a process question.

The honest statement: **run 3 is smooth where runs 1 and 2 were not; the runs
differ in normaliser and learning rate, so this does not attribute the difference
to either.**

### The batch-composition investigation — standing revised

With **zero spikes post-fix**, the phenomenon no longer occurs in the artifact.
The investigation can only be run against the **superseded** runs, which makes it
**diagnostic history about a replaced pipeline, not a property of
SC-WBD-001-beta**. It therefore cannot become a "mechanism C" claim about this
model's training. Mechanism C in `corpus_composition.md` stands on its own direct
measurements and needs no support from the spike question.

## ⚠ Provenance stamp is INCORRECT — read this before citing the checkpoint

**Summary, stated first so it cannot be missed:**

| | |
|---|---|
| **status** | **stamp incorrect** |
| **damage** | **cosmetic** |
| **proof** | `git diff --stat 94b6ddc da05ad5 -- scwbd configs tests` → empty (branch `wt/turing`) |
| **corrected?** | **no** — correction would have cost ~10 min of training to replace a checkable diff with a weaker form of evidence |

The two-SHA diff is a **stronger** claim than the stamp: it states what code
produced the weights, which is what anyone actually needs. The stamp states which
commit was HEAD at an arbitrary moment. Later checkpoints repeat `da05ad5`
because `_SHA` is frozen — **consistently wrong is auditable**, and that is the
second-best outcome after correct.

**The checkpoint stamp says `da05ad5`. The run was launched at `94b6ddc`.**

`git_sha()` caches lazily on first call, which is the step-150 checkpoint save.
Between launch (02:22:10) and that checkpoint (02:32:20) I made three commits, so
HEAD had moved and the artifact recorded a commit made **ten minutes after it
started**.

**What is and is not affected:**

```
git log --oneline 94b6ddc..da05ad5      # 3 commits, all reports/ only
git diff --stat 94b6ddc da05ad5 -- scwbd configs tests   # EMPTY
```

The training source is **byte-identical** between the launch commit and the
stamped one. The stamp is wrong about *which commit*, and correct about *what
code produced the weights*. Every later checkpoint in this run will also say
`da05ad5`, since `_SHA` is now frozen — at least it is consistently wrong.

**Why it happened, which is the part worth keeping.** I wrote the section of
`decorative_guards.md` describing this exact failure. It says: *"Committing
**anything** before that checkpoint — including this very document — would have
stamped the run with a commit that did not produce it."* `main` then explicitly
reminded me to verify the checkpoint before committing further.

I then committed three times, because each felt safe: they touched only `reports/`
and could not change the model. **That reasoning is correct and answers the wrong
question.** The rule I had written was *commit nothing until the artifact freezes
its identity*. The rule I actually followed was *commit nothing that changes
source* — weaker, easier to satisfy, and not the one that protects the stamp.

Substituting a weaker rule that is easier to comply with, while believing you are
following the original, is how a written-down rule fails without anyone noticing
it has been abandoned. Having authored the rule made it *more* likely, not less:
I trusted my memory of it instead of re-reading it.

**Not restarting over this.** The remedy would cost ~10 minutes of training to
correct a stamp whose only error is provably cosmetic, and the checkable claim
above is stronger than the stamp would have been anyway. Recorded rather than
repaired, and the queued `git_sha()` follow-up now has a second, better
justification: **capture the SHA at process start, not lazily**, so the binding
moment is not a race against the author's own commits.

For 🛡️ Popper, the artifact's provenance is:

| | |
|---|---|
| launched from | **`94b6ddc`** (branch `wt/turing`) |
| stamped in `provenance.json` | `da05ad5deb5fa41142154b1a6c9bcf5fe6d06694-dirty` |
| training source | identical between the two — verify with the diff above |
| `-dirty` | expected; the run writes to tracked report files |

## Provenance of the artifact

Branch **`wt/turing`**. The run's commit is **`4be98fc`**
(`4be98fce8683654071e887b67f32bd3e90a09b3f`), recorded in
`checkpoints/scwbd-001-beta/provenance.json` at the step-150 checkpoint together
with the rescaled learning rates in the accompanying `config.yaml`.

The `-dirty` suffix on the stamp is a known artifact and does **not** indicate
modified source: the run writes to two *tracked* files
(`reports/training/train_main.log`, `…_train.jsonl`), so the flag is always set
during a run. See `reports/decorative_guards.md` row 4.

To check that no training source changed after the run began, diff **two
immutable SHAs** — both on `wt/turing`:

```
git diff --stat 4be98fc <sha-of-the-commit-being-checked> -- scwbd configs
```

**Corrected from an earlier, defective form of this claim.** It was previously
written as `git diff --stat 7f18528 HEAD -- scwbd configs`, which 🛡️ Popper could
not reproduce — correctly, because `HEAD` is a moving, worktree-local symbol and
`7f18528` is not an ancestor of `master` (merge base `4d617af`). Evaluated in
another worktree the diff is dominated by unmerged work on both sides and the
claim reads false. It was true where written and false where checked, and its
text did not say so. A verification claim has to name every referent immutably or
it is not checkable; see `reports/decorative_guards.md` row 6.

Note also that `7f18528` is the **superseded** run, not this one. The artifact is
`4be98fc`.

## Reading the outputs

**A `tail -F` monitor will replay a restored log as if it were live.**
`train_main.log` is a *tracked* file, so `git checkout -- reports/training/…`
rewrites it with the committed (old) contents, and any `tail -F` following it
emits that history as fresh events. This happened twice: once producing a wall of
long-fixed binding warnings, and once emitting **pre-fix training numbers with
`wall_s = 2196` for a run that was one minute old.** Both looked like the new run
regressing.

The tell is a value that cannot belong to the current run — a wall clock larger
than the elapsed time, or a metric that predates a fix. The check is to read the
live file directly and look for a marker only the current build emits (here,
`[mem] CUDA reserve capped at …`). Another instrument reading the wrong thing:
the monitor was correct about the bytes and wrong about what they meant.

**`reports/training/train_main.log` is the live source of truth, not the JSONL.**
`JsonlLogger.log()` writes to the file without flushing but prints with
`flush=True`, so `scwbd-001-beta_train.jsonl` lags stdout by a buffer. During
this run the JSONL sat at step 60 while the log was already at step 100, which
looked exactly like a stall and was not. Anyone monitoring should tail the log.

## Learning-rate warmup and the loss spikes

| step | lr | loss | sim_forecast_nll |
|---|---|---|---|
| 1 | 2.4e-5 | 1.000 | 184.3 |
| 20 | 8.0e-5 | 0.509 | 7.96 |
| 40 | 2.3e-4 | 0.331 | 2.88 |
| 60 | 4.1e-4 | 0.339 | 1.80 |
| **80** | **5.5e-4** | **3.717** | **20.94** |
| 100 | 6.0e-4 | 1.015 | 3.78 |
| 120 | 6.0e-4 | 0.842 | 2.62 |
| 140 | 6.0e-4 | 0.632 | 1.73 |
| 160 | 5.9e-4 | 0.594 | 1.52 |
| **180** | **5.9e-4** | **1.831** | **7.19** |
| 200 | 5.8e-4 | 0.600 | 2.18 |

Two spikes, at step 80 as lr reached its 6e-4 plateau and again at 180. Both
recovered within 20 steps. **The envelope is contracting** (3.717 → 1.831) and
`sim_forecast_nll`, the quantity that actually matters here, is monotonically
improving through the spikes: 184 → 1.52 best so far.

Read as noisy progress on a heterogeneous mixture, not instability.

## A confound I introduced, stated plainly

*(Superseded by the restart above — retained because it is the evidence that
motivated it, and because the reasoning below is the reasoning that got it
wrong.)*

The stage learning rates (Stage I 6e-4, II 4e-4, III 3e-4, IV 2e-4, V 1e-4) were
written when the config had `batch: 192`. I cut the batch to **64** for device
memory reasons and **did not revisit the learning rates**. That is a 3× reduction
in batch at unchanged step size — 3× noisier gradient estimates per update. Under
the linear scaling rule Stage I would be ~2e-4; under the sqrt rule usually
preferred for Adam, ~3.5e-4. The configured 6e-4 is above both.

**Why I did not restart to fix it:**

1. `stage.grad_clip = 1.0` is active (`train.py:533`). Gradients are clipped to
   unit norm before the step, so no single hard batch can produce a divergent
   update at this lr. The spikes are loss spikes, not optimiser blow-ups.
2. The spike envelope is contracting and `sim_forecast_nll` is improving through
   them. The observed behaviour is not divergence.
3. Restarting on two recoverable spikes would cost the run for a hypothesis the
   evidence does not support.

**It remains a real confound and is reported as one:** this run's Stage I is
trained at a learning rate chosen for a batch three times larger. If Stage I
results are weak, that is a live candidate explanation and must not be
attributed to the architecture without testing it.

### Pre-committed intervention trigger

**Live and unchanged on the rescaled run.** A rescale is not a guarantee, and
the thresholds below are worth more than my judgement at 4 a.m. with a
half-trained model in front of me. Recorded *before* the data exists, so the
decision is not made post-hoc:

I will stop and rescale the learning rates if **any** of the following holds:

- any logged `loss` or `sim_forecast_nll` is non-finite;
- the running minimum of `sim_forecast_nll` fails to fall below **1.0** by the
  end of Stage I (step 900);
- a spike exceeds **10×** the running floor, or the spike envelope stops
  contracting over any 300-step window;
- Stage II or later shows the same spike pattern *with a rising* envelope.

> ### ⚠ CONDITION 3 FIRED at step 500 (observed 02:00, 2026-08-06)
>
> Recorded here because a trigger that fires and is only discussed in chat has
> not really been honoured.
>
> - **"spike exceeds 10× the running floor"** — step 80, `sim_forecast_nll`
>   **20.980** vs running floor **1.990** = **10.54×**. Over threshold.
> - **"envelope stops contracting over any 300-step window"** — window 200→500:
>   6.91, 8.33, 9.95, 8.34, 7.81. Starts 6.91, ends 7.81. **Not contracting.**
>
> **The run was not stopped.** That is a judgement call made against my own
> written commitment, escalated to `main` rather than taken unilaterally, and it
> is flagged as a call rather than a reading.
>
> The argument for not stopping: the step-80 spike occurs in **both** runs —
> 20.943 at lr 6.0e-4, 20.980 at lr 3.46e-4, i.e. **11.6× and 10.54× against
> their respective floors.** The trigger fires at both learning rates, so it is
> detecting a reproducible hard batch, not lr instability, and the prescribed
> remedy (rescale) is the one action already shown not to remove it.
>
> **The argument against trusting that argument:** I pre-committed a threshold,
> it fired inconveniently, and I immediately produced a reason it was malformed.
> That is indistinguishable from motivated reasoning by introspection. The
> reasoning may be correct *and* I am not the party who should rule on it —
> which is why condition 3 is not being amended without sign-off.
>
> This is instance 8 of the register's pattern, and the most uncomfortable one:
> **a guard I wrote specifically to protect against my own bias turned out
> unable to discriminate between the two hypotheses it existed to separate.**
> Writing a threshold down is necessary and not sufficient; it also has to be
> able to come out differently under the two worlds it is adjudicating.
>
> **Condition 2 is untouched and is the one that matters.** Running-min
> `sim_forecast_nll` = **1.534**, flat since step 360, needing < 1.0 by step 900.
> I have and want no discretion over it.

### Disposition of condition 3 — fired, overridden, superseded

Recorded in this order deliberately. **The old trigger is not amended.** A
trigger that quietly becomes a better trigger the moment it fires is worthless,
however good the replacement is.

1. **Condition 3 FIRED at step ~500.** No qualifier.
2. **The prescribed response (stop and rescale) was overridden by the
   coordinator.** This is an override, not a re-reading of the text. `main` made
   the call and holds the error if it is wrong.
3. **Reason: the trigger is rate-invariant and therefore cannot detect what it
   was written to detect.** The step-80 spike is 11.6× the floor at lr 6.0e-4 and
   10.54× at lr 3.46e-4 — it fires at both rates, so it does not discriminate lr
   instability from a reproducible hard batch, and the remedy it prescribes has
   already been applied once without removing it. **This is verifiable by a third
   party from two runs' logs, which is precisely why it is admissible as grounds
   for override.** An argument that rested on my judgement would not be.
4. **Condition 3b applies from step 500 forward** (supersedes, does not replace):

   > A spike is evidence of learning-rate instability only if its magnitude
   > differs **between runs at matched steps**. Absolute multiples of the running
   > floor are not admissible, because they are rate-invariant here. Compare
   > `sim_forecast_nll` at matched steps across the two LR configurations; if the
   > rescaled run's spikes are systematically larger, that is instability.

> #### ⚠ Condition 3b cannot fire either — the successor inherited the defect
>
> Found by checking which triggers are actually live rather than assuming.
> **The artifact is currently running with no operational spike-detection clause.**
>
> 3b says a spike counts as learning-rate instability *only if its magnitude
> differs between runs at matched steps*. The only runs available for that
> comparison — runs 1 and 2, the two LR configurations — are **both
> pre-normaliser-fix**. Comparing run 3 against them conflates LR with the
> normaliser change; comparing runs 1 and 2 with each other judges a pipeline
> that no longer exists.
>
> **So 3b requires two things on a common scale, and the normaliser fix removed
> the common scale** — the identical defect that killed the a-fortiori argument
> above, in a clause written to replace a guard that had already failed the same
> way once.
>
> **I am not writing a third version.** I authored both previous attempts and
> both were structurally incapable of doing their job; a third from the same
> source is not the fix. Referred to `main` and 🛡️ Popper, who is already
> designing the Stage II bar as a matched control and may find this adjacent.
>
> Reported, not repaired, and the gap is stated plainly: **no live spike trigger
> on this run.**

**Standard for any future amendment**, from `main`'s ruling and worth keeping:
an argument for superseding a fired trigger is admissible only if it is
**checkable by someone else against data the author did not choose.** "It fired
and I think it is malformed" is never sufficient on its own.

### Condition 2 — and the coordinator's pre-committed response

Condition 2 stands exactly as written: running-min `sim_forecast_nll` < 1.0 by
step 900. Currently **1.534**, flat since step 360. I have no discretion over it.

`main` has pre-committed the response **before it resolves**, so neither party
can choose after the fact:

- If condition 2 **fails**, we do **not** rescale. Rate-invariance is
  established, and a third application of the same remedy would be the reflex
  already named in this file.
- **Training continues through Stages II–V regardless.** A complete artifact with
  a documented Stage I failure is more useful than a truncated one, and **G5
  cannot be tested at all without Stage V.**
- The final report states plainly that **Stage I did not meet its own
  preregistered quality bar**, with the number. That is a finding about *this
  model, this corpus and this budget* — **not a verdict on the architecture**,
  and it must not be written as one.
- 🛡️ Popper adjudicates whether the bar was appropriate in the first place.
  Neither I nor `main` rules on that.

### ~~The spikes are periodic~~ — RETRACTED, failed its first out-of-sample test

**Claimed and withdrawn within four minutes.** Kept visible rather than deleted,
because this is the third time in one session I have overclaimed in the same
direction and the pattern matters more than the claim.

The claim was: spikes at 80, 180, 220, 320, 380, 440, 500, gaps
100, 40, 100, **60, 60, 60**, therefore periodic with period 60 — "structure in
the data ordering, not random hard batches".

**The test.** Period 60 from step 320 predicts the next spike at **560**.
Observed: a spike at **540**, and none at 560. The prediction failed at the first
opportunity.

**What the data actually says.** Over sampled steps ≥ 60: **8 spikes in 26
samples = 31 % of sampled steps** have `sim_forecast_nll` > 5. Gaps are
100, 40, 100, 60, 60, 60, 40 — mean 65.7, range 40–100, and every gap is a
multiple of 20 *by construction* because that is the logging grid. At a per-sample
spike probability of 0.31, a run of three equal gaps is entirely unremarkable.
**I fitted a period to a run of three, then called it structure.**

**The finding that survives, and it is still worth having:** spikes are
**frequent and irregular**, not periodic. Roughly **a third of logged steps** show
forecast NLL more than 3× the running floor. That is a real property of training
on this mixture and it does have a training cost — but it is a *rate*, not a
*period*, and it does not imply the ordering structure I claimed.

Whether it belongs in `corpus_composition.md` as mechanism C now depends on the
batch-composition analysis, not on the timing pattern. Deferred to that, per the
pre-commitment.

**The methodological point I keep relearning:** the aliasing caveat I attached
was correct and insufficient. I noted the grid could not resolve the period, and
then reported the period anyway. **A caveat that does not change the claim is
decoration.** The thing that actually settled it was a forward prediction with a
falsifiable target, which cost four minutes and should have come first.

Per `main`'s ruling, the honest instrument change if this is worth resolving is
to **log every step for a bounded window** rather than infer sub-grid structure
from a grid-limited series. Not done on this run — it would require restarting a
job that is now 500+ steps in, for a question that is not blocking.

Absent those, the run continues to completion at the configured rates and the
lr/batch mismatch is reported as a limitation rather than silently patched.

## Throughput

Highly variable with fleet load, on a shared machine:

- ~2.4–2.8 s/step when quiet (steps 160–200)
- ~5.4 s/step under load average 26 (steps 40–80)
- ~5.1 s/step averaged over the first 200 steps

At 2.8 s/step the full 8,700 steps is ~6.8 h; at 5.3 s/step it is ~12.8 h against
a 12 h cap. If the cap is reached the trainer stops gracefully and `resume: true`
continues from the last checkpoint, so the five-stage design is preserved either
way.

## Memory

`gpu_reserved_gb` = 33.31, flat from step 20 onward, against the 40 GB device cap.
`nvidia-smi` agrees at ~34,390 MiB. No creep across 200 steps.
