# Run notes — scwbd-001-beta

Live notes, written during the run so the loss curve does not have to be
reconstructed from memory afterwards.

**The artifact is the run launched 2026-08-06 01:35 with sqrt-scaled learning
rates.** An earlier attempt (commit `7f18528`, 01:08:59) was stopped at step 260
and is superseded — see "The restart" below. Everything in the tables further
down describes that superseded attempt and is retained as the evidence that
motivated the rescale.

## The restart

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
