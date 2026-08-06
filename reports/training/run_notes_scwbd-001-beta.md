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
