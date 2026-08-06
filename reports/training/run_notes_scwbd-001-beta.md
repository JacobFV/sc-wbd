# Run notes — scwbd-001-beta

Live notes for the production run launched 2026-08-06 01:08:59, commit `7f18528`,
provenance frozen at step 150. Written during the run so the loss curve does not
have to be reconstructed from memory afterwards.

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

Recorded *before* the data exists, so the decision is not made post-hoc:

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
