# Platform memory limits on the GB10 (unified memory)

Status: **verified by measurement, 2026-08-06.** Applies to every agent running
GPU work on this machine, not just `scwbd.foundation`.

## The one-sentence version

`systemd-run --user --scope -p MemoryMax=NG` **does not bound CUDA allocations
on this box**, and the cgroup will report a small, reassuring number the entire
time it is not enforcing anything.

## What was measured

A production training run (`scwbd.foundation.train`, `configs/scwbd_001_beta.yaml`)
was launched inside `systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=4G`.

| instrument | reading |
|---|---|
| `nvidia-smi --query-compute-apps=pid,used_memory` | **100,266 MiB (97.9 GB)** |
| the run's cgroup `memory.current` | **8.17 GB** |
| the run's cgroup `memory.peak` | 9.86 GB |
| `MemoryMax` on that cgroup | 40 GB — never fired |
| sum of *all* process RSS on the machine | 12.4 GB |
| `free -g` available | fell 90 GB → **8 GB** in ~6 minutes |

Killing that single process returned available memory from 13 GB to **115 GB**.
The attribution is not ambiguous.

## Why

The GB10 has **one** physical pool. `torch.cuda.get_device_properties().total_memory`
(130.6e9 bytes) and `free -h` (121 GiB) are the same memory in different units.
But the device allocation path does not go through the host page accounting the
memory cgroup charges against, so:

- device memory does **not** appear in any process's RSS;
- device memory is **not** charged to `memory.current`;
- therefore `MemoryMax` cannot see it, and cannot bound it.

Compounding this, PyTorch's caching allocator has **no default ceiling**. It
reserves and retains freed blocks rather than returning them, so on a shared
pool it grows monotonically until something on the machine dies. *Reserved* is
the machine's exposure; *allocated* is only what the model is using right now.

## What actually works

`torch.cuda.set_per_process_memory_fraction(fraction, device_index)` — this
bounds the caching allocator itself, which is the thing doing the allocating.

Wrapped as `scwbd.foundation.util.cap_cuda_reserve(device, limit_gb)` and wired
into `FoundationTrainer.__init__` before the first allocation, driven by
`train.cuda_reserve_gb` in the config.

**Verified adversarially, not assumed.** Cap set to 4 GB, then 16 GB demanded
1 GB at a time:

```
[mem] CUDA reserve capped at 4.0 GB (fraction=0.033 of 121.6 GB unified pool)
  +1GB (#1)  reserved=1.00GB  smi=121240, 1194 MiB
  +1GB (#2)  reserved=2.00GB  smi=121240, 2218 MiB
  +1GB (#3)  reserved=3.00GB  smi=121240, 3242 MiB
  +1GB (#4)  reserved=4.00GB  smi=121240, 4266 MiB
OOM raised after 4 GB -- the cap FIRED
peak reserved = 4.00 GB   cap = 4.0 GB
VERDICT: PASS
```

The reproducer is worth keeping: a guard that has never been made to fire is
not known to work. `torch.device("cuda")` without an index raises here, which
the adversarial test caught before the fix shipped.

## Guidance

1. **Do not size a GPU job against `MemoryMax`.** It bounds host-side
   allocation only. Keep it — it still bounds DataLoader workers, pinned
   buffers and numpy — but do not treat it as protection from CUDA.
2. **Set a device-side cap explicitly** on any job that touches the GPU.
3. **Verify with `nvidia-smi --query-compute-apps=pid,used_memory`**, not with
   `free`, not with RSS, and not with the cgroup. Those three will all look
   fine while the machine is minutes from death.
4. If the job does not fit under its cap, **cut the batch, do not raise the
   cap.** The cap is the budget; the batch is the variable.
5. `nvidia-smi --query-gpu=memory.used` returns `[N/A]` on this platform.
   `--query-compute-apps` works. Use it.

## Sizing the batch: what the working set actually costs

Measured with `gpu_reserved_gb` sampled at step 1 and again at plateau:

| batch | step 1 | plateau | verdict |
|---|---|---|---|
| 64 | 17.92 GB | **33.3 GB** (flat at steps 20 and 40) | fits, 6.7 GB headroom |
| 128 | 35.27 GB | hit the 40 GB cap | does not fit |
| 192 | not logged | 97.9 GB (the uncapped run) | does not fit |

≈0.27 GB/sample at step 1, growing ≈1.86× to plateau. The model reproduces the
uncapped run's 97.9 GB at batch 192, which is why it is now trusted. Batch 128
would need ≈66 GB and batch 192 ≈98 GB, so **64 is the largest batch that fits a
40 GB budget.**

### The measurement error that cost two wrong estimates

The first two batch estimates were wrong, and both came from reading
`allocated by PyTorch` out of the OOM message. **At OOM that number always sits
at the ceiling, whatever the batch was** — 38.93 GB at batch 192, 39.70 GB at
batch 128, both against a 40 GB cap. It looks like a measurement of the working
set and is actually a measurement of the cap.

From it I inferred first that memory was batch-linear at 0.203 GB/sample, then —
when the 33% cut from 192 to 128 barely moved the number — that the working set
was batch-*independent*. Both conclusions were drawn from a statistic that could
not have distinguished them. The working set is in fact close to batch-linear.

The fix is to log `gpu_reserved_gb` at step 1, before the allocator has grown to
fill whatever room it is given.

**An instrument that always reads the same value cannot discriminate between
hypotheses.** Reasoning confidently from one is worse than having no instrument,
because the output has the shape of evidence.

## Throughput is latency-bound, not throughput-bound

Per-step wall time is **independent of batch size**: 4.6 s/step at batch 192 and
4.3-4.7 s/step at batch 64, measured on a quiet machine across steps 1→20 and
20→40.

This is an architectural property, not a tuning artifact. The forward pass
integrates 72 sequential model steps (48 forecast + 24 assimilation context)
through delayed long-range coupling. Each step depends on the last, so the
critical path is a chain of 72 kernel launches whose length no amount of
parallel work shortens. The GPU is latency-bound on that chain and has capacity
to spare *within* each step.

The consequence for budgeting is direct and slightly counterintuitive:

- **the memory budget buys data per step, not wall clock.** Cutting batch
  192 → 64 cost no wall-clock time per step and 3× the data: `traj_s_per_s`
  fell 13.0 → 4.3 for the same 4.6 s.
- so a run capped at 40 GB is not slower than one capped at 98 GB; it is
  *less informative per hour*.

This is what makes **gradient checkpointing** the right next lever rather than a
generic optimisation. It trades recomputation — spare capacity inside each step,
which is exactly what is idle — for memory per sample, which is the binding
constraint on batch. It should buy a larger batch inside the same 40 GB, and
therefore more data per hour at unchanged wall clock. Approved as planned work
for the next run, to be implemented and tested on its own rather than in front
of a 10 h training run.

## The general lesson

Three defects in this project have now had **the same shape: an instrument that
looked informative and could not have been.** They are worth reading together,
because the third was found by someone who had already been burned by the first
two and still walked into it.

| # | instrument | what it appeared to report | why it could not |
|---|---|---|---|
| 1 | `FOUNDATION_BINDING` glob match | which tensors a source may train | `torch.compile` renamed them to `local._orig_mod.*`; exact names matched nothing on CUDA, prefix globs still matched, every CPU test passed |
| 2 | cgroup `memory.current` vs `MemoryMax` | the job's memory footprint | it charges host pages; the allocation was on the device |
| 3 | `allocated by PyTorch` in an OOM message | the working set at that batch | at OOM it always equals the cap, for every batch |

In each case the reading was **stable, plausible, and constant with respect to
the thing being asked about.** That is the tell. A permission set that
half-applies looks enforced; a cgroup reporting 8 GB looks safe; an OOM
reporting 39 GB looks like a working-set measurement.

Two habits follow, and they are cheap:

1. **Make the guard fail on purpose before trusting it.** Every check in
   `tests/foundation/test_compiler_binding.py` is verified to fire by breaking
   a binding deliberately. The CUDA cap was verified by demanding 16 GB against
   a 4 GB ceiling — which caught a real bug in the fix (`torch.device("cuda")`
   carries no index and the call rejects it) *before* it shipped.
2. **Ask what reading would falsify the hypothesis.** If the instrument returns
   the same value under both branches — CPU and CUDA, capped and uncapped,
   batch 64 and batch 192 — it is not evidence, and confident reasoning from it
   is worse than admitting ignorance, because the output has the shape of data.

When a check and the thing it guards live in different spaces — name space,
address space, accounting space — assume it is decorative until it has failed
for you on demand.
