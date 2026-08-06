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

## The general lesson

This is the second defect in this project where **a green-looking guard was
structurally incapable of firing**, and both had the same tell: an asymmetry
between where the check ran and where the work happened.

- `torch.compile` renamed parameters to `local._orig_mod.*`, so exact-name
  gradient permissions silently matched nothing on CUDA while prefix globs kept
  matching — and every CPU test passed. (See
  `scwbd/foundation/util.py:logical_param_name` and
  `tests/foundation/test_compiler_binding.py`.)
- `MemoryMax` measured host pages while the allocation happened on the device.

In both cases the instrument reported control it was not exercising. When a
safety check and the thing it guards live in different namespaces — name space,
address space, accounting space — assume it is decorative until you have made
it fail on purpose.
