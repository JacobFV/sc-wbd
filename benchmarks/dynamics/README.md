# Dynamics benchmarks — platform notes

## A cgroup `MemoryMax` does **not** bound CUDA memory on the GB10

This is the single most important operational fact in this directory, and it
cost the fleet a machine reboot. It is written down here so it outlives the
session that discovered it.

`systemd-run --user --scope -p MemoryMax=NG` bounds **host** allocations only.
On this platform CUDA device allocations are not charged to the cgroup memory
controller, even though the GB10 has one *unified* ~121 GiB pool where the GPU
and the CPU draw on the same physical DRAM.

Two independent measurements:

| observation | cgroup cap | cgroup `memory.current` | actual device memory |
|---|---|---|---|
| 6 GiB CUDA tensor in a fresh scope | 4 GiB | 297 MB → 396 MB (**+99 MB**) | 6144 MB, **not killed** |
| `throughput.py` batch 8192, N=400 | 14 GiB | — (host RSS flat at 1054 MB) | **17006 MB peak, not killed** |
| a training run elsewhere in the fleet | 40 GiB | 8.17 GB (peak 9.86 GB) | **97.9 GB, cap never fired** |

The cgroup reading looks reassuring the entire time. That is the trap: the
number you are watching is not the number that can take the machine down.

### What to do instead

1. **Do not size GPU work against `MemoryMax`.** Verify with
   `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`, which reports
   real per-process device usage. Note it reports *reserved* memory — PyTorch's
   caching allocator holds freed blocks rather than returning them, so a large
   number is often allocator growth rather than live tensors.
2. **Bound the device pool in-process.** `throughput.py --max-cuda-gb N` calls
   `torch.cuda.set_per_process_memory_fraction(N / total)`. This is the only
   thing here that actually caps device memory, and it converts an overshoot
   into a catchable `torch.cuda.OutOfMemoryError` instead of machine-wide
   pressure. Verified: under a 1 GiB pool the batch-1024 point returns
   `status="oom"` and the process survives to record the boundary.
3. **Use both.** The cgroup still bounds numpy, the page cache and host-side
   leaks; the fraction bounds the device. Neither substitutes for the other.

For a 14 GB budget of the 121.6 GiB pool the fraction is ≈ 0.115.

## Running the throughput sweep

```sh
systemd-run --user --scope -p MemoryMax=14G -p MemorySwapMax=2G -- \
    .venv/bin/python benchmarks/dynamics/throughput.py \
        --max-cuda-gb 14 --max-batch 1024 --json reports/dynamics/throughput.json
```

* The JSON is rewritten after **every** point. An earlier version wrote it only
  at the end of the sweep, so the run killed by the OOM reboot left nothing at
  all on disk. Never restore that behaviour.
* A point that does not fit is recorded with `status="oom"` rather than crashing
  the sweep — a documented memory boundary is a legitimate benchmark result.
* `--probe-oom` deliberately attempts an oversized batch last, to find the
  device-side ceiling of *your* budget. Run it with `--max-cuda-gb` set, so it
  probes your allocation rather than the whole machine.

## Reading the numbers

Throughput on this box is **strongly affected by co-tenancy**. In the first
sweep the identical configuration (`wilson_cowan`, N=400, B=256, dt=1 ms, 500
steps) appeared four times and spanned **8.45 s to 22.51 s — a 2.66× spread**,
purely from other agents sharing the GPU. Compare rows *within* one contiguous
block, and treat absolute trajectories/second as valid only for the machine
state in which they were taken. `device.memory_model` in the JSON records this.

Peak memory is dominated by the `(B, E, C)` edge-gather intermediate
(`edge_mem_mb`), not by the state vector: measured peak runs ≈ 20× that term,
since Heun evaluates the drift twice and each delayed read materialises two taps
plus the interpolation. Memory therefore scales with `batch × n_edges`, which is
why N=1600 (383k edges) costs 8 GB at batch 256 while N=400 costs 536 MB.
