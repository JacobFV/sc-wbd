"""Throughput and memory benchmark for the dynamics core on the GB10.

Reports **trajectories/second** and **simulated-seconds/second** as a function of
(backend, n_regions, batch, dt), plus a memory profile.  The batch axis is the
parameter axis: the point of the design is that a thousand parameter sets
integrate in one launch, so the interesting number is not "steps/s" but
"trajectory-seconds of brain time per wall-clock second".

**Memory model.**  The GB10 has *one* unified ~121 GiB pool: CUDA allocations,
numpy and the page cache all draw on the same physical memory, and
``torch.cuda.get_device_properties().total_memory`` is not a second budget on top
of what ``free -h`` reports.

**A cgroup cap does not bound CUDA memory here.**  Measured on this box: under
``systemd-run --user --scope -p MemoryMax=4G``, allocating a 6 GiB CUDA tensor
succeeds and moves the cgroup's ``memory.current`` by only ~99 MB.  The batch-8192
point below reached a 17 GB CUDA peak under a nominal 14 GB cap without being
killed.  So::

    systemd-run --user --scope -p MemoryMax=14G ...   # bounds host/numpy only
    --max-cuda-gb 14                                  # bounds the CUDA pool

Use **both**: the cgroup for host allocations, ``--max-cuda-gb`` (which calls
``torch.cuda.set_per_process_memory_fraction``) for device allocations.  Relying
on the cgroup alone leaves the unified pool exposed to exactly the runaway that
an OOM reboot is made of.

``--max-batch`` (default 1024) bounds the batch axis for the same reason.  A
point that does not fit is recorded with ``status="oom"`` rather than crashing
the run, and the JSON is rewritten after *every* point, so an interrupted sweep
still leaves its completed measurements on disk.

Run::

    .venv/bin/python benchmarks/dynamics/throughput.py            # standard sweep
    .venv/bin/python benchmarks/dynamics/throughput.py --quick
    .venv/bin/python benchmarks/dynamics/throughput.py --json out.json
    .venv/bin/python benchmarks/dynamics/throughput.py --probe-oom  # find the ceiling
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scwbd.dynamics import (  # noqa: E402
    DelayedConnectome,
    EdgeSet,
    JansenRit,
    Kuramoto,
    LearnedNeuralOperator,
    LinearGaussian,
    ReducedWongWang,
    SimConfig,
    StuartLandau,
    WholeBrainSimulator,
    WilsonCowan,
)

BACKENDS = {
    "wilson_cowan": WilsonCowan,
    "jansen_rit": JansenRit,
    "wong_wang": ReducedWongWang,
    "stuart_landau": StuartLandau,
    "kuramoto": Kuramoto,
    "linear_gaussian": LinearGaussian,
    "learned_operator": lambda: LearnedNeuralOperator(state_dim=2, width=64),
}


@dataclass
class Row:
    backend: str
    n_regions: int
    batch: int
    dt: float
    n_steps: int
    density: float
    n_edges: int
    method: str
    wall_s: float
    steps_per_s: float
    traj_per_s: float
    sim_seconds_per_s: float
    peak_mem_mb: float
    state_mem_mb: float
    buffer_mem_mb: float
    #: (B, E, C) gather intermediate -- the dominant allocation at large B or N
    edge_mem_mb: float = 0.0
    #: Peak host RSS (VmHWM) of the process.  MEASURED CAVEAT: on this platform
    #: this tracks only the CPU-side allocations -- it sat at ~1.05 GB across the
    #: whole sweep while ``peak_mem_mb`` ranged from 3 MB to 8 GB, so the CUDA
    #: pool does not appear in VmHWM even though the memory is physically
    #: unified.  Use ``peak_mem_mb`` for the simulation footprint and read this
    #: as interpreter + library overhead only; the two must still be *added*
    #: when sizing against a cgroup cap, because they draw on one pool.
    host_peak_rss_mb: float = 0.0
    status: str = "ok"
    error: str = ""


def _host_peak_rss_mb() -> float:
    """VmHWM: peak resident set size, i.e. what MemoryMax actually counts."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def _free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def bench_one(
    name: str,
    *,
    n_regions: int,
    batch: int,
    dt: float,
    n_steps: int,
    density: float = 0.15,
    method: str = "heun",
    device: str = "cuda",
    seed: int = 0,
    record_every: int = 10,
) -> Row:
    dev = torch.device(device)
    be = BACKENDS[name]().to(dev)
    edges = EdgeSet.random(n_regions, density=density, seed=seed, device=dev)
    con = DelayedConnectome(edges, mode=be.coupling_kind, n_channels=be.n_coupling_channels)
    sim = WholeBrainSimulator(be, con)
    theta = be.sample_theta(batch, n_regions, seed=seed, device=dev)
    theta.set("velocity", torch.full((batch, 1), 5.0, device=dev))
    if name == "stuart_landau":
        theta.set("row_sum", con.row_sum().expand(batch, -1))

    cfg = SimConfig(dt=dt, n_steps=n_steps, seed=seed, record_every=record_every, method=method)
    # warm up kernels and allocator, then measure
    sim.run(theta, SimConfig(dt=dt, n_steps=8, seed=seed, record_every=8))
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    res = sim.run(theta, cfg)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    peak = torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else 0.0
    state_mb = batch * n_regions * be.state_dim * 4 / 2**20
    x, buf, _ = sim.prepare(theta, cfg)
    buf_mb = buf.memory_bytes() / 2**20 if buf is not None else 0.0
    # one (B, E, C) gather per drift evaluation; Heun does two, and each read
    # materialises v0, v1 and the interpolation -- this term, not the state, is
    # what sets the memory ceiling.
    edge_mb = batch * edges.n_edges * be.n_coupling_channels * 4 / 2**20
    del x, buf, res
    sim_seconds = n_steps * dt
    return Row(
        backend=name,
        n_regions=n_regions,
        batch=batch,
        dt=dt,
        n_steps=n_steps,
        density=density,
        n_edges=edges.n_edges,
        method=method,
        wall_s=wall,
        steps_per_s=n_steps / wall,
        traj_per_s=batch / wall,
        sim_seconds_per_s=batch * sim_seconds / wall,
        peak_mem_mb=peak,
        state_mem_mb=state_mb,
        buffer_mem_mb=buf_mb,
        edge_mem_mb=edge_mb,
        host_peak_rss_mb=_host_peak_rss_mb(),
    )


def bench_guarded(name: str, **kw) -> Row:
    """Run one point, converting an OOM into a recorded result rather than a crash.

    A point that does not fit is a measurement, not an error: the batch at which
    this trips is the documented memory boundary of the dynamics core.
    """
    try:
        return bench_one(name, **kw)
    except (torch.cuda.OutOfMemoryError, MemoryError, RuntimeError) as exc:
        msg = str(exc)
        if not isinstance(exc, RuntimeError) or "out of memory" in msg.lower():
            status = "oom"
        else:
            raise
        _free_cuda()
        return Row(
            backend=name, n_regions=kw.get("n_regions", 0), batch=kw.get("batch", 0),
            dt=kw.get("dt", 0.0), n_steps=kw.get("n_steps", 0), density=kw.get("density", 0.15),
            n_edges=0, method=kw.get("method", "heun"), wall_s=float("nan"),
            steps_per_s=float("nan"), traj_per_s=float("nan"), sim_seconds_per_s=float("nan"),
            peak_mem_mb=float("nan"), state_mem_mb=0.0, buffer_mem_mb=0.0,
            host_peak_rss_mb=_host_peak_rss_mb(), status=status, error=msg.split("\n")[0][:300],
        )


def device_info() -> dict:
    info = {"platform": platform.platform(), "torch": torch.__version__, "python": sys.version.split()[0]}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info |= {
            "gpu": p.name,
            "sm": f"{p.major}.{p.minor}",
            "total_memory_mb": p.total_memory / 2**20,
            "multiprocessors": p.multi_processor_count,
        }
    return info


def print_table(rows: list[Row], header: bool = True) -> None:
    hdr = (
        f"{'backend':<17}{'N':>5}{'B':>7}{'edges':>8}{'dt':>8}{'steps':>7}"
        f"{'wall_s':>9}{'steps/s':>10}{'traj/s':>9}{'sim_s/s':>10}{'peakMB':>9}"
        f"{'rssMB':>9}  {'status'}"
    )
    if header:
        print(hdr)
        print("-" * len(hdr))
    for r in rows:
        print(
            f"{r.backend:<17}{r.n_regions:>5}{r.batch:>7}{r.n_edges:>8}{r.dt:>8.4f}{r.n_steps:>7}"
            f"{r.wall_s:>9.3f}{r.steps_per_s:>10.1f}{r.traj_per_s:>9.1f}"
            f"{r.sim_seconds_per_s:>10.1f}{r.peak_mem_mb:>9.1f}{r.host_peak_rss_mb:>9.1f}  {r.status}",
            flush=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", type=str, default="")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-batch", type=int, default=1024,
                    help="largest batch in the batch sweep (memory cap; see module docstring)")
    ap.add_argument("--probe-oom", action="store_true",
                    help="after the capped sweep, attempt one oversized batch to record where it breaks")
    ap.add_argument("--max-cuda-gb", type=float, default=0.0,
                    help="cap this process's CUDA pool (GiB).  MEASURED: a systemd-run "
                         "MemoryMax cgroup does NOT bound CUDA allocations on this box -- a 6 GiB "
                         "tensor allocates fine under a 4 GiB cap and moves memory.current by only "
                         "~99 MB -- so this in-process limit is the only thing that actually caps "
                         "GPU memory, and it turns an overshoot into a catchable OutOfMemoryError.")
    args = ap.parse_args()

    if args.max_cuda_gb > 0 and torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_cuda_gb / total_gb))
        print(f"CUDA pool capped at {args.max_cuda_gb:g} GiB of {total_gb:.1f} GiB", flush=True)

    rows: list[Row] = []
    info = device_info()
    info["memory_model"] = (
        "unified: the CUDA pool and the host allocator are the same physical memory, "
        "so peak_mem_mb and host_peak_rss_mb draw on one budget, not two"
    )
    print(json.dumps(info, indent=2), flush=True)

    def emit(r: Row) -> None:
        """Append a row and flush the JSON immediately.

        The previous version wrote the file only after the whole sweep, so the run
        that was killed mid-sweep left nothing at all behind.  Now every point is
        durable the moment it is measured.
        """
        rows.append(r)
        print_table([r], header=False)
        if args.json:
            Path(args.json).write_text(
                json.dumps({"device": info, "complete": False, "rows": [asdict(x) for x in rows]}, indent=2)
            )
        _free_cuda()

    if args.quick:
        for name in ("wilson_cowan", "linear_gaussian"):
            emit(bench_guarded(name, n_regions=200, batch=128, dt=1e-3, n_steps=200, device=args.device))
        return

    batches = [b for b in (1, 8, 64, 256, 1024, 4096) if b <= args.max_batch]
    print(f"\n== batch scaling (wilson_cowan, N=400, dt=1 ms, 500 steps), capped at B={args.max_batch} ==")
    print_table([], header=True)
    for batch in batches:
        emit(bench_guarded("wilson_cowan", n_regions=400, batch=batch, dt=1e-3, n_steps=500,
                           device=args.device))

    print("\n== region scaling (wilson_cowan, B=256, dt=1 ms, 500 steps) ==")
    print_table([], header=True)
    for n in (100, 200, 400, 800, 1600):
        emit(bench_guarded("wilson_cowan", n_regions=n, batch=256, dt=1e-3, n_steps=500,
                           device=args.device))

    print("\n== backend comparison (N=400, B=256, dt=1 ms, 500 steps) ==")
    print_table([], header=True)
    for name in BACKENDS:
        emit(bench_guarded(name, n_regions=400, batch=256, dt=1e-3, n_steps=500, device=args.device))

    print("\n== step size (wilson_cowan, N=400, B=256, 1 s of brain time) ==")
    print_table([], header=True)
    for dt in (2e-4, 5e-4, 1e-3, 2e-3):
        emit(bench_guarded("wilson_cowan", n_regions=400, batch=256, dt=dt,
                           n_steps=int(round(1.0 / dt)), device=args.device))

    print("\n== integrator (wilson_cowan, N=400, B=256, dt=1 ms, 500 steps) ==")
    print_table([], header=True)
    for method in ("euler_maruyama", "heun", "milstein", "srk"):
        emit(bench_guarded("wilson_cowan", n_regions=400, batch=256, dt=1e-3, n_steps=500,
                           method=method, device=args.device))

    if args.probe_oom:
        print("\n== oversized-batch probe (expected to fail; the failure IS the result) ==")
        print_table([], header=True)
        emit(bench_guarded("wilson_cowan", n_regions=400, batch=4096, dt=1e-3, n_steps=500,
                           device=args.device))

    print("\n== all rows ==")
    print_table(rows)
    if args.json:
        Path(args.json).write_text(
            json.dumps({"device": info, "complete": True, "rows": [asdict(r) for r in rows]}, indent=2)
        )
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
