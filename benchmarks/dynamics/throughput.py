"""Throughput and memory benchmark for the dynamics core on the GB10.

Reports **trajectories/second** and **simulated-seconds/second** as a function of
(backend, n_regions, batch, dt), plus a memory profile.  The batch axis is the
parameter axis: the point of the design is that a thousand parameter sets
integrate in one launch, so the interesting number is not "steps/s" but
"trajectory-seconds of brain time per wall-clock second".

Run::

    .venv/bin/python benchmarks/dynamics/throughput.py            # standard sweep
    .venv/bin/python benchmarks/dynamics/throughput.py --quick
    .venv/bin/python benchmarks/dynamics/throughput.py --json out.json
"""

from __future__ import annotations

import argparse
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


def print_table(rows: list[Row]) -> None:
    hdr = (
        f"{'backend':<17}{'N':>5}{'B':>7}{'edges':>8}{'dt':>8}{'steps':>7}"
        f"{'wall_s':>9}{'steps/s':>10}{'traj/s':>9}{'sim_s/s':>10}{'peakMB':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r.backend:<17}{r.n_regions:>5}{r.batch:>7}{r.n_edges:>8}{r.dt:>8.4f}{r.n_steps:>7}"
            f"{r.wall_s:>9.3f}{r.steps_per_s:>10.1f}{r.traj_per_s:>9.1f}"
            f"{r.sim_seconds_per_s:>10.1f}{r.peak_mem_mb:>9.1f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", type=str, default="")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    rows: list[Row] = []
    info = device_info()
    print(json.dumps(info, indent=2))

    if args.quick:
        for name in ("wilson_cowan", "linear_gaussian"):
            rows.append(bench_one(name, n_regions=200, batch=128, dt=1e-3, n_steps=200, device=args.device))
        print_table(rows)
        return

    print("\n== batch scaling (wilson_cowan, N=400, dt=1 ms, 500 steps) ==")
    for batch in (1, 8, 64, 256, 1024, 4096):
        rows.append(bench_one("wilson_cowan", n_regions=400, batch=batch, dt=1e-3, n_steps=500,
                              device=args.device))
        print_table(rows[-1:])

    print("\n== region scaling (wilson_cowan, B=256, dt=1 ms, 500 steps) ==")
    for n in (100, 200, 400, 800, 1600):
        rows.append(bench_one("wilson_cowan", n_regions=n, batch=256, dt=1e-3, n_steps=500,
                              device=args.device))
        print_table(rows[-1:])

    print("\n== backend comparison (N=400, B=256, dt=1 ms, 500 steps) ==")
    for name in BACKENDS:
        rows.append(bench_one(name, n_regions=400, batch=256, dt=1e-3, n_steps=500, device=args.device))
        print_table(rows[-1:])

    print("\n== step size (wilson_cowan, N=400, B=256, 1 s of brain time) ==")
    for dt in (2e-4, 5e-4, 1e-3, 2e-3):
        rows.append(bench_one("wilson_cowan", n_regions=400, batch=256, dt=dt,
                              n_steps=int(round(1.0 / dt)), device=args.device))
        print_table(rows[-1:])

    print("\n== integrator (wilson_cowan, N=400, B=256, dt=1 ms, 500 steps) ==")
    for method in ("euler_maruyama", "heun", "milstein", "srk"):
        r = bench_one("wilson_cowan", n_regions=400, batch=256, dt=1e-3, n_steps=500,
                      method=method, device=args.device)
        rows.append(r)
        print_table([r])

    print("\n== all rows ==")
    print_table(rows)
    if args.json:
        Path(args.json).write_text(json.dumps({"device": info, "rows": [asdict(r) for r in rows]}, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
